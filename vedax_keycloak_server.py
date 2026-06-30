#!/usr/bin/env python3
"""
====================================================================
  VEDAX KEYCLOAK SERVER  —  production knowledge-management agent
  with role-based UI and Keycloak (LDAP) authentication
====================================================================

ARCHITECTURE
------------
                ┌─────────────┐
   browser ────▶│  /login     │── POST username/password ──▶ Keycloak
                │  (HTML form)│                            (LDAP federated)
                └─────────────┘                              │
                       ▲                                     │
                       │   access_token (JWT)                │
                       │◀────────────────────────────────────┘
                       │
                       │   Authorization: Bearer <jwt>
                       ▼
                ┌──────────────────────────────────────┐
                │  /api/* protected endpoints           │
                │  1. verify token at Keycloak userinfo │
                │  2. resolve LOCAL role from DB        │
                │     (superuser overrides Keycloak)    │
                │  3. enforce role-based permission     │
                │  4. call vedax_core for retrieval/ask │
                └──────────────────────────────────────┘

ROLES
-----
  superuser   — full power: manage users (promote/demote/revoke/delete)
                + admin powers + ask
  admin       — manage documents (upload/delete/categorise) + ask
  user        — ONLY ask query (chat UI)

  First login from Keycloak always lands as 'user' (least privilege).
  SUPERUSER usernames (config) are auto-elevated on first login.
  After that, only superusers can change roles.

AUTO-FETCH
----------
  Drop any .pdf/.txt/.md into ./sop_docs and it is auto-indexed on
  server start.  Admins can also upload via the UI.

ZERO EXTERNAL DEPENDENCY for Keycloak — pure stdlib urllib.

RUN
---
  pip install fastapi uvicorn python-multipart
  python vedax_keycloak_server.py

  All configuration lives in ./config.yaml (or VEDAX_CONFIG env path).
  No more env-variable juggling -- everything is in one auditable
  file.  See config.example.yaml for the schema.
====================================================================
"""

import base64
import json
import os
import secrets
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import vedax_core as core
import vedax_db
from vedax_db import db, Database
import sqlite3


# ════════════════════════════════════════════════════════════════
#  CONFIG  (loaded from ./config.yaml — see config.example.yaml)
# ════════════════════════════════════════════════════════════════

import vedax_config
import vedax_guardrail

CFG = vedax_config.cfg

KEYCLOAK_URL        = CFG.get_path("keycloak.url", "http://localhost:8080")
KEYCLOAK_REALM      = CFG.get_path("keycloak.realm", "master")
KEYCLOAK_CLIENT_ID  = CFG.get_path("keycloak.client_id", "abc-nginx-manager")
KEYCLOAK_VERIFY_TLS = CFG.get_path("keycloak.verify_tls", True)
SUPERUSER_KC_ROLES  = set(CFG.get_path("keycloak.superuser_keycloak_roles") or [])
ADMIN_KC_ROLES      = set(CFG.get_path("keycloak.admin_keycloak_roles") or [])

UPLOAD_DIR = CFG.get_path("documents.upload_dir", "./uploaded_docs")
if CFG.get_path("documents.auto_fetch_dir"):
    core.AUTO_FETCH_DIR = CFG.get_path("documents.auto_fetch_dir")
if CFG.get_path("documents.critical_fetch_dir"):
    core.CRITICAL_FETCH_DIR = CFG.get_path("documents.critical_fetch_dir")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(core.AUTO_FETCH_DIR, exist_ok=True)
os.makedirs(core.CRITICAL_FETCH_DIR, exist_ok=True)

# propagate retrieval / LLM settings from config to vedax_core
if CFG.get_path("retrieval.top_k_max") is not None:
    core.TOP_K = CFG.get_path("retrieval.top_k_max")
if CFG.get_path("retrieval.chunk_tokens") is not None:
    core.CHUNK_TOKENS = CFG.get_path("retrieval.chunk_tokens")
if CFG.get_path("retrieval.overlap_tokens") is not None:
    core.OVERLAP_TOKENS = CFG.get_path("retrieval.overlap_tokens")
if CFG.get_path("retrieval.abstain_threshold") is not None:
    core.ABSTAIN_THRESHOLD = CFG.get_path("retrieval.abstain_threshold")
if CFG.get_path("retrieval.use_dense") is not None:
    core.USE_DENSE = CFG.get_path("retrieval.use_dense")
if CFG.get_path("llm.url"):
    core.LLM_URL = CFG.get_path("llm.url")
if CFG.get_path("llm.model"):
    core.LLM_MODEL = CFG.get_path("llm.model")
if CFG.get_path("llm.api"):
    core.LLM_API = CFG.get_path("llm.api")
if CFG.get_path("llm.token"):
    core.LLM_TOKEN = CFG.get_path("llm.token")

# ── Guardrail (the 5-layer safety net) ──────────────────────────
GUARD = vedax_guardrail.Guardrail(
    CFG.get_path("guardrails") or {"enabled": False},
    db_path=vedax_db.db.path,
)


# ════════════════════════════════════════════════════════════════
#  EXTRA DB TABLES  (on top of vedax_db.py — user registry)
# ════════════════════════════════════════════════════════════════
#
# We add a 'kc_users' table that maps Keycloak username -> local role.
# This is the AUTHORITATIVE role table: superuser can override anything
# Keycloak says.  is_revoked=1 blocks the user from logging in at all.

def _ensure_kc_users_table():
    c = sqlite3.connect(db.path)
    c.executescript("""
      CREATE TABLE IF NOT EXISTS kc_users (
        username      TEXT PRIMARY KEY,
        display_name  TEXT,
        role          TEXT NOT NULL DEFAULT 'user',
        is_revoked    INTEGER NOT NULL DEFAULT 0,
        first_login   REAL,
        last_login    REAL,
        promoted_by   TEXT
      );
    """)
    c.commit()
    c.close()


_ensure_kc_users_table()


def kc_user_get(username: str) -> Optional[dict]:
    c = sqlite3.connect(db.path)
    c.row_factory = sqlite3.Row
    cur = c.cursor()
    cur.execute("SELECT * FROM kc_users WHERE username = ?", (username,))
    row = cur.fetchone()
    c.close()
    return dict(row) if row else None


def kc_user_upsert(username: str, display_name: str, role: str = "pending"):
    """Insert if new (default 'pending' — must be approved by superuser).
    Existing rows are NEVER role-changed by this function; use
    kc_user_update for that."""
    now = time.time()
    c = sqlite3.connect(db.path)
    cur = c.cursor()
    cur.execute(
        """
        INSERT INTO kc_users (username, display_name, role, first_login, last_login)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
          display_name = excluded.display_name,
          last_login   = excluded.last_login
        """,
        (username, display_name, role, now, now),
    )
    c.commit()
    c.close()


def kc_role_from_keycloak_claims(claims: dict) -> str:
    """Map Keycloak realm roles (from JWT) to local roles.  Returns
    one of: 'superuser' / 'admin' / 'pending'.  An unconfigured account
    always lands in 'pending' so the superuser must approve.
    """
    realm_roles = set((claims.get("realm_access") or {}).get("roles") or [])
    client_roles = set()
    for ra in (claims.get("resource_access") or {}).values():
        client_roles.update((ra or {}).get("roles") or [])
    all_roles = realm_roles | client_roles
    if all_roles & SUPERUSER_KC_ROLES:
        return "superuser"
    if all_roles & ADMIN_KC_ROLES:
        return "admin"
    return "pending"


def kc_user_list() -> List[dict]:
    c = sqlite3.connect(db.path)
    c.row_factory = sqlite3.Row
    cur = c.cursor()
    cur.execute("SELECT * FROM kc_users ORDER BY last_login DESC")
    rows = [dict(r) for r in cur.fetchall()]
    c.close()
    return rows


def kc_user_update(username: str, *, role: Optional[str] = None,
                   is_revoked: Optional[int] = None, by: str = ""):
    sets, params = [], []
    if role is not None:
        sets.append("role = ?"); params.append(role)
        sets.append("promoted_by = ?"); params.append(by)
    if is_revoked is not None:
        sets.append("is_revoked = ?"); params.append(int(is_revoked))
    if not sets:
        return
    params.append(username)
    c = sqlite3.connect(db.path)
    c.execute(f"UPDATE kc_users SET {', '.join(sets)} WHERE username = ?", params)
    c.commit()
    c.close()


def kc_user_delete(username: str):
    c = sqlite3.connect(db.path)
    c.execute("DELETE FROM kc_users WHERE username = ?", (username,))
    c.commit()
    c.close()


# ════════════════════════════════════════════════════════════════
#  KEYCLOAK CLIENT  (pure stdlib urllib)
# ════════════════════════════════════════════════════════════════

# Bounded TTL cache for Keycloak userinfo.  WITHOUT a bound this dict
# grows forever — every login mints a fresh JWT, so a long-running
# server would accumulate one stale entry per login until it exhausts
# RAM.  We cap the size and evict expired / oldest entries.
class _TTLCache:
    def __init__(self, max_size=2000, ttl=60):
        self.max_size = max_size
        self.ttl = ttl
        self._store = {}          # key -> (value, expiry)

    def get(self, key):
        item = self._store.get(key)
        if not item:
            return None
        value, expiry = item
        if expiry < __import__("time").time():
            # lazily drop the expired entry
            self._store.pop(key, None)
            return None
        return value

    def set(self, key, value):
        import time as _t
        now = _t.time()
        # opportunistic sweep of expired entries on every write
        if len(self._store) >= self.max_size:
            self._sweep(now)
        # if still over the cap, evict the oldest-expiring entries
        if len(self._store) >= self.max_size:
            for k in sorted(self._store,
                            key=lambda k: self._store[k][1])[:self.max_size // 4]:
                self._store.pop(k, None)
        self._store[key] = (value, now + self.ttl)

    def _sweep(self, now):
        dead = [k for k, (_, exp) in self._store.items() if exp < now]
        for k in dead:
            self._store.pop(k, None)

    def __len__(self):
        return len(self._store)


_USERINFO_CACHE = _TTLCache(max_size=2000, ttl=60)
_USERINFO_TTL = 60   # seconds (kept for backward reference)


def keycloak_login(username: str, password: str) -> dict:
    """
    Hits Keycloak's token endpoint with password grant and returns the
    decoded access_token claims.  Raises HTTPException on failure.
    """
    url = (
        f"{KEYCLOAK_URL.rstrip('/')}"
        f"/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
    )
    body = urllib.parse.urlencode({
            "grant_type": "password",
            "client_id": KEYCLOAK_CLIENT_ID,
            "username": username,
            "password": password,
            "scope": "openid profile email",
        }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        ctx = _ssl_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Keycloak login failed: {e.read().decode('utf-8', 'replace')[:200]}",
        )
    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Keycloak unreachable: {e.reason}",
        )
    return data


def _ssl_context():
    if KEYCLOAK_VERIFY_TLS:
        return None     # default secure verification
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def keycloak_userinfo(access_token: str) -> dict:
    """
    Validates the token by calling /userinfo.  Cached for 60s to keep
    per-request latency low without sacrificing revocation safety much.
    """
    cached = _USERINFO_CACHE.get(access_token)
    if cached is not None:
        return cached
    url = (
        f"{KEYCLOAK_URL.rstrip('/')}"
        f"/realms/{KEYCLOAK_REALM}/protocol/openid-connect/userinfo"
    )
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=401,
                            detail=f"token invalid: {e.code}")
    except urllib.error.URLError as e:
        # Network issue — fall back to JWT payload decoding so the UI
        # does not freeze when Keycloak is briefly unreachable.
        info = _decode_jwt_payload(access_token)
        if not info:
            raise HTTPException(status_code=502,
                                detail=f"Keycloak unreachable: {e.reason}")
    _USERINFO_CACHE.set(access_token, info)
    return info


def _decode_jwt_payload(jwt: str) -> dict:
    """Best-effort decode of the JWT payload (no signature check) for the
    offline-fallback case only."""
    try:
        _, payload, _ = jwt.split(".")
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════
#  AUTH dependency  +  RBAC
# ════════════════════════════════════════════════════════════════

def current_user(request: Request,
                 authorization: str = Header(None)) -> dict:
    """
    Decode the Authorization header, validate the token at Keycloak,
    resolve the LOCAL role from kc_users, and return a small principal
    dict: {'username', 'display_name', 'role'}.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    token = authorization.split(None, 1)[1].strip()
    info = keycloak_userinfo(token)
    username = (info.get("preferred_username")
                or info.get("username")
                or info.get("email")
                or info.get("sub") or "").lower()
    if not username:
        raise HTTPException(status_code=401, detail="No username in token")
    display_name = (info.get("name")
                    or info.get("given_name", "")
                    + (" " + info.get("family_name", "") if info.get("family_name") else "")
                    or username)
    # ensure local record exists
    local = kc_user_get(username)
    if not local:
        kc_user_upsert(username, display_name, role="user")
        local = kc_user_get(username)
    else:
        # refresh last_login + display_name
        kc_user_upsert(username, display_name, role=local["role"])
        local = kc_user_get(username)
    if local["is_revoked"]:
        raise HTTPException(status_code=403, detail="Access revoked by superuser")
    if local["role"] == "pending":
        raise HTTPException(
            status_code=403,
            detail="Account pending superuser approval",
        )
    return {
        "username": username,
        "display_name": display_name,
        "role": local["role"],
    }


def require_role(*roles: str):
    def _dep(user: dict = Depends(current_user)):
        if user["role"] not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Required role: {'/'.join(roles)} (you are '{user['role']}')",
            )
        return user
    return _dep


# Shortcut deps
require_user = current_user        # any authenticated
require_admin = require_role("admin", "superuser")
require_superuser = require_role("superuser")


# ════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ════════════════════════════════════════════════════════════════

app = FastAPI(
    title="VedaX KM Agent (Keycloak)",
    description="Production knowledge-management agent with Keycloak/LDAP auth.",
    version="3.0.0",
)


# ---- Models -----------------------------------------------------

class LoginReq(BaseModel):
    username: str
    password: str


class AskReq(BaseModel):
    query: str
    category: Optional[str] = None


class UserRoleReq(BaseModel):
    username: str
    role: str                       # 'superuser' | 'admin' | 'user'


class UserRevokeReq(BaseModel):
    username: str
    revoke: bool = True


class DocCategoryReq(BaseModel):
    path: str
    category: str
    tags: Optional[List[str]] = None


# ---- LOGIN ------------------------------------------------------

@app.post("/login")
def login(req: LoginReq):
    tokens = keycloak_login(req.username, req.password)
    claims = _decode_jwt_payload(tokens.get("access_token", ""))
    username = (claims.get("preferred_username") or req.username).lower()
    display_name = claims.get("name") or username

    existing = kc_user_get(username)
    if not existing:
        # First-time login.  Map from Keycloak roles to local role.
        # Anyone who is NOT in superuser/admin Keycloak groups lands as
        # 'pending' and CANNOT use the app until a superuser approves.
        initial_role = kc_role_from_keycloak_claims(claims)
        kc_user_upsert(username, display_name, role=initial_role)
        existing = kc_user_get(username)

    if existing["is_revoked"]:
        raise HTTPException(403, "Access revoked by superuser")
    if existing["role"] == "pending":
        # Token is still returned (so the UI can poll / show banner) but
        # the principal cannot reach any /api/* endpoint until promoted.
        return {
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "expires_in": tokens.get("expires_in"),
            "user": {
                "username": username,
                "display_name": display_name,
                "role": "pending",
                "message": ("Your account is awaiting superuser approval. "
                            "Please contact your VedaX administrator."),
            },
        }
    return {
        "access_token": tokens.get("access_token"),
        "refresh_token": tokens.get("refresh_token"),
        "expires_in": tokens.get("expires_in"),
        "user": {
            "username": username,
            "display_name": display_name,
            "role": existing["role"],
        },
    }


@app.get("/me")
def me(user: dict = Depends(current_user)):
    return user


# ---- ASK (every authenticated role can ask) --------------------

@app.post("/api/ask")
def api_ask(req: AskReq, user: dict = Depends(current_user)):
    # L1 INPUT guardrail
    g_in = GUARD.inspect_query(user["username"], user["role"], req.query)
    if not g_in.allowed:
        # trip-wire check
        if GUARD.should_revoke(user["username"], user["role"]):
            kc_user_update(user["username"], is_revoked=1, by="auto-tripwire")
        return {
            "answer": "Request blocked by guardrail.",
            "abstained": True,
            "blocked": True,
            "violations": [
                {"layer": v.layer, "rule": v.rule,
                 "severity": v.severity, "why": v.explanation}
                for v in g_in.violations
            ],
            "debug": {"sanitised_query": g_in.sanitised},
        }
    # use the sanitised query downstream (PII masked etc.)
    safe_query = g_in.sanitised
    result = core.do_ask(safe_query, category=req.category)

    # L3 OUTPUT guardrail (mask PII in answer / catch profanity)
    if result.get("answer") and not result.get("abstained"):
        sources_text = " ".join(s.get("file", "")
                                for s in result.get("sources", []))
        g_out = GUARD.inspect_answer(user["username"], user["role"],
                                     safe_query,
                                     result["answer"], sources_text)
        result["answer"] = g_out.sanitised
        if g_out.violations:
            result["guardrail"] = [
                {"layer": v.layer, "rule": v.rule, "severity": v.severity,
                 "why": v.explanation} for v in g_out.violations
            ]
        if not g_out.allowed:
            result["answer"] = "Response blocked by output guardrail."
            result["abstained"] = True

    core.log_answer(user["username"], user["role"],
                    safe_query, req.category, result)
    return result


@app.post("/api/retrieve")
def api_retrieve(req: AskReq, user: dict = Depends(current_user)):
    return core.do_retrieve(req.query, top_k=core.TOP_K, category=req.category)


# ---- DOCUMENTS (admin + superuser) -----------------------------

@app.get("/api/documents")
def api_list_docs(user: dict = Depends(current_user)):
    # auto-rescan so newly dropped folder files appear without restart
    core.store.rescan_auto_fetch()
    return {"documents": core.store.list_documents()}


def _auto_categorise(path: str, fallback: str) -> str:
    """Apply config.documents.auto_category_rules — first match wins."""
    rules = CFG.get_path("documents.auto_category_rules") or []
    low = path.lower()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        token = (rule.get("match") or "").lower()
        if token and token in low:
            return rule.get("category") or fallback
    return fallback


@app.post("/api/documents/upload")
async def api_upload(
    file: UploadFile = File(...),
    category: str = Form("General"),
    tags: str = Form(""),
    user: dict = Depends(require_admin),
):
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Auto-categorise if admin left the field blank or chose General
    if not category or category.strip().lower() == "general":
        category = _auto_categorise(dest, "General")
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    seconds = core.store.add_document(dest, category=category, tags=tag_list)
    return {"documents": core.store.list_documents(),
            "chunks_indexed": len(core.store.engine.chunks) if core.store.engine else 0,
            "seconds": round(seconds, 2),
            "saved_to": dest,
            "category_applied": category}


@app.delete("/api/documents")
def api_delete_doc(req: DocCategoryReq, user: dict = Depends(require_admin)):
    seconds = core.store.remove_document(req.path)
    return {"documents": core.store.list_documents(),
            "chunks_indexed": len(core.store.engine.chunks) if core.store.engine else 0,
            "seconds": round(seconds, 2)}


@app.post("/api/documents/rescan")
def api_rescan(user: dict = Depends(require_admin)):
    n = core.store.rescan_auto_fetch()
    return {"newly_indexed": n,
            "documents": core.store.list_documents()}


@app.get("/api/categories")
def api_categories(user: dict = Depends(current_user)):
    return {"categories": core.store.all_categories()}


# ---- USER MANAGEMENT (superuser only) --------------------------

@app.get("/api/users")
def api_users(user: dict = Depends(require_superuser)):
    return {"users": kc_user_list()}


@app.get("/api/users/pending")
def api_pending(user: dict = Depends(require_superuser)):
    """List users awaiting approval (role='pending')."""
    pending = [u for u in kc_user_list() if u["role"] == "pending"]
    return {"users": pending, "count": len(pending)}


class ApproveReq(BaseModel):
    username: str
    role: str            # 'user' | 'admin' (superuser cannot be granted here)


@app.post("/api/users/approve")
def api_approve(req: ApproveReq, user: dict = Depends(require_superuser)):
    if req.role not in ("user", "admin"):
        raise HTTPException(400,
            "approve role must be 'user' or 'admin' "
            "(superuser is granted via /api/users/role)")
    target = kc_user_get(req.username)
    if not target:
        raise HTTPException(404, "user not found")
    if target["role"] != "pending":
        raise HTTPException(400,
            f"user is not pending (current role: {target['role']})")
    kc_user_update(req.username, role=req.role, by=user["username"])
    return {"ok": True, "user": kc_user_get(req.username)}


class RejectReq(BaseModel):
    username: str


@app.post("/api/users/reject")
def api_reject(req: RejectReq, user: dict = Depends(require_superuser)):
    """Reject a pending user — revoke them so they cannot log in again
    until a superuser restores them."""
    target = kc_user_get(req.username)
    if not target:
        raise HTTPException(404, "user not found")
    kc_user_update(req.username, is_revoked=1, by=user["username"])
    return {"ok": True, "user": kc_user_get(req.username)}


@app.post("/api/users/role")
def api_set_role(req: UserRoleReq, user: dict = Depends(require_superuser)):
    if req.role not in ("superuser", "admin", "user"):
        raise HTTPException(400, "role must be superuser/admin/user")
    kc_user_update(req.username, role=req.role, by=user["username"])
    return {"ok": True, "user": kc_user_get(req.username)}


@app.post("/api/users/revoke")
def api_revoke(req: UserRevokeReq, user: dict = Depends(require_superuser)):
    if req.username == user["username"]:
        raise HTTPException(400, "Cannot revoke yourself")
    kc_user_update(req.username, is_revoked=1 if req.revoke else 0,
                   by=user["username"])
    return {"ok": True, "user": kc_user_get(req.username)}


@app.delete("/api/users")
def api_delete_user(req: UserRevokeReq, user: dict = Depends(require_superuser)):
    if req.username == user["username"]:
        raise HTTPException(400, "Cannot delete yourself")
    kc_user_delete(req.username)
    return {"ok": True}


# ---- ADMIN dashboards (admin + superuser) ----------------------

@app.get("/api/admin/audit-logs")
def api_audit(limit: int = 50, user: dict = Depends(require_admin)):
    return {"logs": db.get_audit_logs(days=90)[:limit]}


@app.get("/api/admin/unanswered")
def api_unanswered(user: dict = Depends(require_admin)):
    return {"questions": db.get_unanswered(status="open")}


@app.get("/api/admin/trending")
def api_trending(user: dict = Depends(require_admin)):
    return {"questions": db.get_trending_questions(days=30, limit=20)}


@app.get("/api/admin/compliance-report")
def api_compliance(days: int = 90, user: dict = Depends(require_admin)):
    return db.get_compliance_report(days=days)


@app.get("/api/admin/health")
def api_health(user: dict = Depends(require_admin)):
    """Operational health: process RSS, DB size, cache size, index size.
    Lets admins watch the always-on server's resource use."""
    rss_mb = _process_rss_mb()
    engine = core.store.engine
    return {
        "process_rss_mb": rss_mb,
        "userinfo_cache_entries": len(_USERINFO_CACHE),
        "db_size_mb": round(db.db_size_bytes() / 1e6, 2),
        "indexed_documents": len(core.store.documents),
        "indexed_chunks": len(engine.chunks) if engine else 0,
        "guardrail_enabled": GUARD.enabled,
    }


@app.post("/api/admin/purge")
def api_purge(retain_days: int = 180,
              user: dict = Depends(require_superuser)):
    """Manually purge old audit/guardrail rows (superuser only)."""
    return {"deleted": db.purge_old(retain_days=retain_days),
            "db_size_mb": round(db.db_size_bytes() / 1e6, 2)}


def _process_rss_mb() -> float:
    """Resident set size of this process in MB — pure stdlib, Linux /proc
    with a portable fallback."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except Exception:
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KB, macOS reports bytes
            return round(ru / (1024 if ru > 1e7 else 1) / 1024, 1)
        except Exception:
            return -1.0


# ════════════════════════════════════════════════════════════════
#  UI  (role-aware HTML)
#
#  / ............. login page
#  /app ........... main app shell that hides/shows panels based on role
# ════════════════════════════════════════════════════════════════

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>VedaX · login</title>
<style>
  body { font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
         background:#faf9f6; color:#21241f; margin:0;
         display:flex; align-items:center; justify-content:center; min-height:100vh; }
  .card { background:#fff; border:1px solid #e4e1d8; border-radius:12px;
          padding:36px 32px; width:360px; box-shadow:0 8px 24px rgba(0,0,0,.05); }
  h1 { margin:0 0 6px; font-size:1.4rem; }
  .sub { color:#6b6f64; font-size:.85rem; margin-bottom:24px; }
  label { display:block; font-size:.78rem; color:#6b6f64; margin-bottom:6px; }
  input { width:100%; padding:10px 12px; font-size:.95rem; border:1px solid #e4e1d8;
          border-radius:8px; margin-bottom:14px; box-sizing:border-box; background:#faf9f6; }
  button { width:100%; background:#4a5d4e; color:#fff; border:0; border-radius:8px;
           padding:11px; font-size:.95rem; font-weight:600; cursor:pointer; }
  button:disabled { background:#b9c2ba; cursor:wait; }
  .err { color:#a23b2e; font-size:.85rem; margin-top:10px; min-height:1.1em; }
</style></head><body>
<div class="card">
  <h1>VedaX KM Agent</h1>
  <div class="sub">Sign in with your LDAP / Keycloak credentials.</div>
  <label>Username</label>
  <input id="u" autocomplete="username"/>
  <label>Password</label>
  <input id="p" type="password" autocomplete="current-password"/>
  <button id="b">Sign in</button>
  <div class="err" id="e"></div>
</div>
<script>
async function login() {
  const u = document.getElementById('u').value.trim();
  const p = document.getElementById('p').value;
  const b = document.getElementById('b'); const e = document.getElementById('e');
  if (!u || !p) { e.textContent = 'Username + password required'; return; }
  b.disabled = true; e.textContent = '';
  try {
    const r = await fetch('/login', {method:'POST', headers:{'Content-Type':'application/json'},
                                     body: JSON.stringify({username:u, password:p})});
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Login failed');
    if (data.user && data.user.role === 'pending') {
      e.style.color = '#92660c';
      e.textContent = data.user.message
        || 'Account pending superuser approval.';
      return;
    }
    sessionStorage.setItem('vx_token', data.access_token);
    sessionStorage.setItem('vx_user', JSON.stringify(data.user));
    location.href = '/app';
  } catch (err) {
    e.textContent = err.message;
  } finally { b.disabled = false; }
}
document.getElementById('b').addEventListener('click', login);
['u','p'].forEach(id => document.getElementById(id)
   .addEventListener('keydown', e => { if (e.key==='Enter') login(); }));
</script></body></html>"""


APP_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>VedaX KM Agent</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root { --bg:#faf9f6; --surface:#fff; --border:#e4e1d8; --text:#21241f;
          --soft:#6b6f64; --accent:#4a5d4e; --accent-h:#3b4a3e;
          --danger:#a23b2e; --info:#1e5a96; --warn:#92660c; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,"Segoe UI",system-ui,sans-serif;
         background:var(--bg); color:var(--text); }
  header { display:flex; justify-content:space-between; align-items:center;
           padding:14px 24px; background:#fff; border-bottom:1px solid var(--border); }
  header h1 { font-size:1.15rem; margin:0; }
  header .user { font-size:.85rem; color:var(--soft); display:flex; gap:14px; align-items:center; }
  header .badge { background:var(--accent); color:#fff; padding:2px 10px;
                  border-radius:999px; font-size:.7rem; font-weight:700;
                  text-transform:uppercase; letter-spacing:.04em; }
  header .badge.superuser { background:#7c3aed; }
  header .badge.admin     { background:#1e5a96; }
  header .badge.user      { background:var(--accent); }
  header button { background:#fff; border:1px solid var(--border); border-radius:6px;
                  padding:6px 12px; cursor:pointer; font-size:.85rem; }
  nav { background:#fff; border-bottom:1px solid var(--border); padding:0 24px; display:flex; gap:6px; }
  nav button { background:none; border:0; padding:12px 18px; cursor:pointer;
               font-size:.9rem; color:var(--soft); border-bottom:2px solid transparent; }
  nav button.active { color:var(--text); border-bottom-color:var(--accent); }
  main { padding:24px; max-width:1100px; margin:0 auto; }
  .panel { display:none; }
  .panel.active { display:block; }
  .card { background:#fff; border:1px solid var(--border); border-radius:10px;
          padding:22px; margin-bottom:18px; }
  .card h2 { margin:0 0 14px; font-size:.78rem; color:var(--soft);
             text-transform:uppercase; letter-spacing:.06em; }
  input[type=text], input[type=file], select, textarea {
        font-size:.92rem; padding:9px 11px; border:1px solid var(--border);
        border-radius:6px; background:var(--bg); color:var(--text); width:100%; }
  .row { display:flex; gap:10px; margin-bottom:10px; align-items:center; }
  .row > * { flex:1; } .row .small { flex:0 0 auto; }
  button.primary { background:var(--accent); color:#fff; border:0; border-radius:6px;
                   padding:9px 14px; cursor:pointer; font-weight:600; font-size:.88rem; }
  button.primary:hover { background:var(--accent-h); }
  button.primary:disabled { background:#b9c2ba; cursor:not-allowed; }
  button.danger { background:transparent; color:var(--danger); border:1px solid #e4cfc9;
                  padding:6px 12px; border-radius:6px; cursor:pointer; font-size:.85rem; }
  button.danger:hover { background:#fbeeec; }
  table { width:100%; border-collapse:collapse; }
  th { text-align:left; padding:10px 12px; font-size:.78rem;
       text-transform:uppercase; color:var(--soft); border-bottom:2px solid var(--border); }
  td { padding:10px 12px; border-bottom:1px solid var(--border); font-size:.9rem; }
  .chip { display:inline-block; padding:2px 8px; border-radius:999px;
          font-size:.7rem; background:var(--bg); border:1px solid var(--border); color:var(--soft); }
  .chip.cat { background:#eef1ea; color:var(--accent); border-color:#d7e0d4; font-weight:600; }
  .answer { line-height:1.65; font-size:.95rem; }
  .answer p { margin:0 0 10px; }
  .badge-row { margin-top:14px; }
  .b { display:inline-block; font-size:.72rem; font-weight:700; padding:4px 9px;
       border-radius:5px; text-transform:uppercase; letter-spacing:.04em; }
  .b.ok   { background:#e7efe8; color:var(--accent); }
  .b.warn { background:#fbf3df; color:var(--warn); }
  .b.bad  { background:#fbeeec; color:var(--danger); }
  .src { margin-top:8px; font-size:.78rem; color:var(--soft); }
  .empty { color:var(--soft); font-size:.85rem; padding:10px; }
  .grid3 { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:16px; }
  .stat { background:#fff; border:1px solid var(--border); border-radius:10px; padding:18px; }
  .stat h3 { margin:0 0 8px; font-size:.78rem; color:var(--soft); text-transform:uppercase; }
  .stat .v { font-size:2rem; font-weight:700; color:var(--accent); }
  .stat .s { font-size:.82rem; color:var(--soft); margin-top:4px; }
  .status { font-size:.82rem; margin-top:10px; min-height:1.2em; }
  .status.ok { color:var(--accent); } .status.err { color:var(--danger); }
</style></head><body>
<header>
  <h1>VedaX KM Agent</h1>
  <div class="user">
    <span id="uname">…</span>
    <span class="badge" id="urole">…</span>
    <button onclick="logout()">Sign out</button>
  </div>
</header>
<nav id="nav"></nav>
<main>


<div class="panel active" id="ask-panel">
    <div class="card">
      <h2>Ask the SOP knowledge base</h2>
      <textarea id="q" rows="4"
        placeholder="Ask your question?"
        style="width:100%;box-sizing:border-box;resize:vertical;font-size:1rem;
               padding:14px;border:1.5px solid var(--border);border-radius:8px;
               background:#fff;color:#21241f;font-family:inherit;
               margin-bottom:12px;line-height:1.5;"></textarea>
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:4px;">
        <select id="catSel" style="flex:1;max-width:220px;">
          <option value="">All categories</option>
        </select>
        <button class="primary" id="askBtn" style="flex:0 0 auto;padding:10px 24px;font-size:.95rem;">
          Ask ↵
        </button>
      </div>
      <div style="font-size:.75rem;color:var(--soft);margin-bottom:12px;">
        Ctrl + Enter se bhi bhej sakte ho
      </div>
      <div id="ans"></div>
    </div>
  </div>

  <div class="panel" id="docs-panel">
    <div class="card">
      <h2>Upload SOP document</h2>
      <div class="row">
        <input type="file" id="f" accept=".pdf,.txt,.md"/>
        <input type="text" id="cat" placeholder="Category (e.g. HR)" style="flex:0 0 200px;"/>
        <input type="text" id="tags" placeholder="Tags, comma separated" style="flex:0 0 240px;"/>
        <button class="primary small" id="upBtn">Upload</button>
      </div>
      <div class="status" id="upStatus"></div>
    </div>
    <div class="card">
      <h2>Indexed documents</h2>
      <button class="primary small" id="rescanBtn" style="margin-bottom:10px;">Rescan ./sop_docs folder</button>
      <table id="docTable">
        <tr><th>File</th><th>Category</th><th>Tags</th><th></th></tr>
      </table>
    </div>
  </div>

  <div class="panel" id="pending-panel">
    <div class="card">
      <h2>Pending approvals  ·  superuser only</h2>
      <div class="status" id="pendingEmpty"></div>
      <table id="pendingTable">
        <tr><th>Username</th><th>Name</th><th>First login</th><th>Decide</th></tr>
      </table>
    </div>
  </div>

  <div class="panel" id="users-panel">
    <div class="card">
      <h2>Users  ·  superuser only</h2>
      <table id="userTable">
        <tr><th>Username</th><th>Name</th><th>Role</th><th>Status</th><th>Last login</th><th>Actions</th></tr>
      </table>
    </div>
  </div>

  <div class="panel" id="audit-panel">
    <div class="card">
      <h2>Compliance dashboard</h2>
      <div class="grid3" id="statRow"></div>
    </div>
    <div class="card">
      <h2>Trending questions (30d)</h2>
      <table id="trendingTable"><tr><th>Query</th><th>Count</th><th>Category</th></tr></table>
    </div>
    <div class="card">
      <h2>Unanswered questions</h2>
      <table id="unansTable"><tr><th>Query</th><th>User</th><th>Confidence</th><th>When</th></tr></table>
    </div>
    <div class="card">
      <h2>Recent queries (50)</h2>
      <table id="auditTable"><tr><th>When</th><th>User</th><th>Query</th><th>Grounded</th><th>Status</th></tr></table>
    </div>
  </div>
</main>

<script>
const token = sessionStorage.getItem('vx_token');
const userJSON = sessionStorage.getItem('vx_user');
if (!token || !userJSON) { location.href = '/'; }
const user = JSON.parse(userJSON);

document.getElementById('uname').textContent = user.display_name || user.username;
const roleEl = document.getElementById('urole');
roleEl.textContent = user.role; roleEl.className = 'badge ' + user.role;

function logout() { sessionStorage.clear(); location.href = '/'; }

function buildNav() {
  const tabs = [
    {id:'ask-panel', label:'Ask', roles:['user','admin','superuser']},
    {id:'docs-panel', label:'Documents', roles:['admin','superuser']},
    {id:'audit-panel', label:'Audit & Trends', roles:['admin','superuser']},
    {id:'pending-panel', label:'Pending', roles:['superuser']},
    {id:'users-panel', label:'Users', roles:['superuser']},
  ];
  const nav = document.getElementById('nav');
  nav.innerHTML = '';
  tabs.filter(t => t.roles.includes(user.role)).forEach((t, i) => {
    const b = document.createElement('button');
    b.textContent = t.label;
    if (i === 0) b.classList.add('active');
    b.addEventListener('click', () => switchTab(t.id, b));
    nav.appendChild(b);
  });
}
function switchTab(panelId, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(panelId).classList.add('active');
  btn.classList.add('active');
  if (panelId === 'docs-panel') loadDocs();
  if (panelId === 'users-panel') loadUsers();
  if (panelId === 'audit-panel') loadAudit();
  if (panelId === 'pending-panel') loadPending();
}

// PENDING APPROVALS  (superuser only)
async function loadPending() {
  try {
    const d = await api('/api/users/pending');
    const t = document.getElementById('pendingTable');
    const empty = document.getElementById('pendingEmpty');
    t.innerHTML = '<tr><th>Username</th><th>Name</th><th>First login</th><th>Decide</th></tr>';
    if (!d.users.length) {
      empty.textContent = 'No pending approvals.';
      return;
    }
    empty.textContent = '';
    d.users.forEach(u => {
      const tr = document.createElement('tr');
      const when = u.first_login ? new Date(u.first_login * 1000).toLocaleString() : '-';
      tr.innerHTML =
        '<td>' + escapeHtml(u.username) + '</td>' +
        '<td>' + escapeHtml(u.display_name || '') + '</td>' +
        '<td>' + when + '</td>' +
        '<td></td>';
      const td = tr.children[3];
      ['user','admin'].forEach(r => {
        const b = document.createElement('button');
        b.className = 'primary'; b.style.marginRight = '6px';
        b.style.fontSize = '.75rem';
        b.textContent = 'Approve as ' + r;
        b.onclick = async () => {
          await api('/api/users/approve', {method:'POST',
            body:{username: u.username, role: r}});
          loadPending(); loadUsers();
        };
        td.appendChild(b);
      });
      const rej = document.createElement('button');
      rej.className = 'danger'; rej.textContent = 'Reject';
      rej.onclick = async () => {
        if (!confirm('Reject ' + u.username + '?')) return;
        await api('/api/users/reject', {method:'POST',
          body:{username: u.username}});
        loadPending();
      };
      td.appendChild(rej);
      t.appendChild(tr);
    });
  } catch (e) { console.error(e); }
}

async function api(path, opts = {}) {
  opts.headers = Object.assign({'Authorization': 'Bearer ' + token}, opts.headers || {});
  if (opts.body && !(opts.body instanceof FormData)) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  }
  const r = await fetch(path, opts);
  if (r.status === 401) { sessionStorage.clear(); location.href = '/'; return; }
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
  return data;
}

// ASK
async function loadCats() {
  try {
    const d = await api('/api/categories');
    const sel = document.getElementById('catSel');
    sel.innerHTML = '<option value="">All categories</option>'
      + d.categories.map(c => `<option value="${c}">${c}</option>`).join('');
  } catch (e) {}
}
function escapeHtml(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function renderAnswer(d) {
  const box = document.getElementById('ans');
  if (d.error) { box.innerHTML = `<div class="status err">${escapeHtml(d.error)}</div>`; return; }
  const html = (typeof marked !== 'undefined')
    ? marked.parse(d.answer || '')
    : '<p>' + escapeHtml(d.answer || '') + '</p>';
  let out = '<div class="answer">' + html + '</div>';
  if (d.grounding) {
    const b = d.grounding.badge;
    const cls = b === 'OK' ? 'ok' : b === 'WARN' ? 'warn' : 'bad';
    out += '<div class="badge-row"><span class="b ' + cls + '">Grounded ' +
           Math.round(d.grounding.grounded_fraction * 100) + '%</span></div>';
  }
  if (d.sources && d.sources.length) {
    out += '<div class="src">Sources: ' +
           d.sources.map(s => {
             const label = escapeHtml(s.file);
             if (s.is_critical) {
               const t = s.critical_title ? ' — ' + escapeHtml(s.critical_title) : '';
               return '<span class="b bad">⚠ CRITICAL' + t + '</span> ' + label;
             }
             return label;
           }).join(', ') + '</div>';
  }
  if (d.abstained) {
    out += '<div class="badge-row"><span class="b warn">Abstained — '
         + 'document me jawab nahi mila</span></div>';
  }
  box.innerHTML = out;
}
async function ask() {
  const q = document.getElementById('q').value.trim();
  const cat = document.getElementById('catSel').value;
  const btn = document.getElementById('askBtn');
  if (!q) return;
  btn.disabled = true;
  document.getElementById('ans').innerHTML = '<div class="status">Soch raha hai...</div>';
  try {
    const d = await api('/api/ask', {method:'POST', body:{query:q, category:cat || null}});
    renderAnswer(d);
  } catch (e) {
    document.getElementById('ans').innerHTML = `<div class="status err">${escapeHtml(e.message)}</div>`;
  } finally { btn.disabled = false; }
}
document.getElementById('askBtn').addEventListener('click', ask);
document.getElementById('q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask();
});

// DOCS
async function loadDocs() {
  try {
    const d = await api('/api/documents');
    const t = document.getElementById('docTable');
    t.innerHTML = '<tr><th>File</th><th>Category</th><th>Tags</th><th></th></tr>';
    if (!d.documents.length) {
      t.innerHTML += '<tr><td colspan=4 class="empty">No documents indexed yet.</td></tr>';
      return;
    }
    d.documents.forEach(doc => {
      const tr = document.createElement('tr');
      const fname = doc.path.split(/[\\\\/]/).pop();
      tr.innerHTML = '<td>' + escapeHtml(fname) + '<br><span style="font-size:.75rem;color:#6b6f64;">' + escapeHtml(doc.path) + '</span></td>' +
                     '<td><span class="chip cat">' + escapeHtml(doc.category) + ' v' + (doc.version || '1.0') + '</span></td>' +
                     '<td>' + (doc.tags || []).map(x => '<span class="chip">' + escapeHtml(x) + '</span>').join(' ') + '</td>' +
                     '<td></td>';
      const btn = document.createElement('button');
      btn.className = 'danger'; btn.textContent = 'Delete';
      btn.onclick = async () => {
        if (!confirm('Delete ' + fname + '?')) return;
        await api('/api/documents', {method:'DELETE', body:{path: doc.path, category: doc.category}});
        loadDocs();
      };
      tr.children[3].appendChild(btn);
      t.appendChild(tr);
    });
  } catch (e) { console.error(e); }
}
document.getElementById('upBtn').addEventListener('click', async () => {
  const f = document.getElementById('f');
  const cat = document.getElementById('cat').value.trim() || 'General';
  const tags = document.getElementById('tags').value.trim();
  const st = document.getElementById('upStatus'); st.textContent = '';
  if (!f.files.length) { st.textContent='Choose a file first'; st.className='status err'; return; }
  const fd = new FormData(); fd.append('file', f.files[0]); fd.append('category', cat); fd.append('tags', tags);
  try {
    await api('/api/documents/upload', {method:'POST', body: fd});
    st.textContent = 'Uploaded.'; st.className='status ok';
    f.value=''; document.getElementById('cat').value=''; document.getElementById('tags').value='';
    loadDocs();
  } catch (e) { st.textContent = e.message; st.className='status err'; }
});
document.getElementById('rescanBtn').addEventListener('click', async () => {
  try { const r = await api('/api/documents/rescan', {method:'POST'});
    alert('Newly indexed: ' + r.newly_indexed); loadDocs();
  } catch (e) { alert(e.message); }
});

// USERS
async function loadUsers() {
  const d = await api('/api/users');
  const t = document.getElementById('userTable');
  t.innerHTML = '<tr><th>Username</th><th>Name</th><th>Role</th><th>Status</th><th>Last login</th><th>Actions</th></tr>';
  d.users.forEach(u => {
    const tr = document.createElement('tr');
    const last = u.last_login ? new Date(u.last_login*1000).toLocaleString() : '-';
    const status = u.is_revoked ? '<span class="b bad">REVOKED</span>' : '<span class="b ok">active</span>';
    tr.innerHTML = '<td>' + escapeHtml(u.username) + '</td>' +
                   '<td>' + escapeHtml(u.display_name || '') + '</td>' +
                   '<td><span class="chip cat">' + escapeHtml(u.role) + '</span></td>' +
                   '<td>' + status + '</td>' +
                   '<td>' + last + '</td>' +
                   '<td></td>';
    const td = tr.children[5];
    if (u.username !== user.username) {
      ['user','admin','superuser'].forEach(r => {
        if (r === u.role) return;
        const b = document.createElement('button'); b.className='primary'; b.style.marginRight='4px'; b.style.fontSize='.75rem';
        b.textContent = 'Make ' + r;
        b.onclick = async () => { await api('/api/users/role', {method:'POST', body:{username:u.username, role:r}}); loadUsers(); };
        td.appendChild(b);
      });
      const rev = document.createElement('button'); rev.className='danger';
      rev.textContent = u.is_revoked ? 'Restore' : 'Revoke';
      rev.onclick = async () => { await api('/api/users/revoke', {method:'POST', body:{username:u.username, revoke:!u.is_revoked}}); loadUsers(); };
      td.appendChild(rev);
      const del = document.createElement('button'); del.className='danger'; del.style.marginLeft='4px';
      del.textContent = 'Delete';
      del.onclick = async () => { if (confirm('Delete '+u.username+'?')) { await api('/api/users', {method:'DELETE', body:{username:u.username}}); loadUsers(); } };
      td.appendChild(del);
    } else {
      td.innerHTML = '<span style="color:#6b6f64;font-size:.78rem;">(you)</span>';
    }
    t.appendChild(tr);
  });
}

// AUDIT
async function loadAudit() {
  try {
    const c = await api('/api/admin/compliance-report');
    document.getElementById('statRow').innerHTML =
      `<div class="stat"><h3>Total queries</h3><div class="v">${c.total_queries}</div><div class="s">${c.abstained_count} abstained (${c.abstained_pct}%)</div></div>
       <div class="stat"><h3>Avg grounding</h3><div class="v">${c.avg_grounded_pct}%</div><div class="s">of answered Q's backed by chunks</div></div>
       <div class="stat"><h3>Open unanswered</h3><div class="v">${c.open_unanswered}</div><div class="s">awaiting review</div></div>`;
    const tr = await api('/api/admin/trending');
    const tt = document.getElementById('trendingTable');
    tt.innerHTML = '<tr><th>Query</th><th>Count</th><th>Category</th></tr>';
    tr.questions.forEach(q => tt.innerHTML += `<tr><td>${escapeHtml(q.query)}</td><td>${q.count}</td><td>${escapeHtml(q.category)}</td></tr>`);
    const un = await api('/api/admin/unanswered');
    const ut = document.getElementById('unansTable');
    ut.innerHTML = '<tr><th>Query</th><th>User</th><th>Confidence</th><th>When</th></tr>';
    un.questions.forEach(q => {
      const when = new Date(q.timestamp*1000).toLocaleString();
      ut.innerHTML += `<tr><td>${escapeHtml(q.query)}</td><td>${escapeHtml(q.user_id)}</td><td>${q.confidence.toFixed(2)}</td><td>${when}</td></tr>`;
    });
    const al = await api('/api/admin/audit-logs?limit=50');
    const at = document.getElementById('auditTable');
    at.innerHTML = '<tr><th>When</th><th>User</th><th>Query</th><th>Grounded</th><th>Status</th></tr>';
    al.logs.forEach(l => {
      const when = new Date(l.timestamp*1000).toLocaleString();
      const grd = l.grounded_fraction ? Math.round(l.grounded_fraction*100)+'%' : '-';
      const st = l.abstained ? '<span class="b bad">abstained</span>' : '<span class="b ok">answered</span>';
      at.innerHTML += `<tr><td>${when}</td><td>${escapeHtml(l.user_id)}</td><td>${escapeHtml((l.query||'').substring(0,50))}…</td><td>${grd}</td><td>${st}</td></tr>`;
    });
  } catch (e) { console.error(e); }
}

// init
buildNav();
loadCats();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def root_login():
    return LOGIN_HTML


@app.get("/app", response_class=HTMLResponse)
def root_app():
    return APP_HTML


# ════════════════════════════════════════════════════════════════
#  RETENTION  —  keep an always-on server bounded
# ════════════════════════════════════════════════════════════════

_RETAIN_DAYS = CFG.get_path("retention.audit_days", 180)


def _retention_worker():
    """Daily background purge so the SQLite file does not grow forever.
    Pure stdlib threading; daemon thread dies with the process."""
    import threading
    import time as _t

    def loop():
        while True:
            _t.sleep(24 * 3600)
            try:
                db.purge_old(retain_days=_RETAIN_DAYS)
            except Exception as exc:   # never let the worker kill the app
                print(f"[retention] purge failed: {exc}")

    t = threading.Thread(target=loop, name="vedax-retention", daemon=True)
    t.start()
    return t


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    # one purge at startup so a restart also trims the DB
    try:
        db.purge_old(retain_days=_RETAIN_DAYS)
    except Exception:
        pass
    _retention_worker()
    n0 = core.store.rescan_auto_fetch()
    host = CFG.get_path("server.host", "0.0.0.0")
    port = CFG.get_path("server.port", 8000)
    print("=" * 68)
    print("  VEDAX KM Agent  ·  Keycloak edition  ·  guardrail-protected")
    print("=" * 68)
    print(f"  config       : {vedax_config._DEFAULT_CONFIG_PATH}")
    print(f"  Keycloak     : {KEYCLOAK_URL}")
    print(f"  Realm        : {KEYCLOAK_REALM}")
    print(f"  Client ID    : {KEYCLOAK_CLIENT_ID}")
    print(f"  Superuser KC : {sorted(SUPERUSER_KC_ROLES) or '(none configured)'}")
    print(f"  Admin KC     : {sorted(ADMIN_KC_ROLES) or '(none configured)'}")
    print(f"  Auto-fetch   : {core.AUTO_FETCH_DIR}  (indexed {n0} docs)")
    print(f"  Critical SOPs: {core.CRITICAL_FETCH_DIR}  (atomic chunks)")
    print(f"  Guardrail    : {'ENABLED ✓' if GUARD.enabled else 'disabled'}")
    print()
    print(f"  🌐 Login:     http://localhost:{port}/")
    print(f"  📱 App:       http://localhost:{port}/app")
    print(f"  📖 API docs:  http://localhost:{port}/docs")
    print("=" * 68)
    uvicorn.run(app, host=host, port=port)
