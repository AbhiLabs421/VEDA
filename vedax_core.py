#!/usr/bin/env python3
"""
====================================================================
  VEDAX CORE — shared logic, dono servers (MCP aur FastAPI) yahi use
  karte hain. Isme hai: multi-document index management, retrieval,
  confidence/abstention check, LLM call, citation verification.

  Yeh file khud koi server nahi hai — isko directly run mat karo.
  Use karne ke liye:
    - vedax_mcp_server.py     (FastMCP / MCP clients ke liye)
    - vedax_fastapi_server.py (FastAPI / REST HTTP ke liye)
====================================================================
"""

import sys
import os
import time
from typing import List, Optional, Dict

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

# Repo root is on sys.path automatically when run from the repo;
# fallback to a sibling 'prediction-abhishek-kumar-dev' folder for users
# who unzipped the repo next to these server files.
for _candidate in (".", "./prediction-abhishek-kumar-dev",
                   os.path.dirname(os.path.abspath(__file__))):
    if os.path.exists(os.path.join(_candidate, "vedax", "__init__.py")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        VEDA_DIR = _candidate
        break
else:
    VEDA_DIR = "."

try:
    from vedax import VedaX
    from vedax.llm import stream_chat
    from vedax.grounding import retrieval_confidence, verify_citations
    from vedax.intent import subject_coverage
except ImportError as e:
    print(f"❌ VedaX import nahi hua: {e}", file=sys.stderr)
    print(f"   Check karo: VEDA_DIR = {VEDA_DIR!r} sahi hai kya?", file=sys.stderr)
    raise

from vedax_db import db

# Multiple documents — sab ek saath load honge, same index mein.
# Pehle se yahan list mein daal do, ya runtime mein add_document() se add karo.
DOCUMENTS: List[str] = [
    "./test1.pdf",
    "./test2.pdf",
]

# SOP categories — dropdown mein yeh dikhenge. Naya custom category bhi
# upload ke time free-text se de sakte ho, woh automatically list mein
# aa jaayega (all_categories() existing + predefined dono milata hai).
PREDEFINED_CATEGORIES: List[str] = ["HR", "Finance", "IT", "Compliance", "Admin", "General"]

# ── Ollama / LLM gateway ──────────────────────────────────────────
LLM_URL = "https://ollamagw.xyzindia.net"
LLM_MODEL = "gpt-oss:20b"
LLM_API = "ollama"     # "ollama" → /api/chat   |  "openai" → /v1/chat/completions
LLM_TOKEN = None

# ── Retrieval settings ─────────────────────────────────────────────
TOP_K = 10
CHUNK_TOKENS = 120
OVERLAP_TOKENS = 20
# Production threshold: 0.30 -- 0.01 was so lenient that even gibberish
# slipped past.  Adjust here if you want stricter (0.45) / looser (0.20).
ABSTAIN_THRESHOLD = 0.30
USE_DENSE = False

# Folder that is auto-fetched on server startup (and rescanned every
# refresh) so the path-based "drop a PDF here and it gets indexed"
# workflow works without an upload step.
AUTO_FETCH_DIR = "./sop_docs"

# Folder for COMPLIANCE-CRITICAL SOPs.  Every file dropped here is
# indexed as a single ATOMIC chunk — partial retrieval is impossible
# (Step 2 will never come back without Steps 1 and 3).  Inline
# [[CRITICAL: title]]...[[/CRITICAL]] markers inside ordinary docs
# get the same treatment.  See docs/critical-blocks.md.
CRITICAL_FETCH_DIR = "./critical_sops"


# ════════════════════════════════════════════════════════════════
#  ENGINE STORE — multi-document index state (process-cached)
# ════════════════════════════════════════════════════════════════

class EngineStore:
    """
    VedaX engine ek baar mein saare documents se banta hai (VedaX ka
    apna multi-add API use karke: engine.add(path1); engine.add(path2);
    ...; engine._finalize()). Add/remove document pe poora index
    rebuild hota hai — yeh thoda slow hai per documents lists ke liye
    sahi hai (VedaX ke andar incremental-add ka guarantee nahi hai).

    doc_meta mein har document ka category aur tags rakhe jaate hain —
    yeh index rebuild se independent hai (sirf category badalne pe
    rebuild ki zaroorat nahi).
    """

    def __init__(self):
        self.engine = None
        self.documents: List[str] = []
        self.doc_meta: Dict[str, dict] = {}  # path -> {"category", "tags", "added_at"}

    def build(self, paths: List[str]) -> float:
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Document(s) nahi mile: {missing}")

        t0 = time.time()
        engine = VedaX(use_dense=USE_DENSE, chunk_tokens=CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS)
        # Folder convention: anything inside CRITICAL_FETCH_DIR is
        # marked so the engine emits a whole-file atomic chunk.
        from veda.critical_blocks import is_critical_file
        for p in paths:
            if is_critical_file(p, CRITICAL_FETCH_DIR):
                engine.mark_critical_path(p)
        for p in paths:
            engine.add(p)
        engine._finalize()

        self.engine = engine
        self.documents = list(paths)
        # naye/default documents ko default category de do agar already nahi hai
        for p in paths:
            if p not in self.doc_meta:
                self.doc_meta[p] = {"category": "General", "tags": [], "added_at": time.time()}
        return time.time() - t0

    def ensure_loaded(self):
        """
        Engine ko load karta hai. Priority order:
          1) AUTO_FETCH_DIR mein jo files hain unhe pick karo
          2) DOCUMENTS list (config) ke files
          3) Agar dono empty hain, engine None reh jata hai aur UI
             'koi document nahi hai' dikha degi
        """
        if self.engine is None and not self.documents:
            auto = self._auto_fetch_paths()
            initial = auto if auto else DOCUMENTS
            if initial:
                try:
                    self.build(initial)
                    # auto-fetched docs ko 'General' category mein default
                    # daal do (admin baad mein change kar sakta hai)
                except FileNotFoundError:
                    pass
        return self.engine

    @staticmethod
    def _auto_fetch_paths() -> List[str]:
        """AUTO_FETCH_DIR + CRITICAL_FETCH_DIR ke andar saare supported
        files (PDF/TXT/MD/XLSX/DOCX) return karta hai."""
        exts = (".pdf", ".txt", ".md", ".xlsx", ".docx")
        out = []
        for root in (AUTO_FETCH_DIR, CRITICAL_FETCH_DIR):
            if not os.path.isdir(root):
                continue
            for fn in sorted(os.listdir(root)):
                full = os.path.join(root, fn)
                if os.path.isfile(full) and fn.lower().endswith(exts):
                    out.append(full)
        return out

    def rescan_auto_fetch(self) -> int:
        """
        Auto-fetch folder ko dobara scan karta hai aur jo naye files mile
        hain unhe index mein add karta hai. Return: kitne naye files add hue.
        """
        current = set(self.documents)
        new_paths = [p for p in self._auto_fetch_paths() if p not in current]
        if not new_paths:
            return 0
        for p in new_paths:
            self.doc_meta.setdefault(p, {
                "category": "General",
                "tags": ["auto-fetched"],
                "version": "1.0",
                "effective_date": time.time(),
                "deprecated_date": None,
                "added_at": time.time(),
            })
        self.build(self.documents + new_paths)
        return len(new_paths)

    def add_document(self, path: str, category: str = "General", tags: Optional[List[str]] = None, version: str = "1.0") -> float:
        """
        Naya document add karta hai category/tags/version ke saath. Agar document
        pehle se hi indexed hai, to sirf category/tags/version update ho jaate
        hain — index rebuild nahi hota (rebuild sirf naye document pe
        chahiye hota hai).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Document nahi mila: {path}")
        self.ensure_loaded()
        now = time.time()
        self.doc_meta[path] = {
            "category": (category or "General").strip() or "General",
            "tags": tags or [],
            "version": version,
            "effective_date": now,
            "deprecated_date": None,
            "added_at": now,
        }
        db.register_sop_version(path, category, now, version, tags or [])
        if path in self.documents:
            return 0.0  # already indexed — sirf metadata update hua
        return self.build(self.documents + [path])

    def remove_document(self, path: str) -> float:
        if path not in self.documents:
            raise ValueError(f"Document load nahi hai: {path}")
        remaining = [p for p in self.documents if p != path]
        if not remaining:
            raise ValueError("Kam se kam ek document rehna chahiye — pehle naya add karo.")
        self.doc_meta.pop(path, None)
        return self.build(remaining)

    def list_documents(self) -> List[dict]:
        """Har document ka path + category + tags — UI/API ko isi format mein dikhana hai."""
        self.ensure_loaded()
        return [
            {
                "path": p,
                "category": self.doc_meta.get(p, {}).get("category", "General"),
                "tags": self.doc_meta.get(p, {}).get("tags", []),
                "version": self.doc_meta.get(p, {}).get("version", "1.0"),
                "effective_date": self.doc_meta.get(p, {}).get("effective_date"),
                "deprecated": self.doc_meta.get(p, {}).get("deprecated_date") is not None,
            }
            for p in self.documents
        ]

    def all_categories(self) -> List[str]:
        """Predefined categories + jo categories abhi actually use ho rahi hain, dono milake."""
        self.ensure_loaded()
        used = {self.doc_meta.get(p, {}).get("category", "General") for p in self.documents}
        return sorted(set(PREDEFINED_CATEGORIES) | used)

    def paths_in_category(self, category: str) -> List[str]:
        self.ensure_loaded()
        return [p for p in self.documents if self.doc_meta.get(p, {}).get("category", "General") == category]

    def status(self) -> dict:
        engine = self.ensure_loaded()
        if engine is None:
            return {
                "documents": [],
                "chunks_indexed": 0,
                "mode": "no documents loaded yet",
                "top_k": TOP_K,
                "llm_model": LLM_MODEL,
                "llm_url": LLM_URL,
                "categories": [],
            }
        mode = (
            "hybrid (BM25 + HD expansion + MiniLM dense)" if engine.use_dense
            else "lexical + hyperdimensional (no dense — onnxruntime not used)"
        )
        cat_counts: Dict[str, int] = {}
        for p in self.documents:
            c = self.doc_meta.get(p, {}).get("category", "General")
            cat_counts[c] = cat_counts.get(c, 0) + 1
        return {
            "documents": self.list_documents(),
            "chunks_indexed": len(engine.chunks),
            "mode": mode,
            "top_k": TOP_K,
            "llm_model": LLM_MODEL,
            "llm_url": LLM_URL,
            "categories": [{"category": c, "count": n} for c, n in sorted(cat_counts.items())],
        }


store = EngineStore()


# ════════════════════════════════════════════════════════════════
#  RETRIEVE / ASK — same pipeline jo original script mein tha
# ════════════════════════════════════════════════════════════════

def log_answer(
    user_id: str,
    role: str,
    query: str,
    category: Optional[str],
    answer_data: dict,
):
    """Log a Q&A interaction to audit_logs aur track_question."""
    answer = answer_data.get("answer", "")
    grounded = answer_data.get("grounding", {})
    grounded_fraction = grounded.get("grounded_fraction", 0.0) if grounded else 0.0
    abstained = answer_data.get("abstained", False)
    sources = [s.get("file", "") for s in answer_data.get("sources", [])]
    confidence = answer_data.get("debug", {}).get("confidence", 0.0)

    db.add_audit_log(user_id, role, query, category, answer, grounded_fraction, abstained, sources, confidence)
    db.track_question(query, category)

    if abstained:
        db.add_unanswered(user_id, query, category, confidence)

def _matches_category(file_field: str, allowed_paths: List[str]) -> bool:
    """
    VedaX ke hits mein 'file' field exact path ho sakta hai ya sirf
    filename — isliye dono tarah match karte hain. (Agar tumhari VedaX
    build mein 'file' kuch alag format mein hai, yahan adjust kar lena.)
    """
    if file_field in allowed_paths:
        return True
    base = os.path.basename(file_field)
    return any(os.path.basename(p) == base for p in allowed_paths)


def do_retrieve(query: str, top_k: int = TOP_K, category: Optional[str] = None) -> dict:
    """Sirf retrieval (LLM call nahi) — debug ke liye. `category` diya to sirf usi category ke docs se."""
    engine = store.ensure_loaded()
    if engine is None:
        return {"error": "Koi document load nahi hai abhi — pehle ek document add/upload karo."}

    allowed_paths = None
    if category:
        allowed_paths = store.paths_in_category(category)
        if not allowed_paths:
            return {"error": f"'{category}' category mein koi document nahi hai."}

    # category filter hai to zyada hits le ke baad mein filter/trim karte hain
    search_k = top_k * 5 if allowed_paths else top_k
    res = engine.smart_search(query, max_keep=search_k)
    hits, parsed = res["hits"], res["parsed"]

    if allowed_paths is not None:
        hits = [h for h in hits if _matches_category(h["file"], allowed_paths)][:top_k]

    confidence, reasons = retrieval_confidence(query, hits, engine.sem)
    subj_cov = subject_coverage(parsed, hits)
    if subj_cov >= 0.1 and hits:
        confidence = max(confidence, 0.4 + 0.5 * subj_cov)

    return {
        "intent": parsed.get("intent"),
        "subject": parsed.get("subject"),
        "typo_corrected": parsed.get("typo_corrected", False),
        "category_filter": category,
        "k_selected": res["k_selected"],
        "dropped": res["dropped"],
        "confidence": round(confidence, 3),
        "subject_coverage_pct": round(subj_cov * 100, 1),
        "signals": reasons,
        "would_abstain": confidence < ABSTAIN_THRESHOLD or not hits,
        "chunks": [
            {
                "rank": i + 1,
                "file": h["file"],
                "snippet": " ".join(h["snippet"].split())[:300],
                "is_critical": bool(h.get("is_critical")),
                "critical_title": h.get("critical_title"),
            }
            for i, h in enumerate(hits)
        ],
    }


def do_ask(query: str, category: Optional[str] = None) -> dict:
    """Full pipeline: retrieve -> confidence/abstention -> LLM -> citation verification. `category` se scope kar sakte ho."""
    engine = store.ensure_loaded()
    if engine is None:
        return {
            "answer": "Pehle koi document add/upload karo, fir sawaal poochho.",
            "abstained": True,
            "debug": {},
        }

    allowed_paths = None
    if category:
        allowed_paths = store.paths_in_category(category)
        if not allowed_paths:
            return {
                "answer": f"'{category}' category mein koi document nahi hai.",
                "abstained": True,
                "debug": {"category_filter": category},
            }

    search_k = TOP_K * 5 if allowed_paths else TOP_K
    res = engine.smart_search(query, max_keep=search_k)
    hits, parsed = res["hits"], res["parsed"]

    if allowed_paths is not None:
        hits = [h for h in hits if _matches_category(h["file"], allowed_paths)][:TOP_K]

    confidence, reasons = retrieval_confidence(query, hits, engine.sem)
    subj_cov = subject_coverage(parsed, hits)
    if subj_cov >= 0.1 and hits:
        confidence = max(confidence, 0.4 + 0.5 * subj_cov)

    debug_info = {
        "intent": parsed.get("intent"),
        "subject": parsed.get("subject"),
        "category_filter": category,
        "k_selected": res["k_selected"],
        "dropped": res["dropped"],
        "confidence": round(confidence, 3),
        "subject_coverage_pct": round(subj_cov * 100, 1),
        "signals": reasons,
    }

    if confidence < ABSTAIN_THRESHOLD or not hits:
        return {"answer": "Not in the provided documents.", "abstained": True, "debug": debug_info}

    def _fmt_ctx(i, h):
        tag = ""
        if h.get("is_critical"):
            tag = f" ⚠ CRITICAL — {h.get('critical_title') or 'compliance block'}"
        return f"[{i+1}] {h['file']}{tag}\n{h['snippet']}"
    context = "\n\n".join(_fmt_ctx(i, h) for i, h in enumerate(hits))
    system_msg = (
        "You answer questions STRICTLY from the provided context. "
        "Cite sources as [1], [2] inline at the end of every claim. "
        "If a question cannot be answered from the context, reply "
        "exactly: 'Not in the provided documents.' "
        "If a chunk is marked ⚠ CRITICAL, reproduce its steps "
        "VERBATIM and in order — do NOT summarise, paraphrase, "
        "reorder or omit any step. "
        "Do not invent facts. Be concise."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
    ]

    answer_parts = []
    try:
        for chunk in stream_chat(url=LLM_URL, model=LLM_MODEL, messages=messages, api=LLM_API, token=LLM_TOKEN):
            answer_parts.append(chunk)
    except Exception as e:
        return {
            "answer": None,
            "error": f"LLM call failed: {e}",
            "hint": f"Check {LLM_URL} reachable hai? Model {LLM_MODEL!r} loaded hai? Try: curl {LLM_URL}/api/tags",
            "debug": debug_info,
        }

    answer = "".join(answer_parts)

    grounding_report = None
    if answer.strip() and answer.strip() != "Not in the provided documents.":
        results, grounded_fraction = verify_citations(answer, hits)
        badge = "OK" if grounded_fraction >= 0.8 else "WARN" if grounded_fraction >= 0.5 else "UNGROUNDED"
        grounding_report = {
            "grounded_fraction": round(grounded_fraction, 3),
            "badge": badge,
            "unsupported_sentences": [
                {"citations": s["citations"], "sentence": s["sentence"][:200]}
                for s in results if not s["supported"]
            ],
        }

    return {
        "answer": answer,
        "abstained": False,
        "debug": debug_info,
        "grounding": grounding_report,
        "sources": [
            {
                "rank": i + 1,
                "file": h["file"],
                "is_critical": bool(h.get("is_critical")),
                "critical_title": h.get("critical_title"),
            }
            for i, h in enumerate(hits)
        ],
    }
