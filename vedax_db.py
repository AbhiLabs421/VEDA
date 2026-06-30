#!/usr/bin/env python3
"""
====================================================================
  VEDAX DATABASE — SQLite models
  Audit logs, unanswered questions, roles, SOP versions, trending Qs
====================================================================
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict

DB_PATH = "./vedax_data.db"


class Database:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.init_schema()

    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def init_schema(self):
        """Create tables if they don't exist."""
        c = self.conn()
        cur = c.cursor()

        # ── Audit logs ──────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                user_id TEXT,
                role TEXT,
                query TEXT,
                category TEXT,
                answer TEXT,
                grounded_fraction REAL,
                abstained INTEGER,
                sources TEXT,
                confidence REAL
            )
        """)

        # ── Unanswered questions ────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS unanswered_questions (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                user_id TEXT,
                query TEXT,
                category TEXT,
                confidence REAL,
                status TEXT DEFAULT 'open'
            )
        """)

        # ── Trending questions ──────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS question_counts (
                id INTEGER PRIMARY KEY,
                query_hash TEXT UNIQUE,
                query TEXT,
                count INTEGER DEFAULT 1,
                category TEXT,
                last_asked REAL
            )
        """)

        # ── SOP versions ────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sop_versions (
                id INTEGER PRIMARY KEY,
                path TEXT,
                category TEXT,
                effective_date REAL,
                deprecated_date REAL,
                version TEXT,
                tags TEXT
            )
        """)

        # ── Roles & users ───────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY,
                user_id TEXT UNIQUE,
                role TEXT,
                name TEXT,
                allowed_categories TEXT,
                api_key TEXT UNIQUE
            )
        """)

        c.commit()
        c.close()

    def add_audit_log(
        self,
        user_id: str,
        role: str,
        query: str,
        category: Optional[str],
        answer: str,
        grounded_fraction: float,
        abstained: bool,
        sources: List[str],
        confidence: float,
    ):
        """Log a Q&A interaction."""
        c = self.conn()
        cur = c.cursor()
        cur.execute(
            """
            INSERT INTO audit_logs
            (timestamp, user_id, role, query, category, answer, grounded_fraction, abstained, sources, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().timestamp(),
                user_id,
                role,
                query,
                category or "general",
                answer[:1000] if answer else "",  # truncate
                grounded_fraction,
                1 if abstained else 0,
                json.dumps(sources),
                confidence,
            ),
        )
        c.commit()
        c.close()

    def add_unanswered(
        self, user_id: str, query: str, category: Optional[str], confidence: float
    ):
        """Log a question we couldn't answer."""
        c = self.conn()
        cur = c.cursor()
        cur.execute(
            """
            INSERT INTO unanswered_questions
            (timestamp, user_id, query, category, confidence, status)
            VALUES (?, ?, ?, ?, ?, 'open')
            """,
            (datetime.now().timestamp(), user_id, query, category or "general", confidence),
        )
        c.commit()
        c.close()

    def track_question(self, query: str, category: Optional[str]):
        """Track question frequency (for trending Qs)."""
        import hashlib

        query_hash = hashlib.md5(query.encode()).hexdigest()
        c = self.conn()
        cur = c.cursor()
        cur.execute(
            """
            INSERT INTO question_counts (query_hash, query, category, last_asked)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(query_hash) DO UPDATE SET count = count + 1, last_asked = ?
            """,
            (query_hash, query, category or "general", datetime.now().timestamp(), datetime.now().timestamp()),
        )
        c.commit()
        c.close()

    def get_audit_logs(self, user_id: Optional[str] = None, days: int = 30, category: Optional[str] = None) -> List[dict]:
        """Get audit logs (filter by user, category, or date range)."""
        c = self.conn()
        cur = c.cursor()
        since = datetime.now().timestamp() - (days * 86400)
        query = "SELECT * FROM audit_logs WHERE timestamp > ?"
        params = [since]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY timestamp DESC LIMIT 1000"
        cur.execute(query, params)
        rows = cur.fetchall()
        c.close()
        return [dict(row) for row in rows]

    def get_unanswered(self, status: str = "open", days: int = 30) -> List[dict]:
        """Get unanswered questions."""
        c = self.conn()
        cur = c.cursor()
        since = datetime.now().timestamp() - (days * 86400)
        cur.execute(
            """
            SELECT * FROM unanswered_questions
            WHERE timestamp > ? AND status = ?
            ORDER BY timestamp DESC
            """,
            (since, status),
        )
        rows = cur.fetchall()
        c.close()
        return [dict(row) for row in rows]

    def mark_unanswered_resolved(self, unanswered_id: int, resolution: str = "resolved"):
        """Mark an unanswered question as resolved (after admin adds FAQ, etc)."""
        c = self.conn()
        cur = c.cursor()
        cur.execute("UPDATE unanswered_questions SET status = ? WHERE id = ?", (resolution, unanswered_id))
        c.commit()
        c.close()

    def get_trending_questions(self, days: int = 30, limit: int = 10) -> List[dict]:
        """Get most asked questions."""
        c = self.conn()
        cur = c.cursor()
        since = datetime.now().timestamp() - (days * 86400)
        cur.execute(
            """
            SELECT query, category, count, last_asked
            FROM question_counts
            WHERE last_asked > ?
            ORDER BY count DESC
            LIMIT ?
            """,
            (since, limit),
        )
        rows = cur.fetchall()
        c.close()
        return [dict(row) for row in rows]

    def register_role(self, user_id: str, role: str, name: str, allowed_categories: List[str], api_key: str):
        """Register a user with a role and allowed categories."""
        c = self.conn()
        cur = c.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO roles (user_id, role, name, allowed_categories, api_key)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, role, name, json.dumps(allowed_categories), api_key),
        )
        c.commit()
        c.close()

    def get_role(self, user_id: str) -> Optional[dict]:
        """Get user role and allowed categories."""
        c = self.conn()
        cur = c.cursor()
        cur.execute("SELECT * FROM roles WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        c.close()
        return dict(row) if row else None

    def get_role_by_key(self, api_key: str) -> Optional[dict]:
        """Verify API key and return role info."""
        c = self.conn()
        cur = c.cursor()
        cur.execute("SELECT * FROM roles WHERE api_key = ?", (api_key,))
        row = cur.fetchone()
        c.close()
        return dict(row) if row else None

    def register_sop_version(
        self, path: str, category: str, effective_date: float, version: str = "1.0", tags: List[str] = None
    ):
        """Register a SOP with version and effective date."""
        c = self.conn()
        cur = c.cursor()
        cur.execute(
            """
            INSERT INTO sop_versions (path, category, effective_date, version, tags)
            VALUES (?, ?, ?, ?, ?)
            """,
            (path, category, effective_date, version, json.dumps(tags or [])),
        )
        c.commit()
        c.close()

    def deprecate_sop(self, path: str):
        """Mark a SOP as deprecated."""
        c = self.conn()
        cur = c.cursor()
        cur.execute("UPDATE sop_versions SET deprecated_date = ? WHERE path = ?", (datetime.now().timestamp(), path))
        c.commit()
        c.close()

    def get_compliance_report(self, days: int = 90) -> dict:
        """Generate a compliance report (query stats, grounding %, etc)."""
        c = self.conn()
        cur = c.cursor()
        since = datetime.now().timestamp() - (days * 86400)

        cur.execute("SELECT COUNT(*) as total FROM audit_logs WHERE timestamp > ?", (since,))
        total_queries = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) as total FROM audit_logs WHERE timestamp > ? AND abstained = 1", (since,))
        abstained_count = cur.fetchone()["total"]

        cur.execute("SELECT AVG(grounded_fraction) as avg FROM audit_logs WHERE timestamp > ? AND abstained = 0", (since,))
        avg_grounded = cur.fetchone()["avg"] or 0.0

        cur.execute(
            """
            SELECT category, COUNT(*) as count, AVG(grounded_fraction) as avg_grounded
            FROM audit_logs WHERE timestamp > ? AND abstained = 0
            GROUP BY category
            """,
            (since,),
        )
        by_category = [dict(row) for row in cur.fetchall()]

        cur.execute("SELECT COUNT(*) as total FROM unanswered_questions WHERE timestamp > ? AND status = 'open'", (since,))
        open_unanswered = cur.fetchone()["total"]

        c.close()
        return {
            "period_days": days,
            "total_queries": total_queries,
            "abstained_count": abstained_count,
            "abstained_pct": round(100 * abstained_count / total_queries, 1) if total_queries else 0,
            "avg_grounded_pct": round(100 * avg_grounded, 1),
            "by_category": by_category,
            "open_unanswered": open_unanswered,
        }

    # ── Retention: keep the DB from growing forever ─────────────────

    def purge_old(self, retain_days: int = 180) -> dict:
        """Delete audit / guardrail / question-count rows older than
        ``retain_days``.  Call this from a daily scheduler (or on
        startup) so an always-on server's SQLite file stays bounded.
        Resolved/closed unanswered questions are also purged."""
        cutoff = datetime.now().timestamp() - retain_days * 86400
        c = self.conn()
        cur = c.cursor()
        deleted = {}
        for table, col in (("audit_logs", "timestamp"),
                           ("unanswered_questions", "timestamp"),
                           ("question_counts", "last_asked")):
            cur.execute(f"DELETE FROM {table} WHERE {col} < ?", (cutoff,))
            deleted[table] = cur.rowcount
        # guardrail_events table is created by vedax_guardrail; guard the call
        try:
            cur.execute("DELETE FROM guardrail_events WHERE ts < ?", (cutoff,))
            deleted["guardrail_events"] = cur.rowcount
        except sqlite3.OperationalError:
            pass
        c.commit()
        cur.execute("VACUUM")     # reclaim disk
        c.close()
        return deleted

    def db_size_bytes(self) -> int:
        """Current size of the SQLite file on disk."""
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0


db = Database()
