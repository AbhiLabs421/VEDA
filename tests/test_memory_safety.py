"""Tests for memory/retention safety: bounded userinfo cache + DB purge."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vedax_keycloak_server as vks
from vedax_db import Database


class TestTTLCache(unittest.TestCase):
    def test_bounded_under_many_logins(self):
        cache = vks._TTLCache(max_size=500, ttl=60)
        for i in range(5000):
            cache.set(f"tok{i}", {"u": i})
        # never exceeds max_size meaningfully (evicts oldest in batches)
        self.assertLessEqual(len(cache), 500)

    def test_expired_entry_auto_removed(self):
        cache = vks._TTLCache(max_size=10, ttl=60)
        cache.set("a", {"u": 1})
        # force-expire
        cache._store["a"] = ({"u": 1}, time.time() - 1)
        self.assertIsNone(cache.get("a"))
        self.assertNotIn("a", cache._store)

    def test_fresh_entry_returned(self):
        cache = vks._TTLCache(max_size=10, ttl=60)
        cache.set("a", {"u": 1})
        self.assertEqual(cache.get("a"), {"u": 1})

    def test_sweep_runs_on_overflow(self):
        cache = vks._TTLCache(max_size=4, ttl=60)
        # add 3 already-expired + fill to overflow
        for i in range(3):
            cache._store[f"old{i}"] = ({"u": i}, time.time() - 1)
        for i in range(4):
            cache.set(f"new{i}", {"u": i})
        # the expired olds should have been swept
        self.assertTrue(all(k.startswith("new") for k in cache._store))


class TestRetention(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = Database(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_purge_removes_old_audit(self):
        # one fresh, one ancient log
        self.db.add_audit_log("u", "user", "q new", "HR", "ans",
                              1.0, False, [], 0.9)
        # craft an ancient row directly
        c = self.db.conn()
        old_ts = time.time() - 365 * 86400
        c.execute(
            "INSERT INTO audit_logs "
            "(timestamp, user_id, role, query, category, answer, "
            " grounded_fraction, abstained, sources, confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (old_ts, "u", "user", "q old", "HR", "ans", 1.0, 0, "[]", 0.9))
        c.commit(); c.close()

        before = len(self.db.get_audit_logs(days=99999))
        self.assertEqual(before, 2)
        deleted = self.db.purge_old(retain_days=180)
        self.assertGreaterEqual(deleted["audit_logs"], 1)
        after = len(self.db.get_audit_logs(days=99999))
        self.assertEqual(after, 1)   # only the fresh one survives

    def test_db_size_reported(self):
        self.assertGreater(self.db.db_size_bytes(), 0)


class TestHealthHelpers(unittest.TestCase):
    def test_rss_reported(self):
        rss = vks._process_rss_mb()
        # should be a positive number on Linux
        self.assertGreater(rss, 0)


if __name__ == "__main__":
    unittest.main()
