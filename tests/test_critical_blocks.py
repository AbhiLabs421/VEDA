"""Atomic-chunking tests for the compliance-critical SOP feature.

Two protection mechanisms are tested:

  1. Inline ``[[CRITICAL: title]] ... [[/CRITICAL]]`` markers.
  2. Whole-file folder convention (every file inside a designated
     ``critical_sops`` folder is one atomic chunk).

For each, we verify:

  * the block is emitted as a SINGLE chunk;
  * NO step inside the block is split across chunks;
  * NO ``[[CRITICAL...]]`` marker leaks into the chunk text;
  * the chunk is tagged with ``is_critical`` + ``critical_title``;
  * retrieval (``smart_search``) surfaces those flags to callers.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veda.critical_blocks import (
    parse_critical_spans,
    expand_to_critical,
    strip_markers,
    is_critical_file,
)
from vedax import VedaX


class TestParser(unittest.TestCase):
    def test_single_block(self):
        text = "Hello. [[CRITICAL: My Title]]inside[[/CRITICAL]] world"
        spans = parse_critical_spans(text)
        self.assertEqual(len(spans), 1)
        s, e, title = spans[0]
        self.assertEqual(title, "My Title")
        self.assertIn("inside", text[s:e])
        self.assertIn("[[CRITICAL", text[s:e])
        self.assertIn("[[/CRITICAL]]", text[s:e])

    def test_multiple_blocks(self):
        text = ("Pre [[CRITICAL: A]]aa[[/CRITICAL]] mid "
                "[[CRITICAL: B]]bb[[/CRITICAL]] post")
        spans = parse_critical_spans(text)
        self.assertEqual([t for _, _, t in spans], ["A", "B"])

    def test_unclosed_marker_is_tolerated(self):
        text = "Some text [[CRITICAL: Never Closed]] more text no close"
        spans = parse_critical_spans(text)
        self.assertEqual(spans, [])  # don't index the half-block

    def test_case_insensitive(self):
        text = "x [[critical: Lower]]yy[[/CRITICAL]] z"
        spans = parse_critical_spans(text)
        self.assertEqual(len(spans), 1)

    def test_strip_markers(self):
        text = "[[CRITICAL: T]]\nStep 1\nStep 2\n[[/CRITICAL]]"
        out = strip_markers(text)
        self.assertNotIn("[[CRITICAL", out)
        self.assertNotIn("[[/CRITICAL", out)
        self.assertIn("Step 1", out)
        self.assertIn("Step 2", out)


class TestExpand(unittest.TestCase):
    def test_no_overlap_no_expand(self):
        spans = [(100, 200, "X")]
        s, e, t = expand_to_critical(0, 50, spans)
        self.assertEqual((s, e, t), (0, 50, None))

    def test_partial_overlap_expands_both_sides(self):
        spans = [(100, 200, "X")]
        s, e, t = expand_to_critical(150, 175, spans)
        self.assertEqual((s, e, t), (100, 200, "X"))

    def test_left_overlap_expands_right(self):
        spans = [(100, 200, "X")]
        s, e, t = expand_to_critical(50, 150, spans)
        self.assertEqual((s, e, t), (50, 200, "X"))


class TestInlineMarkerChunking(unittest.TestCase):
    def _write(self, body):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "sop.txt")
        with open(p, "w") as f:
            f.write(body)
        return p

    def test_block_emitted_once_atomically(self):
        body = (
            "Routine intro paragraph. " * 30 +
            "\n\n[[CRITICAL: Trade Cancel]]\n"
            "Step 1: Freeze settlement queue.\n"
            "Step 2: Notify risk desk.\n"
            "Step 3: Dual approval CRO + CFO.\n"
            "Step 4: Reverse the trade.\n"
            "Step 5: File regulatory report.\n"
            "[[/CRITICAL]]\n\n" +
            "Routine outro paragraph. " * 30
        )
        path = self._write(body)
        eng = VedaX(use_dense=False, chunk_tokens=20, overlap_tokens=5)
        eng.add(path)
        crit = [c for c in eng.chunks if c[2].get("is_critical")]
        self.assertEqual(len(crit), 1, "expected exactly one critical chunk")
        _, text, meta = crit[0]
        self.assertEqual(meta["critical_title"], "Trade Cancel")
        for n in (1, 2, 3, 4, 5):
            self.assertIn(f"Step {n}", text,
                          f"Step {n} missing — critical block was split!")
        self.assertNotIn("[[CRITICAL", text)
        self.assertNotIn("[[/CRITICAL", text)

    def test_non_critical_chunks_not_flagged(self):
        body = ("Pure routine. " * 50 +
                "\n[[CRITICAL: X]]inside[[/CRITICAL]]\n" +
                "More routine. " * 50)
        path = self._write(body)
        eng = VedaX(use_dense=False, chunk_tokens=20, overlap_tokens=5)
        eng.add(path)
        non = [c for c in eng.chunks if not c[2].get("is_critical")]
        self.assertGreater(len(non), 2)
        for _, _, m in non:
            self.assertIsNone(m.get("critical_title"))


class TestFolderConvention(unittest.TestCase):
    def test_is_critical_file(self):
        with tempfile.TemporaryDirectory() as root:
            inside = os.path.join(root, "sop.txt")
            with open(inside, "w") as f:
                f.write("x")
            outside = os.path.join(tempfile.gettempdir(), "other.txt")
            self.assertTrue(is_critical_file(inside, root))
            self.assertFalse(is_critical_file(outside, root))

    def test_whole_file_is_one_atomic_chunk(self):
        # Big file (would normally produce many chunks) — but because
        # it lives in the critical folder, it comes out as ONE chunk.
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "incident_runbook.txt")
            steps = "\n".join(f"Step {i}: do thing {i}." for i in range(1, 30))
            with open(path, "w") as f:
                f.write(steps)
            eng = VedaX(use_dense=False, chunk_tokens=10, overlap_tokens=2)
            eng.mark_critical_path(path)
            eng.add(path)
            self.assertEqual(len(eng.chunks), 1)
            _, text, meta = eng.chunks[0]
            self.assertTrue(meta["is_critical"])
            self.assertTrue(meta["whole_file"])
            for i in range(1, 30):
                self.assertIn(f"Step {i}", text)


class TestRetrievalSurfacesFlags(unittest.TestCase):
    def test_smart_search_returns_is_critical(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sop.txt")
            body = (
                "routine intro paragraph about things. " * 20 +
                "\n\n[[CRITICAL: KYC Red Flags]]\n"
                "Reject if PEP customer from sanctioned country.\n"
                "Escalate if cash transaction over 10 lakh "
                "single instance.\n"
                "[[/CRITICAL]]\n\n" +
                "routine outro paragraph about other matters. " * 20
            )
            with open(path, "w") as f:
                f.write(body)
            eng = VedaX(use_dense=False, chunk_tokens=20, overlap_tokens=5)
            eng.add(path)
            res = eng.smart_search(
                "what about kyc red flags PEP sanctioned", max_keep=3)
            hits = res["hits"]
            critical_hits = [h for h in hits if h.get("is_critical")]
            self.assertGreaterEqual(len(critical_hits), 1)
            self.assertEqual(critical_hits[0]["critical_title"], "KYC Red Flags")


if __name__ == "__main__":
    unittest.main()
