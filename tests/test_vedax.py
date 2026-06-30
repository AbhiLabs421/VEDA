"""Tests for the VEDA-X user tool (offline: dense stage disabled)."""

import os
import tempfile
import unittest

from vedax import VedaX
from vedax.extract import extract_text, iter_documents


class TestExtract(unittest.TestCase):
    def test_txt_and_directory_walk(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.txt"), "w") as f:
                f.write("alpha document about solar panels")
            os.mkdir(os.path.join(d, "sub"))
            with open(os.path.join(d, "sub", "b.md"), "w") as f:
                f.write("beta notes about wind turbines")
            with open(os.path.join(d, "c.xyz"), "w") as f:
                f.write("unsupported type, should be skipped")
            docs = dict(iter_documents([d]))
            texts = " ".join(docs.values())
            self.assertEqual(len(docs), 2)
            self.assertIn("solar", texts)
            self.assertIn("wind", texts)

    def test_unsupported_raises(self):
        with self.assertRaises(ValueError):
            extract_text("/tmp/whatever.xyz")


class TestVedaXLexical(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        docs = {
            "contract.txt": "Penalty of two percent applies for each week "
                            "of delivery delay, capped at ten percent. "
                            "Arbitration in New Delhi settles disputes.",
            "policy.txt": "Health checkups are reimbursed up to eight "
                          "thousand rupees. Hospitalization covers spouse "
                          "and children after one year.",
            "recipe.txt": "Soak basmati rice, fry onions golden, layer "
                          "chicken with saffron milk and steam the biryani.",
        }
        for name, text in docs.items():
            with open(os.path.join(self.dir.name, name), "w") as f:
                f.write(text)
        self.engine = VedaX(use_dense=False).add(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_search_routes_to_right_file(self):
        hits = self.engine.search("penalty for late delivery", k=1)
        self.assertIn("contract", hits[0]["file"])
        hits = self.engine.search("hospitalization coverage for spouse", k=1)
        self.assertIn("policy", hits[0]["file"])

    def test_save_load_roundtrip(self):
        fd, path = tempfile.mkstemp(suffix=".vedax")
        os.close(fd)
        try:
            self.engine.save(path)
            loaded = VedaX.load(path)
            hits = loaded.search("biryani rice saffron", k=1)
            self.assertIn("recipe", hits[0]["file"])
        finally:
            os.unlink(path)

    def test_plain_rag_requires_dense(self):
        with self.assertRaises(RuntimeError):
            self.engine.search_plain_rag("anything")


if __name__ == "__main__":
    unittest.main()
