"""Tests for VEDA — stdlib unittest only, run with: python -m unittest -v"""

import os
import tempfile
import unittest

from veda import Veda
from veda.encoder import SemanticMemory, tokenize
from veda.hypervector import NNZ, token_hv


class TestHypervector(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(token_hv("hello"), token_hv("hello"))

    def test_distinct_tokens_nearly_orthogonal(self):
        a = dict(token_hv("hello"))
        b = dict(token_hv("world"))
        overlap = sum(1 for p in a if p in b)
        self.assertLess(overlap, NNZ // 4)

    def test_roles_independent(self):
        self.assertNotEqual(token_hv("hello", role=0), token_hv("hello", role=1))

    def test_exact_nnz(self):
        self.assertEqual(len(token_hv("x")), NNZ)

    def test_truncation_is_prefix(self):
        self.assertEqual(token_hv("hello", nnz=16), token_hv("hello")[:16])


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.engine = Veda()
        self.engine.add("cars", "The engine roared as the car sped down the "
                                "highway past trucks and motorcycles.")
        self.engine.add("ocean", "Whales and dolphins swim through the deep "
                                 "ocean hunting fish beneath the waves.")
        self.engine.add("music", "The guitarist played a melody while the "
                                 "drummer kept rhythm for the singing crowd.")

    def test_keyword_query_ranks_right_doc_first(self):
        hits = self.engine.search("dolphins swimming in the sea", k=3)
        self.assertEqual(hits[0]["doc"], "ocean")

    def test_spans_point_into_source(self):
        hits = self.engine.search("guitar melody", k=1)
        hit = hits[0]
        original = self.engine.read_span(hit["doc"], hit["start"], hit["end"])
        self.assertIn(hit["snippet"].split()[0], original)


class TestSemantics(unittest.TestCase):
    def test_learned_association_bridges_synonyms(self):
        """'doctor' never appears in the medical doc, but co-occurs with
        'physician' and 'hospital' elsewhere — query-side expansion should
        still pull the medical doc above an unrelated one."""
        engine = Veda()
        engine.add("glossary", "A doctor, also called a physician, works at "
                               "a hospital. The doctor helps at the hospital.")
        engine.add("medical", "The physician examined the patient at the "
                              "hospital and prescribed a careful treatment.")
        engine.add("cars", "The mechanic repaired the engine and changed "
                           "the tyres of the racing car in the garage.")
        hits = engine.search("doctor", k=3)
        scores = {h["doc"]: h["score"] for h in hits}
        self.assertGreater(scores.get("medical", 0.0), scores.get("cars", 0.0))


class TestBoundedMemory(unittest.TestCase):
    def test_vocab_eviction(self):
        sem = SemanticMemory(max_vocab=50)
        tokens = [f"word{i}" for i in range(500)]
        sem.observe(tokens)
        self.assertLessEqual(len(sem.assoc), 50)

    def test_assoc_pruning(self):
        sem = SemanticMemory(max_assoc=8)
        sem.observe(tokenize(" ".join(f"target neighbour{i}" for i in range(200))))
        self.assertLessEqual(len(sem.assoc["target"]), 4 * 8)

    def test_observation_budget_samples(self):
        sem = SemanticMemory(obs_budget=100)
        sem.observe([f"w{i}" for i in range(10000)])
        self.assertLessEqual(sum(sem.obs_seen.values()), 101)


class TestVotingIndex(unittest.TestCase):
    def test_search_finds_needle_among_many_chunks(self):
        engine = Veda(chunk_tokens=20, overlap_tokens=4)
        filler = ("The committee discussed quarterly budgets and approved "
                  "the meeting minutes without further comment. ") * 80
        needle = ("The volcano erupted suddenly, spewing lava and ash over "
                  "the sleeping village below the mountain. ")
        engine.add("haystack", filler + needle + filler)
        engine.index.flat_max = 10  # force the voting path, not flat scan
        hits = engine.search("volcano lava eruption", k=3)
        self.assertTrue(any("volcano" in h["snippet"] for h in hits))
        # Posting lists were actually built and are non-trivial. (The
        # repetitive filler words are stopword-damped below the anchor
        # threshold by design, so only informative words are posted.)
        self.assertGreater(len(engine.index._postings), 10)


class TestFileStreaming(unittest.TestCase):
    def test_add_file_and_span_readback(self):
        text = ("Solar panels convert sunlight into electricity. " * 50
                + "The secret recipe uses saffron and cardamom in the kheer. "
                + "Wind turbines spin in the coastal breeze. " * 50)
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        try:
            engine = Veda(block_bytes=512)  # force multiple blocks
            engine.add_file(path, doc_id="energy")
            hits = engine.search("saffron cardamom recipe", k=3)
            self.assertTrue(any("saffron" in h["snippet"] for h in hits))
        finally:
            os.unlink(path)


class TestPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        engine = Veda()
        engine.add("space", "The rocket carried a satellite into orbit "
                            "around the planet for astronomy research.")
        fd, path = tempfile.mkstemp(suffix=".veda")
        os.close(fd)
        try:
            engine.save(path)
            loaded = Veda.load(path)
            hits = loaded.search("satellite orbit", k=1)
            self.assertEqual(hits[0]["doc"], "space")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
