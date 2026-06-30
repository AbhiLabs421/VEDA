"""Tests for the abstention and citation-verification guards."""

import unittest

from veda.encoder import SemanticMemory, tokenize
from vedax.grounding import (retrieval_confidence, should_abstain,
                             split_sentences, verify_citations)


def _sem(*texts):
    sem = SemanticMemory()
    for t in texts:
        sem.observe(tokenize(t))
    return sem


class TestAbstention(unittest.TestCase):
    def setUp(self):
        # Corpus is all about contracts; off-topic queries must abstain.
        self.sem = _sem(
            "The vendor shall deliver milestones on schedule. "
            "Penalty of two percent of contract value for each week of "
            "delay, capped at ten percent. Arbitration in New Delhi.",
            "Payment terms are net thirty days. Termination requires "
            "sixty days written notice.",
        )

    def test_on_topic_query_does_not_abstain(self):
        hits = [
            {"score": 0.6, "file": "c.txt",
             "snippet": "Penalty of two percent of contract value for each "
                        "week of delay, capped at ten percent."},
            {"score": 0.4, "file": "c.txt",
             "snippet": "Arbitration in New Delhi settles all disputes."},
            {"score": 0.3, "file": "c.txt",
             "snippet": "Payment terms are net thirty days."},
        ]
        conf, _ = retrieval_confidence(
            "what is the penalty for late delivery", hits, self.sem)
        self.assertGreaterEqual(conf, 0.3,
                                f"on-topic query got confidence {conf}")
        self.assertFalse(should_abstain(
            "what is the penalty for late delivery", hits, self.sem))

    def test_off_topic_query_abstains(self):
        hits = [
            {"score": 0.1, "file": "c.txt",
             "snippet": "Penalty of two percent of contract value."},
            {"score": 0.08, "file": "c.txt",
             "snippet": "Arbitration in New Delhi settles disputes."},
        ]
        self.assertTrue(should_abstain(
            "what was the FY2023 revenue of Microsoft", hits, self.sem))

    def test_empty_hits_force_abstain(self):
        conf, reasons = retrieval_confidence("anything", [], self.sem)
        self.assertEqual(conf, 0.0)
        self.assertIn("no_results", reasons)


class TestSentenceSplit(unittest.TestCase):
    def test_handles_typical_punctuation(self):
        out = split_sentences("Revenue grew. Costs rose! What now? Hmm.")
        self.assertEqual(len(out), 4)

    def test_empty_input(self):
        self.assertEqual(split_sentences(""), [])


class TestCitationVerification(unittest.TestCase):
    def setUp(self):
        self.hits = [
            {"file": "10K.pdf",
             "snippet": "Purchases of property plant and equipment "
                        "totalled 1,577 million dollars in 2018."},
            {"file": "10K.pdf",
             "snippet": "Revenue was 32,765 million in 2018 versus "
                        "31,657 million in 2017."},
            {"file": "10K.pdf",
             "snippet": "Arbitration of disputes is conducted in "
                        "New York."},
        ]

    def test_supported_citation_passes(self):
        answer = ("The FY2018 capital expenditure was 1,577 million "
                  "dollars [1].")
        results, roll_up = verify_citations(answer, self.hits)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["supported"])
        self.assertEqual(roll_up, 1.0)

    def test_unsupported_citation_flagged(self):
        # Cites chunk 1 (capex) for a claim about Cayman Islands taxes.
        answer = ("The Cayman Islands tax holiday saved 2 billion "
                  "dollars [1].")
        results, roll_up = verify_citations(answer, self.hits)
        self.assertFalse(results[0]["supported"])
        self.assertLess(roll_up, 0.5)

    def test_mixed_answer(self):
        answer = ("Capital expenditure was 1,577 million [1]. The "
                  "company invented the wheel [2].")
        results, roll_up = verify_citations(answer, self.hits)
        self.assertEqual(len(results), 2)
        statuses = [r["supported"] for r in results]
        self.assertTrue(statuses[0])
        self.assertFalse(statuses[1])
        self.assertAlmostEqual(roll_up, 0.5, places=2)

    def test_no_citations_no_verification(self):
        results, roll_up = verify_citations("No facts here.", self.hits)
        self.assertEqual(results, [])
        # Default: vacuously grounded when nothing was claimed.
        self.assertEqual(roll_up, 1.0)

    def test_meta_citation_footer_does_not_flag_ungrounded(self):
        """An LLM-emitted 'Sources: [1], [5]' attribution line carries
        no factual claim and used to falsely score 0% grounded."""
        answer = ("Capital expenditure was 1,577 million [1].\n"
                  "Sources: [1], [3]")
        results, roll_up = verify_citations(answer, self.hits)
        # Only the real claim is checked; the bare attribution is skipped.
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["supported"])
        self.assertEqual(roll_up, 1.0)


if __name__ == "__main__":
    unittest.main()
