"""Acronym normalisation tests — hyphenated short acronyms common in
Indian regulatory documents (NDS-OM, KYC-OVD, etc.) should match the
same chunks regardless of how the user types them."""

import unittest

from veda.encoder import tokenize


class TestHyphenAcronym(unittest.TestCase):
    def test_hyphen_form_in_corpus_matches_joined_query(self):
        corpus_tokens = tokenize("RBI's NDS-OM is an order matching system.")
        query_tokens = tokenize("ndsom")
        self.assertIn("ndsom", corpus_tokens)
        self.assertEqual(query_tokens, ["ndsom"])
        self.assertTrue(set(query_tokens) & set(corpus_tokens))

    def test_joined_form_in_corpus_matches_hyphenated_query(self):
        corpus_tokens = tokenize("The NDSOM platform is screen based.")
        query_tokens = tokenize("nds-om")
        self.assertIn("ndsom", corpus_tokens)
        # Hyphenated query expands too.
        self.assertIn("ndsom", query_tokens)

    def test_all_three_query_styles_overlap(self):
        corpus = tokenize("NDS-OM matches buy and sell orders.")
        for styled_query in ("NDS-OM", "ndsom", "nds-om", "nds om"):
            self.assertTrue(
                set(tokenize(styled_query)) & set(corpus),
                f"query {styled_query!r} did not overlap with corpus")

    def test_normal_words_untouched(self):
        # The healer must not turn ordinary hyphens into acronym joins.
        self.assertNotIn("highspeed",
                         tokenize("This is a high-performance vehicle."))


if __name__ == "__main__":
    unittest.main()
