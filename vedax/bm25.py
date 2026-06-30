"""Okapi BM25 baseline (pure Python, exact).

Anserini/BEIR convention parameters: k1=0.9, b=0.4.
"""

import math
from collections import Counter

from veda.encoder import tokenize


class BM25:
    def __init__(self, k1=0.9, b=0.4):
        self.k1 = k1
        self.b = b
        self.doc_tf = {}      # doc_id -> Counter(term)
        self.doc_len = {}
        self.df = Counter()
        self.n_docs = 0
        self.avg_len = 0.0

    def index(self, docs):
        for doc_id, text in docs.items():
            terms = tokenize(text)
            tf = Counter(terms)
            self.doc_tf[doc_id] = tf
            self.doc_len[doc_id] = len(terms)
            for term in tf:
                self.df[term] += 1
        self.n_docs = len(self.doc_tf)
        self.avg_len = (sum(self.doc_len.values()) / self.n_docs
                        if self.n_docs else 0.0)
        # Inverted index for fast scoring.
        self._postings = {}
        for doc_id, tf in self.doc_tf.items():
            for term, freq in tf.items():
                self._postings.setdefault(term, []).append((doc_id, freq))

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query, k=100):
        scores = Counter()
        k1, b = self.k1, self.b
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, freq in postings:
                dl = self.doc_len[doc_id]
                denom = freq + k1 * (1 - b + b * dl / self.avg_len)
                scores[doc_id] += idf * freq * (k1 + 1) / denom
        return scores.most_common(k)
