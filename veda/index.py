"""Holographic anchor-voting index.

Candidate generation and ranking are split:

* Every chunk posts the anchor coordinates of its informative words
  (deterministic hash positions — see ``word_anchors``) into per-coordinate
  posting lists, which are plain stdlib arrays.
* A query probes the anchors of its own words, their stems and their
  learned context expansion; chunks accumulate weighted votes.
* Only the best-voted candidates get a full cosine rescore against their
  holographic signature.

A chunk sharing a rare word with the query is found through the exact
same hash coordinates the query probes — a "needle" is never diluted the
way it is in summation trees. Search cost is (probes x posting length +
rescore), sublinear in chunk count: no vector DB, no ANN library. Small
corpora are scanned flat, which is both exact and faster.
"""

import heapq
from array import array
from collections import Counter

from .hypervector import cosine_sig_dense, l2_dense, sparsify


class HoloIndex:
    def __init__(self, leaf_top=256, rescore=400, flat_max=1500,
                 post_cap=10000):
        self.leaf_top = leaf_top    # signature entries kept per chunk
        self.rescore = rescore      # candidates given a full cosine pass
        self.flat_max = flat_max    # below this many chunks, scan flat
        self.post_cap = post_cap    # max chunks listed per coordinate
        self.leaves = []            # list of (signature, payload)
        self._anchors = []          # per-leaf anchor coordinate sets
        self._postings = None       # coord -> array('I') of leaf ids

    def add_leaf(self, dense, payload, anchors=()):
        self.leaves.append((sparsify(dense, self.leaf_top), payload))
        self._anchors.append(tuple(anchors))
        self._postings = None

    def _build(self):
        postings = {}
        cap = self.post_cap
        for leaf_id, anchor_set in enumerate(self._anchors):
            for pos in anchor_set:
                plist = postings.get(pos)
                if plist is None:
                    plist = postings[pos] = array("I")
                if len(plist) < cap:
                    plist.append(leaf_id)
        self._postings = postings

    def search(self, query_dense, k=5, probes=None):
        """``probes``: {coordinate: vote_weight} built by the caller from
        the query's words, stems and context expansion."""
        if not self.leaves:
            return []
        qnorm = l2_dense(query_dense)

        if len(self.leaves) <= self.flat_max or not probes:
            candidate_ids = range(len(self.leaves))
        else:
            if self._postings is None:
                self._build()
            votes = Counter()
            postings = self._postings
            for pos, weight in probes.items():
                plist = postings.get(pos)
                if plist:
                    for leaf_id in plist:
                        votes[leaf_id] += weight
            if not votes:
                candidate_ids = range(len(self.leaves))
            else:
                candidate_ids = [
                    leaf_id for leaf_id, _ in
                    heapq.nlargest(self.rescore, votes.items(),
                                   key=lambda kv: kv[1])
                ]

        results = sorted(
            (
                (cosine_sig_dense(self.leaves[i][0], query_dense, qnorm),
                 self.leaves[i][1])
                for i in candidate_ids
            ),
            key=lambda sp: sp[0],
            reverse=True,
        )
        return results[:k]

    def memory_bytes(self):
        """Approximate index footprint: signatures + posting lists."""
        if self._postings is None:
            self._build()
        total = 0
        for (positions, values, _), _payload in self.leaves:
            total += len(positions) * 2 + len(values)
        for plist in self._postings.values():
            total += len(plist) * 4
        return total
