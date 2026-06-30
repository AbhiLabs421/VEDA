"""High-level engine: ingest documents (in-memory or streamed from disk),
search semantically, optionally save/load the whole engine as one file.

Chunks are addressed as (doc_id, start, end) spans into the original
text — the document itself is the storage; the index keeps only compact
holographic signatures. ``add_file`` streams a file block by block, so
RAM stays bounded no matter how large the file is.
"""

import os
import pickle

import heapq

from .encoder import (STEM_NNZ, SemanticMemory, encode_chunk, encode_query,
                      light_stem, tokenize_spans, word_anchors)
from .hypervector import token_hv
from .index import HoloIndex

# Words more frequent than this carry no candidate-generation signal
# (stopwords); they are still part of the signature, just not posted.
ANCHOR_MIN_WEIGHT = 0.2


class Veda:
    def __init__(self, chunk_tokens=64, overlap_tokens=12,
                 block_bytes=1 << 20, **sem_kwargs):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.block_bytes = block_bytes
        self.sem = SemanticMemory(**sem_kwargs)
        self.index = HoloIndex()
        self.docs = {}  # doc_id -> ("text", str) | ("file", path)

    # ------------------------------------------------------------ ingest

    def add(self, doc_id, text):
        """Ingest an in-memory document."""
        self.docs[doc_id] = ("text", text)
        self._ingest_block(doc_id, text, base_offset=0)

    def add_file(self, path, doc_id=None):
        """Stream a file of any size; only spans are kept, not the text.

        Note: spans are byte offsets; snippets of multi-byte (non-ASCII)
        text may be clipped at span edges.
        """
        doc_id = doc_id or os.path.basename(path)
        self.docs[doc_id] = ("file", path)
        offset = 0
        with open(path, "rb") as f:
            carry = b""
            while True:
                block = f.read(self.block_bytes)
                data = carry + block
                if not data:
                    break
                if block:
                    cut = max(data.rfind(b" "), data.rfind(b"\n"))
                    if cut <= 0:
                        cut = len(data)
                else:
                    cut = len(data)
                text = data[:cut].decode("utf-8", "ignore")
                self._ingest_block(doc_id, text, base_offset=offset)
                carry = data[cut:]
                offset += cut
                if not block:
                    break
        return doc_id

    def _ingest_block(self, doc_id, text, base_offset):
        spans = tokenize_spans(text)
        if not spans:
            return
        from .encoder import tokenize as _tokenize_with_acronyms
        self.sem.observe(_tokenize_with_acronyms(text))
        step = max(1, self.chunk_tokens - self.overlap_tokens)
        n = len(spans)
        i = 0
        while i < n:
            j = min(n, i + self.chunk_tokens)
            words = [spans[t][0] for t in range(i, j)]
            start = base_offset + spans[i][1]
            end = base_offset + spans[j - 1][2]
            self.index.add_leaf(encode_chunk(words, self.sem),
                                (doc_id, start, end),
                                anchors=self._chunk_anchors(words))
            if j >= n:
                break
            i += step

    def _chunk_anchors(self, words):
        """Posting keys for a chunk: anchors of its informative words."""
        anchors = set()
        for word in set(words):
            if self.sem.token_weight(word) < ANCHOR_MIN_WEIGHT:
                continue
            anchors.update(word_anchors(word))
            stem = light_stem(word)
            if stem:
                anchors.update(p for p, _ in token_hv(stem, role=2, nnz=2))
        return anchors

    def _query_probes(self, tokens):
        """{coordinate: weight} from query words, stems and learned context."""
        probes = {}

        def vote(pos, weight):
            if weight > probes.get(pos, 0.0):
                probes[pos] = weight

        for word in set(tokens):
            wt = self.sem.token_weight(word)
            for pos in word_anchors(word):
                vote(pos, wt)
            stem = light_stem(word)
            if stem:
                for pos, _ in token_hv(stem, role=2, nnz=2):
                    vote(pos, 0.6 * wt)
            ctx = self.sem.assoc.get(word)
            if ctx:
                top = heapq.nlargest(8, ctx.items(),
                                     key=lambda kv: abs(kv[1]))
                for pos, _ in top:
                    vote(pos, 0.5 * wt)
        return probes

    # ------------------------------------------------------------ search

    def search(self, query, k=5):
        tokens = [w for w, _, _ in tokenize_spans(query)]
        if not tokens or not self.index.leaves:
            return []
        hits = self.index.search(encode_query(tokens, self.sem), k=k,
                                 probes=self._query_probes(tokens))
        return [
            {
                "doc": doc_id,
                "score": round(score, 4),
                "start": start,
                "end": end,
                "snippet": " ".join(self.read_span(doc_id, start, end).split()),
            }
            for score, (doc_id, start, end) in hits
        ]

    def read_span(self, doc_id, start, end):
        """Fetch original text for a span (from memory or straight off disk)."""
        kind, src = self.docs[doc_id]
        if kind == "text":
            return src[start:end]
        with open(src, "rb") as f:
            f.seek(start)
            return f.read(end - start).decode("utf-8", "ignore")

    # ----------------------------------------------------------- persist

    def save(self, path):
        """Persist the whole engine as a single portable file (no server,
        no DB — just bytes)."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            engine = pickle.load(f)
        if not isinstance(engine, cls):
            raise TypeError("not a Veda index file: %r" % path)
        return engine

    # ------------------------------------------------------------- stats

    def stats(self):
        return {
            "documents": len(self.docs),
            "chunks": len(self.index.leaves),
            "index_bytes": self.index.memory_bytes() if self.index.leaves else 0,
            "vocab": len(self.sem.freq),
        }
