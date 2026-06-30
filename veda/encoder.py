"""Text encoding + on-the-fly distributional semantics.

SemanticMemory implements bounded-memory random indexing: words accumulate
the hypervectors of their neighbours, so words used in similar contexts
converge to similar signatures — semantics emerge from the documents
themselves, without any pretrained model.

Scale design: chunks are encoded with cheap identity vectors only, while
the learned semantic expansion is applied on the QUERY side (a query has a
handful of words, a corpus has millions). Synonym bridging is preserved —
the query for "doctor" expands into the positions of "physician" — but
ingest cost per token stays small. Co-occurrence learning is sampled
under a fixed budget, so observation cost is constant regardless of
document size.
"""

import math
import re
from collections import Counter

from .hypervector import NNZ, token_hv

# Latin words/numbers + Devanagari runs.
WORD_RE = re.compile(r"[a-zA-Z0-9]+|[ऀ-ॿ]+")

# Hyphenated acronyms (NDS-OM, RBI-KYC, GST-IN, etc.) — common in Indian
# regulatory documents. Each one is emitted in three forms so a user
# query can land on any of them: hyphenated, joined, and split.
_HYPHEN_ACRONYM = re.compile(r"\b([A-Za-z]{2,5})-([A-Za-z]{2,5})\b")


def tokenize(text):
    """Word tokens (lowercased). Hyphenated short acronyms also yield a
    joined form: 'NDS-OM' -> ['nds', 'om', 'ndsom'] so queries like
    'ndsom' or 'NDS-OM' both match the same corpus chunks."""
    expanded = _HYPHEN_ACRONYM.sub(
        lambda m: f"{m.group(0)} {m.group(1)}{m.group(2)}", text)
    return [m.group(0).lower() for m in WORD_RE.finditer(expanded)]


def tokenize_spans(text):
    """[(word, char_start, char_end)] — tokenize once, reuse for chunking."""
    return [(m.group(0).lower(), m.start(), m.end())
            for m in WORD_RE.finditer(text)]


class SemanticMemory:
    """Streaming, bounded-memory distributional semantics."""

    def __init__(self, window=3, max_vocab=20000, max_assoc=96,
                 ctx_weight=0.6, obs_budget=100000, obs_cap=64, assoc_nnz=16):
        self.window = window
        self.max_vocab = max_vocab
        self.max_assoc = max_assoc
        self.ctx_weight = ctx_weight
        self.obs_budget = obs_budget  # max sampled positions per observe()
        self.obs_cap = obs_cap        # max lifetime observations per word
        self.assoc_nnz = assoc_nnz    # truncated hv length for accumulation
        self.assoc = {}               # word -> Counter(position -> weight)
        self.freq = Counter()         # word -> occurrences seen (full count)
        self.obs_seen = Counter()     # word -> observations consumed
        self._cache = {}              # word -> precomputed semantic entries

    # Flush accumulated pairs every this many sampled positions, so the
    # transient pair buffer stays small however large the document is.
    SLAB = 25000

    def observe(self, tokens):
        """Learn co-occurrence structure from a token stream (sampled)."""
        if not tokens:
            return
        self._cache.clear()
        self.freq.update(tokens)
        n = len(tokens)
        stride = max(1, n // self.obs_budget)
        w = self.window
        obs_seen = self.obs_seen
        obs_cap = self.obs_cap
        pair_counts = Counter()
        used = 0
        for i in range(0, n, stride):
            word = tokens[i]
            if obs_seen[word] >= obs_cap:
                continue
            obs_seen[word] += 1
            lo = i - w if i >= w else 0
            hi = i + w + 1
            if hi > n:
                hi = n
            for j in range(lo, hi):
                if j != i:
                    pair_counts[(word, tokens[j])] += 1
            used += 1
            if used % self.SLAB == 0:
                self._flush(pair_counts)
                pair_counts = Counter()
        self._flush(pair_counts)
        if len(self.assoc) > self.max_vocab:
            self._evict()

    def _flush(self, pair_counts):
        # On high-entropy text most pairs occur once and carry no
        # distributional signal; dropping them saves the bulk of the work.
        if len(pair_counts) > 30000:
            pair_counts = {p: c for p, c in pair_counts.items() if c > 1}
        for (word, neigh), cnt in pair_counts.items():
            ctr = self.assoc.setdefault(word, Counter())
            for pos, sign in token_hv(neigh, nnz=self.assoc_nnz):
                ctr[pos] += sign * cnt
            if len(ctr) > 4 * self.max_assoc:
                self._prune(ctr)

    def _prune(self, ctr):
        top = sorted(ctr.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ctr.clear()
        ctr.update(dict(top[: self.max_assoc]))

    def _evict(self):
        keep = {w for w, _ in self.freq.most_common(self.max_vocab // 2)}
        self.assoc = {w: c for w, c in self.assoc.items() if w in keep}

    def token_weight(self, word):
        """Inverse-frequency damping so stopwords don't drown the signal."""
        return 1.0 / (1.0 + math.log1p(self.freq.get(word, 0)))

    def semantic_entries(self, word):
        """Sparse semantic vector: identity hash-vector + learned context."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        base = [(pos, float(sign)) for pos, sign in token_hv(word)]
        ctr = self.assoc.get(word)
        if ctr:
            top = sorted(ctr.items(), key=lambda kv: abs(kv[1]), reverse=True)
            top = top[: self.max_assoc]
            ctx_l2 = math.sqrt(sum(v * v for _, v in top))
            if ctx_l2:
                scale = self.ctx_weight * math.sqrt(NNZ) / ctx_l2
                base.extend((pos, val * scale) for pos, val in top)
        entries = tuple(base)
        self._cache[word] = entries
        return entries


BIGRAM_NNZ = 16
STEM_NNZ = 32
ANCHORS = 4  # leading hash coords of a word used as posting keys
_SUFFIXES = ("ing", "ed", "es", "e", "s", "ly")


def word_anchors(word):
    """First few hash coordinates of a word — deterministic posting keys.
    A query probing the same word probes the exact same coordinates."""
    return [pos for pos, _ in token_hv(word)[:ANCHORS]]


def light_stem(word):
    """Tiny suffix stripper so 'cooled' and 'cooling' meet in a shared
    stem space (role=2). Not a real stemmer — just the high-value cases."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return None


def encode_chunk(tokens, sem, dim=None):
    """Encode a chunk: identity vectors + light bigram trace. Hot path —
    cost is ~80 float adds per token; unique-word counting collapses
    repeats, so repetitive text is even cheaper."""
    from .hypervector import new_dense
    dense = new_dense()
    token_weight = sem.token_weight
    for word, cnt in Counter(tokens).items():
        wt = token_weight(word) * cnt
        for pos, sign in token_hv(word):
            dense[pos] += sign * wt
        stem = light_stem(word)
        if stem:
            for pos, sign in token_hv(stem, role=2, nnz=STEM_NNZ):
                dense[pos] += sign * wt * 0.6
    for (a, b), cnt in Counter(zip(tokens, tokens[1:])).items():
        wt = 0.5 * cnt * token_weight(b)
        for pos, sign in token_hv(a + "\x1f" + b, role=1, nnz=BIGRAM_NNZ):
            dense[pos] += sign * wt
    return dense


def encode_query(tokens, sem):
    """Encode a query: full semantic expansion (identity + learned
    context), so synonym bridging happens here, where it's cheap."""
    from .hypervector import new_dense
    dense = new_dense()
    prev = None
    for word in tokens:
        wt = sem.token_weight(word)
        for pos, val in sem.semantic_entries(word):
            dense[pos] += val * wt
        stem = light_stem(word)
        if stem:
            for pos, sign in token_hv(stem, role=2, nnz=STEM_NNZ):
                dense[pos] += sign * wt * 0.6
        if prev is not None:
            for pos, sign in token_hv(prev + "\x1f" + word, role=1,
                                      nnz=BIGRAM_NNZ):
                dense[pos] += 0.5 * sign
        prev = word
    return dense
