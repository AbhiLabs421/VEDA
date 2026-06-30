"""Deterministic sparse hypervectors.

The core trick that removes the vector database: a token's vector is a
pure function of its bytes. ``token_hv("doctor")`` always yields the same
sparse ternary hypervector, on any machine, with no lookup table, no
trained weights and no storage. The "embedding table" is a hash function.

Vectors are sparse ternary: NNZ positions out of DIM, each +1 or -1.
Random sparse ternary vectors in high dimensions are almost surely
near-orthogonal, which is what makes superposition (bundling) work.

Signatures (compressed bundles) are stored quantized: uint16 positions +
int8 values + precomputed norm. Every consumer normalises, so the
quantisation scale cancels out of all similarity scores. ~3 bytes per
entry, regardless of how much text was bundled in.
"""

import hashlib
import heapq
import math
from array import array
from functools import lru_cache
from itertools import count

DIM = 2048  # hypervector dimensionality
NNZ = 64    # non-zero entries per token vector


@lru_cache(maxsize=131072)
def token_hv(token, role=0, nnz=NNZ):
    """Sparse ternary hypervector for ``token``: tuple of (position, sign).

    ``role`` namespaces the vector space (0 = unigram, 1 = bigram, ...).
    ``nnz`` truncates the vector; because generation is a deterministic
    stream, ``token_hv(t, nnz=16)`` is a prefix of ``token_hv(t)``, so
    truncated vectors still overlap with full ones.
    Deterministic, stateless, regenerated on demand — never stored.
    """
    entries = {}
    msg = ("%d\x1f%s" % (role, token)).encode("utf-8")
    counter = 0
    while len(entries) < nnz:
        digest = hashlib.blake2b(
            msg + counter.to_bytes(4, "little"), digest_size=64
        ).digest()
        for i in range(0, 63, 3):
            pos = ((digest[i] << 8) | digest[i + 1]) % DIM
            if pos not in entries:
                entries[pos] = 1 if digest[i + 2] & 1 else -1
                if len(entries) == nnz:
                    break
        counter += 1
    return tuple(entries.items())


def new_dense():
    """Fresh dense accumulator (used transiently while encoding a chunk)."""
    return [0.0] * DIM


def add_sparse(dense, sparse, weight=1.0):
    """Bundle sparse (pos, val) entries into a dense accumulator."""
    for pos, val in sparse:
        dense[pos] += val * weight


def sparsify(dense, top):
    """Compress a dense bundle into its ``top`` strongest coordinates.

    Returns a signature (positions, int8 values, norm). The hot part runs
    at C speed: map/zip feed heapq.nlargest without a Python-level loop.
    """
    best = heapq.nlargest(top, zip(map(abs, dense), count()))
    maxabs = best[0][0] if best else 0.0
    if not maxabs:
        return array("H"), array("b"), 0.0
    scale = 127.0 / maxabs
    positions = array("H")
    values = array("b")
    for absval, pos in best:
        if not absval:
            break
        q = int(dense[pos] * scale)
        if q:
            positions.append(pos)
            values.append(q)
    norm = math.sqrt(sum(v * v for v in values))
    return positions, values, norm


def l2_dense(dense):
    return math.sqrt(sum(v * v for v in dense))


def cosine_sig_dense(sig, dense, dense_norm):
    """Cosine similarity between a compact signature and a dense query.
    Quantisation scale cancels because both norms are of the same values."""
    positions, values, norm = sig
    if not norm or not dense_norm:
        return 0.0
    dot = 0.0
    for pos, val in zip(positions, values):
        dot += val * dense[pos]
    return dot / (norm * dense_norm)
