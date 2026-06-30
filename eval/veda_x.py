"""VEDA-X: training-free hybrid retrieval pipeline.

Stages:
  1. First pass: BM25 + dense (MiniLM) runs.
  2. Hyperdimensional query expansion (the novel piece): candidate terms
     from the top first-pass documents are scored by the cosine between
     their corpus-local hypervector context (VEDA's random-indexing
     semantics) and the encoded query — i.e. RM3-style feedback where
     term selection is semantic instead of purely frequency-based.
     The expanded query re-runs BM25.
  3. Dense pseudo-relevance feedback: the query embedding is nudged
     toward the centroid of the top fused documents and re-searched.
  4. Weighted reciprocal-rank fusion of all runs (weights tuned on the
     validation split only).

Everything is training-free and runs on CPU.
"""

import math
from collections import Counter, defaultdict

import numpy as np

from veda.encoder import encode_query, tokenize
from veda.hypervector import l2_dense

K = 100


def rrf(runs, weights, k=K):
    fused = {}
    qids = set()
    for run in runs.values():
        qids.update(run)
    for qid in qids:
        votes = defaultdict(float)
        for name, run in runs.items():
            w = weights.get(name, 0.0)
            if not w:
                continue
            for rank, doc_id in enumerate(run.get(qid, [])):
                votes[doc_id] += w / (60 + rank)
        fused[qid] = [d for d, _ in
                      sorted(votes.items(), key=lambda kv: kv[1],
                             reverse=True)[:k]]
    return fused


def hyperdimensional_expansion(query, fb_doc_ids, docs, sem, bm25,
                               n_terms=10, fb_docs=10):
    """Pick expansion terms from feedback docs by hypervector similarity
    between each candidate term's semantic vector and the encoded query."""
    qtokens = tokenize(query)
    qdense = encode_query(qtokens, sem)
    qnorm = l2_dense(qdense) or 1.0
    qset = set(qtokens)

    candidates = Counter()
    for doc_id in fb_doc_ids[:fb_docs]:
        for term in set(tokenize(docs[doc_id])):
            if term not in qset and len(term) > 2:
                candidates[term] += 1

    scored = []
    for term, df_fb in candidates.items():
        entries = sem.semantic_entries(term)
        dot = sum(val * qdense[pos] for pos, val in entries)
        tnorm = math.sqrt(sum(val * val for _, val in entries)) or 1.0
        sim = dot / (tnorm * qnorm)
        if sim > 0:
            # Semantic affinity x feedback support x corpus rarity.
            scored.append((sim * math.log1p(df_fb) * bm25.idf(term), term))
    scored.sort(reverse=True)
    return [term for _, term in scored[:n_terms]]


def run_veda_x(docs, queries, bm25, retriever, sem,
               bm25_run, dense_run, weights1=None,
               expand_weight=0.4, prf_alpha=0.5, prf_docs=3):
    """Returns dict of named runs (for fusion tuning downstream)."""
    weights1 = weights1 or {"bm25": 0.5, "dense": 0.5}
    first = rrf({"bm25": bm25_run, "dense": dense_run}, weights1)

    # Stage 2: hyperdimensional query expansion -> BM25 re-run.
    bm25_x = {}
    for qid, query in queries.items():
        fb = first.get(qid, [])
        terms = hyperdimensional_expansion(query, fb, docs, sem, bm25)
        expanded = query + " " + " ".join(terms)
        base = {d: s for d, s in bm25.search(query, k=K * 2)}
        exp = {d: s for d, s in bm25.search(expanded, k=K * 2)}
        merged = Counter()
        for d, s in base.items():
            merged[d] += (1 - expand_weight) * s
        for d, s in exp.items():
            merged[d] += expand_weight * s
        bm25_x[qid] = [d for d, _ in merged.most_common(K)]

    # Stage 3: dense PRF re-run.
    doc_pos = {d: i for i, d in enumerate(retriever.doc_ids)}
    qids = list(queries)
    qembs = retriever.model.embed([queries[q] for q in qids])
    dense_x = {}
    for qid, qemb in zip(qids, qembs):
        fb = [doc_pos[d] for d in first.get(qid, [])[:prf_docs]
              if d in doc_pos]
        if fb:
            centroid = retriever.embs[fb].mean(axis=0)
            qemb = qemb + prf_alpha * centroid
            qemb = qemb / (np.linalg.norm(qemb) or 1.0)
        sims = retriever.embs @ qemb
        top = np.argsort(-sims)[:K]
        dense_x[qid] = [retriever.doc_ids[i] for i in top]

    return {"bm25_x": bm25_x, "dense_x": dense_x}
