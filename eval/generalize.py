"""Generalization test: run BM25, MiniLM and VEDA-X on three BEIR
datasets from different domains. The honest research claim — that VEDA-X
beats the standard RAG retriever beyond a single dataset — is what this
script either confirms or kills.

    python -m eval.generalize                     # all 3 datasets
    python -m eval.generalize nfcorpus scifact    # subset

Fusion weights are tuned on each dataset's validation split only; the
table reports the held-out test split. Paired bootstrap p-values
compare the VEDA-X fusion against the dense (MiniLM) baseline.
"""

import itertools
import random
import sys
import time

from eval.datasets import load
from eval.metrics import evaluate, ndcg_at_k
from eval.veda_x import K, rrf, run_veda_x
from vedax.bm25 import BM25
from vedax.dense import MiniLM
from veda.encoder import SemanticMemory, tokenize


def _bm25_run(docs, queries):
    bm25 = BM25()
    bm25.index(docs)
    return ({qid: [d for d, _ in bm25.search(q, k=K)]
             for qid, q in queries.items()}, bm25)


def _dense_run(docs, queries, cache_key):
    import os
    import numpy as np
    cache = f"/tmp/beir_dense_{cache_key}.npz"
    doc_ids = sorted(docs)
    if os.path.isfile(cache):
        data = np.load(cache, allow_pickle=True)
        if list(data["doc_ids"]) == doc_ids:
            embs = data["embs"]
        else:
            embs = None
    else:
        embs = None
    model = MiniLM()
    if embs is None:
        embs = model.embed([docs[d] for d in doc_ids])
        np.savez(cache, doc_ids=np.array(doc_ids), embs=embs)
    qids = list(queries)
    qembs = model.embed([queries[q] for q in qids])
    pos = {d: i for i, d in enumerate(doc_ids)}
    run = {}
    for qid, qe in zip(qids, qembs):
        sims = embs @ qe
        top = np.argsort(-sims)[:K]
        run[qid] = [doc_ids[i] for i in top]

    class _Retriever:
        def __init__(self, embs, doc_ids, model):
            self.embs = embs
            self.doc_ids = doc_ids
            self.model = model

    return run, _Retriever(embs, doc_ids, model)


def _build_sem(docs):
    sem = SemanticMemory()
    for text in docs.values():
        sem.observe(tokenize(text))
    return sem


def _tune_fusion(runs, val_qrels, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Grid-search RRF weights on the validation split."""
    names = list(runs)
    val_runs = {n: {qid: r.get(qid, []) for qid in val_qrels}
                for n, r in runs.items()}
    best = ({}, -1.0)
    for combo in itertools.product(grid, repeat=len(names)):
        if not any(combo):
            continue
        weights = dict(zip(names, combo))
        score = evaluate(rrf(val_runs, weights), val_qrels)["nDCG@10"]
        if score > best[1]:
            best = (weights, score)
    return best


def _paired_bootstrap(run_a, run_b, qrels, n=5000, seed=0):
    """One-sided p-value: P(mean(a - b) <= 0) under resampling."""
    diffs = []
    for qid, rels in qrels.items():
        diffs.append(ndcg_at_k(run_a.get(qid, []), rels)
                     - ndcg_at_k(run_b.get(qid, []), rels))
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    m = len(diffs)
    wins = 0
    for _ in range(n):
        s = sum(diffs[rng.randrange(m)] for _ in range(m)) / m
        if s <= 0:
            wins += 1
    return mean, wins / n


def run_dataset(name):
    print(f"\n=== {name.upper()} ===", flush=True)
    corpus, queries, val_qrels, test_qrels = load(name)
    needed = set(val_qrels) | set(test_qrels)
    queries = {qid: q for qid, q in queries.items() if qid in needed}
    print(f"corpus={len(corpus)} judged queries={len(queries)} "
          f"test={len(test_qrels)} val={len(val_qrels)}", flush=True)

    t0 = time.time()
    bm25_run, bm25 = _bm25_run(corpus, queries)
    print(f"  BM25 done in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    dense_run, retriever = _dense_run(corpus, queries, name)
    print(f"  dense done in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    sem = _build_sem(corpus)
    xruns = run_veda_x(corpus, queries, bm25, retriever, sem,
                       bm25_run, dense_run)
    print(f"  VEDA-X stages done in {time.time() - t0:.0f}s", flush=True)

    all_runs = {"bm25": bm25_run, "dense": dense_run,
                "bm25_x": xruns["bm25_x"], "dense_x": xruns["dense_x"]}
    weights, val_score = _tune_fusion(all_runs, val_qrels)
    fused = rrf(all_runs, weights)

    bm25_m = evaluate(bm25_run, test_qrels)
    dense_m = evaluate(dense_run, test_qrels)
    bx_m = evaluate(xruns["bm25_x"], test_qrels)
    dx_m = evaluate(xruns["dense_x"], test_qrels)
    fused_m = evaluate(fused, test_qrels)

    gain, p = _paired_bootstrap(fused, dense_run, test_qrels)
    return {
        "name": name, "weights": weights, "val_ndcg": val_score,
        "bm25": bm25_m, "dense": dense_m,
        "bm25_x": bx_m, "dense_x": dx_m, "vedax": fused_m,
        "gain_vs_dense": gain, "p_value": p,
    }


def main():
    names = sys.argv[1:] or ["nfcorpus", "scifact", "fiqa"]
    results = [run_dataset(name) for name in names]

    print("\n" + "=" * 96)
    header = (f"{'dataset':<10} {'BM25':>7} {'MiniLM':>7} "
              f"{'BM25+X':>7} {'Dense+P':>7} {'VEDA-X':>7} "
              f"{'gain':>7} {'p':>7}")
    print(header)
    print("-" * len(header))
    wins = 0
    for r in results:
        print(f"{r['name']:<10} "
              f"{r['bm25']['nDCG@10']:>7.4f} {r['dense']['nDCG@10']:>7.4f} "
              f"{r['bm25_x']['nDCG@10']:>7.4f} {r['dense_x']['nDCG@10']:>7.4f} "
              f"{r['vedax']['nDCG@10']:>7.4f} {r['gain_vs_dense']:>+7.4f} "
              f"{r['p_value']:>7.4f}")
        if r["vedax"]["nDCG@10"] > r["dense"]["nDCG@10"]:
            wins += 1
    print("-" * len(header))
    print(f"VEDA-X beats the dense RAG retriever on "
          f"{wins}/{len(results)} datasets")
    print("(p = paired bootstrap one-sided over test queries; "
          "all weights tuned on validation only)")


if __name__ == "__main__":
    main()
