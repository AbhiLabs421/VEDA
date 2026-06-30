"""NFCorpus head-to-head: BM25 vs MiniLM (RAG retriever) vs VEDA vs VEDA-X.

Methodology: all fusion weights and stage parameters are tuned on the
validation qrels only; the table reports the held-out test split.

    python -m eval.run_eval                  # everything
    python -m eval.run_eval bm25 dense       # subset
"""

import itertools
import sys
import time

from eval.bm25 import BM25
from eval.data_nfcorpus import load_corpus, load_qrels, load_queries
from eval.metrics import evaluate
from eval.veda_x import K, rrf, run_veda_x
from veda import Veda
from veda.encoder import SemanticMemory, tokenize


def run_bm25(docs, queries):
    bm25 = BM25()
    bm25.index(docs)
    return {qid: [d for d, _ in bm25.search(q, k=K)]
            for qid, q in queries.items()}, bm25


def run_dense(docs, queries):
    from eval.dense_minilm import DenseRetriever
    retriever = DenseRetriever(docs)
    qids = list(queries)
    qembs = retriever.model.embed([queries[q] for q in qids])
    run = {}
    for qid, emb in zip(qids, qembs):
        run[qid] = [d for d, _ in retriever.search(None, k=K, query_emb=emb)]
    return run, retriever


def run_veda(docs, queries):
    engine = Veda()
    for doc_id, text in docs.items():
        engine.add(doc_id, text)
    run = {}
    for qid, q in queries.items():
        seen = {}
        for hit in engine.search(q, k=K * 3):
            if hit["doc"] not in seen:
                seen[hit["doc"]] = hit["score"]
        ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
        run[qid] = [d for d, _ in ranked[:K]]
    return run, engine


def build_sem(docs):
    sem = SemanticMemory()
    for text in docs.values():
        sem.observe(tokenize(text))
    return sem


def tune_fusion(runs, val_qrels, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Grid-search RRF weights on the validation split only."""
    names = list(runs)
    val_runs = {
        name: {qid: run.get(qid, []) for qid in val_qrels}
        for name, run in runs.items()
    }
    best = ({}, -1.0)
    for combo in itertools.product(grid, repeat=len(names)):
        if not any(combo):
            continue
        weights = dict(zip(names, combo))
        score = evaluate(rrf(val_runs, weights), val_qrels)["nDCG@10"]
        if score > best[1]:
            best = (weights, score)
    return best


def main():
    want = set(sys.argv[1:]) or {"bm25", "dense", "veda", "veda_x", "hybrid"}
    docs = load_corpus()
    queries = load_queries()
    test_qrels = load_qrels("test")
    val_qrels = load_qrels("validation")
    needed = set(test_qrels) | set(val_qrels)
    queries = {qid: q for qid, q in queries.items() if qid in needed}
    print(f"corpus={len(docs)} docs, judged queries={len(queries)}, "
          f"test queries={len(test_qrels)}\n")

    runs, rows = {}, []

    t0 = time.time()
    bm25_run, bm25_obj = run_bm25(docs, queries)
    runs["bm25"] = bm25_run
    if "bm25" in want:
        rows.append(("bm25", evaluate(bm25_run, test_qrels),
                     time.time() - t0))

    t0 = time.time()
    dense_run, retriever = run_dense(docs, queries)
    runs["dense"] = dense_run
    if "dense" in want:
        rows.append(("dense (MiniLM, the RAG retriever)",
                     evaluate(dense_run, test_qrels), time.time() - t0))

    if "veda" in want:
        t0 = time.time()
        veda_run, _ = run_veda(docs, queries)
        runs["veda"] = veda_run
        rows.append(("veda (zero-dep core)",
                     evaluate(veda_run, test_qrels), time.time() - t0))

    if "veda_x" in want or "hybrid" in want:
        t0 = time.time()
        sem = build_sem(docs)
        xruns = run_veda_x(docs, queries, bm25_obj, retriever, sem,
                           bm25_run, dense_run)
        runs.update(xruns)
        elapsed = time.time() - t0
        rows.append(("bm25 + hyperdim. expansion",
                     evaluate(xruns["bm25_x"], test_qrels), elapsed))
        rows.append(("dense + PRF",
                     evaluate(xruns["dense_x"], test_qrels), 0.0))

    if "hybrid" in want or "veda_x" in want:
        fusable = {n: runs[n] for n in
                   ("bm25", "dense", "bm25_x", "dense_x") if n in runs}
        weights, val_score = tune_fusion(fusable, val_qrels)
        fused = rrf(fusable, weights)
        rows.append(("VEDA-X (full fusion)", evaluate(fused, test_qrels),
                     0.0))
        print(f"fusion weights (val nDCG@10={val_score:.4f}): {weights}\n")

    header = (f"{'system':<36} {'nDCG@10':>8} {'R@100':>8} "
              f"{'MRR@10':>8} {'time':>7}")
    print(header)
    print("-" * len(header))
    for name, m, elapsed in rows:
        print(f"{name:<36} {m['nDCG@10']:>8.4f} {m['Recall@100']:>8.4f} "
              f"{m['MRR@10']:>8.4f} {elapsed:>6.0f}s")


if __name__ == "__main__":
    main()
