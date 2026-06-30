"""Standard IR metrics: nDCG@k, Recall@k, MRR@k.

``ranking``: ordered list of doc_ids (best first).
``rels``: {doc_id: graded relevance} for one query.
"""

import math


def ndcg_at_k(ranking, rels, k=10):
    dcg = 0.0
    for i, doc_id in enumerate(ranking[:k]):
        rel = rels.get(doc_id, 0)
        if rel:
            dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2)
               for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranking, rels, k=100):
    if not rels:
        return 0.0
    hits = sum(1 for doc_id in ranking[:k] if doc_id in rels)
    return hits / len(rels)


def mrr_at_k(ranking, rels, k=10):
    for i, doc_id in enumerate(ranking[:k]):
        if doc_id in rels:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(run, qrels, k_ndcg=10, k_recall=100):
    """``run``: {query_id: ordered doc_id list}. Averages over queries
    that have qrels (standard convention)."""
    ndcgs, recalls, mrrs = [], [], []
    for qid, rels in qrels.items():
        ranking = run.get(qid, [])
        ndcgs.append(ndcg_at_k(ranking, rels, k_ndcg))
        recalls.append(recall_at_k(ranking, rels, k_recall))
        mrrs.append(mrr_at_k(ranking, rels))
    n = len(ndcgs) or 1
    return {
        "nDCG@10": sum(ndcgs) / n,
        "Recall@100": sum(recalls) / n,
        "MRR@10": sum(mrrs) / n,
        "queries": len(ndcgs),
    }
