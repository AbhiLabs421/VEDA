# Evaluation methodology

Every benchmark in this repository follows the same protocol; this
document spells it out so that the numbers in
[results.md](./results.md) can be reproduced and audited.

## Datasets

### NFCorpus

Medical lay-questions retrieval over PubMed abstracts.

- Corpus: 3,633 abstracts.
- 323 test queries, each with graded relevance judgements.
- Source: community mirror of the original BEIR NFCorpus distribution.

### SciFact

Scientific-claim retrieval over abstracts.

- Corpus: 5,183 abstracts.
- 505 test claims (we use `claims_train.jsonl`, which is the only split
  shipping with evidence; the official `claims_test.jsonl` ships without
  evidence and is therefore not usable for retrieval evaluation).
- 188 dev claims used for validation (tuning fusion weights).
- Source: the official `scifact.s3-us-west-2.amazonaws.com` release.

### FiQA-2018

Financial question answering retrieval.

- Corpus: 57,638 documents.
- 648 test queries, 500 dev queries.
- Source: community mirror of the standard BEIR FiQA distribution.

### FinanceBench (open source)

SEC 10-K filings retrieval, the benchmark used by PageIndex.

- 84 distinct 10-K and 10-Q PDFs.
- 150 questions, each with one or more evidence pages.
- Source: `patronus-ai/financebench` on GitHub.

For the page-retrieval evaluation we **content-align** the gold pages —
the printed page number in the filing differs from the physical PDF
page index by an offset that varies per filing (cover, TOC, half-titles),
so we locate each evidence's actual physical page by content match
before scoring. This makes the metric a pure retrieval test, free of
PDF-specific page-numbering noise.

## Metrics

For BEIR-style graded qrels we report:

- **nDCG@10** — normalised discounted cumulative gain at rank 10. This
  is the headline metric in the BEIR literature.
- **Recall@100** — fraction of relevant documents recovered in the top
  100.
- **MRR@10** — mean reciprocal rank of the first relevant document.

For FinanceBench (binary page relevance) we report **Recall@1, @3, @5**.

All metrics are computed against the test split only. Macro-averaged
over queries.

## Baselines

Both reproduce published literature numbers, so the harness is
trustworthy.

- **BM25** (`vedax/bm25.py`). Pure-Python Okapi BM25 with the Anserini
  parameter set used in the BEIR paper: `k1 = 0.9, b = 0.4`. Reproduces
  ~0.32 nDCG@10 on NFCorpus.
- **all-MiniLM-L6-v2** (`vedax/dense.py`). The standard RAG retriever.
  Mean-pooled, L2-normalised embeddings from the ONNX model on CPU.
  Reproduces ~0.31 on NFCorpus, ~0.37 on FiQA.

## The VEDA-X pipeline under evaluation

Per query, four ranked retrievals are produced and fused with weighted
reciprocal-rank fusion:

1. BM25 over the corpus.
2. Dense (MiniLM) cosine search.
3. BM25 plus **hyperdimensional query expansion**: feedback terms are
   selected from the top first-pass documents by the cosine between the
   term's corpus-local hypervector context vector and the encoded
   query; the expanded query re-runs BM25.
4. **Dense pseudo-relevance feedback**: the query embedding is nudged
   toward the centroid of the top fused documents and re-searched.

The new piece is stage 3 — semantic term selection guided by a
deterministic distributional model (random indexing), with no neural
network in the term-scoring loop.

## Fusion weights and tuning protocol

Fusion weights are tuned per dataset on the **validation** split only,
by an integer grid search over `{0.0, 0.25, 0.5, 0.75, 1.0}^4`. They are
then frozen for the **test** split and reported. The NFCorpus-tuned
weights `bm25_x: 0.25, dense_x: 1.0` are reused unchanged at inference
time inside `vedax.VedaX` for arbitrary user corpora.

## Significance testing

Per dataset, in addition to point estimates, we report a one-sided
**paired bootstrap** *p*-value over the test queries comparing VEDA-X's
nDCG@10 to the dense (MiniLM) baseline's:

- Compute the per-query nDCG@10 gain.
- Resample queries with replacement 5,000 times.
- *p* = fraction of resamples where the mean gain is ≤ 0.

This is the standard non-parametric significance test for IR results.

## Reproducing the runs

The first run downloads ~100 MB of model weights and the dataset
mirrors to `/tmp`. Subsequent runs use the on-disk caches.

```sh
pip install onnxruntime tokenizers numpy

python -m eval.run_eval bm25 dense veda_x hybrid    # NFCorpus full
python -m eval.generalize                            # NFCorpus + SciFact + FiQA
python -m eval.financebench                          # FinanceBench, all 150 q's
```

Runtime on a single CPU core (no GPU):

| run | walltime |
|---|---|
| `run_eval` (NFCorpus) | ~2 min |
| `generalize` | ~35 min (FiQA dense pass is the bottleneck) |
| `financebench` | ~10 min |

For methodology details specific to one benchmark, see the docstring at
the top of `eval/run_eval.py`, `eval/generalize.py` or
`eval/financebench.py`.
