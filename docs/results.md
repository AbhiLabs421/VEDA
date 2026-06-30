# Benchmark results

All numbers are measured on a single CPU core, no GPU, with fusion
weights tuned on each dataset's validation split and the test split
held out. Reproduce with the commands in
[evaluation.md](./evaluation.md).

## Headline

| benchmark | domain | metric | RAG retriever | **VEDA-X** | relative |
|---|---|---|---|---|---|
| NFCorpus | medical lay Q&A | nDCG@10 | 0.3195 | **0.3522** | +10.2% |
| SciFact | scientific claims | nDCG@10 | 0.8177 | **0.8578** |  +4.9% |
| FiQA | financial Q&A | nDCG@10 | 0.3687 | **0.3799** |  +3.0% |
| FinanceBench | SEC 10-K page retrieval | Recall@1 | 0.147 | **0.153** |  +4.1% |

VEDA-X beats the standard RAG retriever on every public IR benchmark
we have tested.

## NFCorpus (full breakdown)

| system | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| BM25 (k1 = 0.9, b = 0.4) | 0.3062 | 0.2376 | 0.5080 |
| all-MiniLM-L6-v2 — the standard RAG retriever | 0.3195 | 0.3147 | 0.5091 |
| BM25 + hyperdimensional expansion (ours) | 0.3330 | 0.2948 | 0.5167 |
| dense + pseudo-relevance feedback | 0.3454 | 0.3387 | 0.5219 |
| **VEDA-X (full fusion)** | **0.3522** | **0.3387** | **0.5376** |

Paired bootstrap over 323 test queries: mean nDCG@10 gain vs the dense
RAG retriever +0.033, *p* < 0.0001 (128 queries improved, 68 worsened,
127 tied).

## Generalisation to three BEIR datasets

| dataset | domain | BM25 | MiniLM (RAG) | **VEDA-X** | gain | p |
|---|---|---|---|---|---|---|
| NFCorpus | medical lay Q&A | 0.3062 | 0.3195 | **0.3522** | +0.0327 | <0.0001 |
| SciFact | scientific claims | 0.8352 | 0.8177 | **0.8578** | +0.0401 | 0.0004 |
| FiQA | financial Q&A | 0.2309 | 0.3687 | **0.3799** | +0.0112 | 0.0082 |

VEDA-X beats the dense RAG retriever on **3/3 datasets**, every gain
statistically significant.

## FinanceBench (PageIndex's home turf)

Page-level retrieval over SEC 10-Ks, 150 open-source questions,
content-aligned gold pages.

| system | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| BM25 | 0.113 | 0.187 | 0.207 |
| MiniLM (RAG retriever) | 0.147 | 0.240 | 0.313 |
| **VEDA-X** | **0.153** | **0.267** | **0.320** |

VEDA-X wins every metric on PageIndex's own chosen benchmark — without
any LLM in the retrieval loop, on one CPU core, with weights tuned on
NFCorpus (not on FinanceBench). Honest framing: absolute numbers are
modest because FinanceBench questions are heavily paraphrased relative
to the answer-page tables — this is a hard retrieval setting, not a
weakness of the system. PageIndex's marketing 98.7% is a different
metric (whole-pipeline answer accuracy with their LLM evaluator); it is
not Recall@k and is not directly comparable.

## Million-token benchmark (the core, no neural)

A 6.5 MB synthetic haystack with 12 distinctive needle sentences hidden
inside; queries are paraphrase-ish word subsets, not exact strings.
Pure-stdlib `veda.Veda` core, no MiniLM, no BM25 fusion.

| metric | result |
|---|---|
| ingest | ~50 s (~20,000 tokens/s) |
| chunks indexed | 19,236 |
| query latency | ~20 ms |
| Recall@5 | 12/12 needles |
| index size | ~21 MB (in RAM, no server) |

Reproduce with `python bench.py`.

## What the numbers do **not** say

- They do not claim VEDA-X beats every closed-source RAG product.
  Closed products (PageIndex, OpenAI File Search, Anthropic File Search,
  managed vector DBs with ColBERT, Cohere Rerank, BGE-M3) are not
  evaluated here because we cannot run them deterministically.
- They do not claim VEDA-X beats large reranker models (Cohere Rerank
  3, BGE-Reranker-v2-Gemma) — those add an extra inference pass and
  would beat any first-pass retriever including ours. We omit them on
  purpose: this project's promise is a **CPU-only, training-free**
  retriever with no neural network strictly required.
- They do not claim the absolute numbers (e.g. ~15% R@1 on FinanceBench)
  are state of the art. They claim the **relative** position vs the
  standard RAG retriever is.

See [research.md](./research.md) for what is genuinely novel and what
remains to be proven.
