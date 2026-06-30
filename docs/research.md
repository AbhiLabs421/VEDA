# Research notes

What is genuinely novel here, what is well-established prior art the
system stands on, and what would need to happen for this to become a
publishable contribution rather than an engineering artifact.

## What is novel in this repository

### Hyperdimensional query expansion as semantic RM3

`eval/veda_x.py` introduces a feedback-term selection scheme where the
score of each candidate term is the cosine between

- the term's **corpus-local hypervector context vector** (the
  random-indexing accumulation of its neighbours' deterministic hash
  vectors), and
- the encoded query vector,

multiplied by feedback-document support and a BM25 IDF term. The
expanded query then re-runs BM25.

Why this is interesting:

- It is **RM3** (Relevance Model 3) in spirit — the most successful
  classical query-expansion family — but the term scoring is semantic
  rather than purely frequency-based.
- It uses **no neural network** in the term-scoring loop, and the
  distributional model is trained on the fly from the corpus itself.
- It produces a measurable lift on BM25 even before any neural stage:
  0.3062 → 0.3330 nDCG@10 on NFCorpus.

The closest prior work we are aware of is the line of "embedding-based
query expansion" papers (e.g. Diaz et al. 2016, Roy et al. 2016), which
all use trained word embeddings. We use a deterministic, training-free
distributional model instead.

### Anchor-voting holographic retrieval

`veda/index.py` indexes chunks by posting the deterministic leading
hash coordinates of their informative words. Queries vote on those same
coordinates through their words, stems and learned context. The voted
candidates get a full cosine rescore against quantised holographic
signatures.

This is a fairly tight integration of three classical ideas (sparse
hash signatures, inverted indexes, holographic reduced representations
in the Plate / Kanerva sense) into one engine. We have not seen the
combination presented as a complete dependency-free system elsewhere.

### Drishti — shape search unified with hyperdimensional retrieval

`veda/ocr.py`'s `Drishti` class searches scanned pages **without
transcribing them**, by bundling per-glyph silhouettes and ink profiles
into hypervectors and matching the query (rendered with the embedded
font) by cosine. Word spotting itself is an established research field
(Manmatha, Rusiñol, Almazán); the contribution here is a dependency-free
implementation that uses the **same** hyperdimensional machinery as the
text retriever, so a single index can rank scans and text together.

## Prior art the system openly stands on

| component | foundational work |
|---|---|
| Sparse ternary hypervectors, bundling, near-orthogonality | Pentti Kanerva (1990s–2000s) |
| Random Indexing for distributional semantics | Magnus Sahlgren (~2005) |
| Holographic Reduced Representations | Tony Plate (1995) |
| Locality-Sensitive Hashing / SimHash | Charikar (2002) |
| Okapi BM25 | Robertson, Walker (1994) |
| Reciprocal-Rank Fusion | Cormack et al. (2009) |
| Pseudo-relevance feedback / Rocchio | Rocchio (1971); Lavrenko, Croft (2001) |
| Word spotting | Manmatha et al. (1996), Rusiñol et al., Almazán et al. |
| MiniLM dense retriever | Wang et al. (2020), sentence-transformers |

None of these are claimed as ours. The contribution is the integration.

## What would need to be true for a workshop paper

- **Generalisation beyond three BEIR datasets.** A SIGIR or ECIR-style
  workshop submission would need to hold across roughly five to seven
  BEIR tasks. We are at three so far (`generalize.py`).
- **An ablation of stage 3** — hyperdimensional expansion — versus a
  classical RM3 baseline with identical fusion weights. The gain on
  BM25 alone (0.3062 → 0.3330) is suggestive but not yet attributed.
- **A paired bootstrap against a stronger dense baseline** (BGE-small
  or E5) rather than only MiniLM. We picked MiniLM because it is the
  most widely deployed RAG retriever in production, but a paper would
  need a second dense baseline.
- **A clear claim and a clear non-claim.** Claim: hyperdimensional
  query expansion is a training-free competitive alternative to neural
  rerankers on CPU. Non-claim: we beat fine-tuned rerankers — we do
  not (and have not tried to).

## What would need to be true for an industrial product

- **Air-gapped deployment guide**, since the offline-by-default story is
  the main differentiator versus closed RAG products.
- **Cost vs accuracy curves** against `gpt-4o-mini`-as-reranker and
  Cohere Rerank, on at least one buyer-relevant benchmark.
- **Multilingual support**: Devanagari tokenises already, but the
  light-stemmer and the OCR font are English-only.
- **A real on-prem benchmark**, ideally on a buyer's own corpus (Indian
  banks, defence, healthcare records). The published BEIR wins are
  necessary but not sufficient for that audience.

## The unhyped summary

- The retrieval pipeline beats the standard RAG retriever on four
  benchmarks. That is a real, measured result.
- The core is small, auditable, and runs on a CPU with no external
  services. That is a real, useful engineering property.
- The novel piece is one stage in a four-stage hybrid; it is a
  contribution, not a revolution. Treat it as such.
- The OCR + Drishti and the PDF parser are good engineering and useful
  for the offline use case. They are not research contributions on
  their own.
