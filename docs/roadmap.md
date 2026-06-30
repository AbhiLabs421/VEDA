# Roadmap

What this project still needs in priority order. Each item is sized
roughly so that the work is concrete.

## Near-term — strengthen the research claim

1. **Two more BEIR datasets** (ArguAna, SCIDOCS or Touche-2020) to take
   the generalisation evidence from three to five. Roughly 1 day's
   compute on the existing harness.
2. **Ablate hyperdimensional query expansion vs classical RM3** with
   identical fusion weights, on the existing BEIR datasets, to show that
   the gain from stage 3 is not just "any expansion".
3. **One stronger dense baseline** (BGE-small or E5-small) alongside
   MiniLM, so the dense baseline is not single-point.

## Near-term — strengthen the engineering

4. **Incremental tree / postings updates** so adding new documents
   does not rebuild the whole index.
5. **Numpy fast path** for the core hot loops behind an optional flag,
   without breaking the zero-dependency default.
6. **JPEG decoder** so scanned-page support actually covers the most
   common scan format. Currently PNG / BMP / PGM / PPM only.
7. **Devanagari and other-script light stemmer** for honest
   multilingual support. Tokenisation already handles Devanagari.

## Medium-term — what an enterprise pitch needs

8. **A real on-prem benchmark** on a buyer's own corpus (Indian banks,
   defence, healthcare) — the BEIR wins are necessary but not
   sufficient for that audience.
9. **Cost-vs-accuracy curves** against `gpt-4o-mini`-as-reranker and
   Cohere Rerank on at least one buyer-relevant benchmark.
10. **A minimal HTTP server** (`vedax serve`) exposing `/upload`,
    `/search`, `/chat` so the system can be put behind any UI without
    Python on the client.

## Long-term — what would make it actually different

11. **Drishti benchmarked** against an established word-spotting
    benchmark (George Washington manuscripts, IAM Historical) so the
    transcription-free search becomes a defensible research claim.
12. **A causal-aware retriever** that can answer "why did X happen"
    using the corpus's local causal graph rather than just lexical /
    semantic similarity.
13. **A streaming on-disk index** so the engine can search a corpus
    larger than RAM with no service to set up.

## Explicit non-goals

- A vector database. The point of the project is to not need one.
- An LLM in the retrieval loop. Optional in the chat layer, never in
  the retriever.
- Beating fine-tuned rerankers (Cohere Rerank 3, BGE-Reranker-v2). They
  add an extra inference pass; the project's promise is CPU-only,
  training-free retrieval.
