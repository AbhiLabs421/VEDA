# Architecture

The system is two layers stacked on a strict zero-dependency core.

```
+--------------------------------------------------------------+
|  vedax  (hybrid pipeline + LLM chat + file extractors)       |
|     BM25  +  MiniLM (ONNX)  +  hyperdimensional expansion    |
|     +  pseudo-relevance feedback  +  RRF fusion              |
+--------------------------------------------------------------+
|  veda   (zero-dependency core)                               |
|     hash hypervectors  -> chunk signatures  -> voting index  |
|     PDF parser, image decoders, OCR, Drishti shape search    |
+--------------------------------------------------------------+
```

This document explains each component. For the experimental claims that
back the design, see [evaluation.md](./evaluation.md) and
[results.md](./results.md).

## Layer 1: the `veda` core

### Hash-generated hypervectors (`veda/hypervector.py`)

Every token maps to a sparse ternary vector — 64 nonzero positions out
of 2048, each `+1` or `-1` — derived deterministically from
`blake2b(token)`. Three consequences:

- No embedding table to store; vectors are regenerated from the bytes on
  demand. The "vocabulary cost" is exactly zero.
- Distinct tokens are near-orthogonal with overwhelming probability, so
  the vectors behave like trained embedding rows for superposition and
  cosine purposes.
- Generation is a deterministic byte stream, so a truncated `nnz=16`
  vector is a prefix of the full `nnz=64` vector. This lets us mix
  full-strength and truncated vectors in the same bundle for cheaper
  context vectors.

### Streaming distributional semantics (`veda/encoder.py`)

`SemanticMemory` implements bounded-memory **random indexing** (Magnus
Sahlgren, ~2005):

- Each token accumulates the hypervectors of its neighbours in a counter.
- Co-occurrence observations are sampled under a fixed budget and slab-
  flushed, so observation cost is constant for any input size.
- Per-token signatures are pruned to a top-N coordinate set; vocabulary
  is evicted at a maximum size. Memory is bounded regardless of input.

A query for `doctor` expands into the coordinates of `physician` if
those words ever co-occurred in the corpus. The expansion happens on
the **query side**, so ingest stays cheap.

### Anchor-voting index (`veda/index.py`)

The deterministic, sublinear retriever:

- Every chunk **posts** the leading hash coordinates ("anchors") of its
  informative words into a per-coordinate posting array. Stopwords are
  frequency-damped out.
- A query **probes** the anchors of its words, their light-stemmed
  forms, and its learned random-indexing context.
- Candidate chunks accumulate weighted votes, and the best-voted
  candidates get a full cosine rescore against their `int8`-quantised
  holographic signature.

Because the probe coordinates are exactly the post coordinates, a rare
"needle" chunk is never diluted. This is the key advantage over
summation-tree indexes for HD retrieval.

### Built-in PDF extraction (`veda/pdftext.py`)

A pure-stdlib subset of the PDF spec:

- Xref-free object scanner, with `ObjStm` (object stream) expansion for
  modern compressed PDFs.
- Filters: `FlateDecode` (zlib + all five PNG predictors), `LZWDecode`,
  `ASCIIHexDecode`, `ASCII85Decode`, `RunLengthDecode`.
- Page-tree walk in document order with inherited `/Resources`.
- `ToUnicode` CMap parsing (`bfchar` / `bfrange`) for subset fonts.
- Content-stream interpreter for the text operators
  (`Tj`, `'`, `"`, `TJ`).
- **Glyph-spacing healer**: SEC-grade typesetters emit one `Tj` per
  glyph with a small `Td` hop between letters and a wider one between
  words. The healer detects this pattern per line and fuses glyph runs
  back into words. See [pdf.md](./pdf.md).

### Built-in OCR + Drishti (`veda/ocr.py`)

- A 5×7 bitmap font is embedded in code; it serves both as the renderer
  (tests, demos, query synthesis) and as the recognition templates, so
  the system carries its own ground truth.
- OCR pipeline: Otsu binarization, salt-and-pepper despeckle, line
  segmentation by ink projection, connected-component glyph
  segmentation (with vertical merge for `!`, `?`, `:`), scale-normalised
  template matching with aspect-ratio gating.
- **Drishti**: search a scanned page **without transcribing it**. Word
  images get holographic signatures (quantised glyph silhouettes + ink
  profiles bundled into hypervectors); queries are rendered with the
  embedded font and matched by cosine. Survives noise that breaks
  `OCR + grep`. See [ocr.md](./ocr.md).

### Image decoders (`veda/imageio.py`)

Python's standard library has no image decoder, so this module decodes
from the formats' specifications:

- PNG: `zlib` inflate plus hand-undone row filters (Sub, Up, Average,
  Paeth). Grayscale / RGB / palette / alpha all map to grayscale output.
- BMP: uncompressed 8/24/32-bit.
- PGM / PPM: ASCII and binary variants.

## Layer 2: the `vedax` hybrid pipeline

The class `vedax.VedaX` runs four ranked retrievals per query and fuses
them:

1. **BM25** over the chunks.
2. **MiniLM** (ONNX, CPU) dense retrieval.
3. **BM25 + hyperdimensional query expansion** (the distinctive piece):
   feedback terms from the top first-pass chunks are scored by the
   cosine between the term's corpus-local hypervector context and the
   encoded query. The expanded query re-runs BM25.
4. **Dense pseudo-relevance feedback**: the query embedding is nudged
   toward the centroid of the top fused chunks and re-searched.

All four rankings are combined with weighted reciprocal-rank fusion. The
weights used at retrieval time are the ones tuned on the NFCorpus
validation split (`bm25_x: 0.25, dense_x: 1.0`).

### Chat layer (`vedax/llm.py`)

A stdlib `urllib` client that speaks both protocols:

- **Ollama**: `POST {url}/api/chat`, newline-delimited JSON stream.
- **OpenAI**: `POST {url}/v1/chat/completions`, Server-Sent-Events.

`VedaX.chat()` builds a system prompt that constrains the model to the
retrieved chunks and streams the tokens as they arrive.

## Layer 3: the `eval` harness

Three independent runs, each producing a reproducible table:

- `eval/run_eval.py` — single-dataset head-to-head on NFCorpus.
- `eval/generalize.py` — three BEIR datasets (NFCorpus, SciFact, FiQA)
  with paired-bootstrap significance testing.
- `eval/financebench.py` — page-level retrieval over the 150
  open-source FinanceBench questions, with content-aligned gold pages.

Baselines reproduce published numbers, so the harness is trustworthy.
See [evaluation.md](./evaluation.md) for the methodology.
