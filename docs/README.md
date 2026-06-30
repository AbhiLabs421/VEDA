# VEDA / VEDA-X Documentation

VEDA — **V**ectorless **E**mbedding-free **D**ocument **A**rchitecture.

A document retrieval and grounded-chat stack with a strict
zero-dependency core, designed to beat the standard RAG retriever on
real benchmarks without a vector database or a trained embedding model.

## Table of contents

1. [Getting started](./getting-started.md) — install and your first query.
2. [CLI reference](./cli.md) — every command (`veda.py`, `python -m
   vedax`, `python -m veda`).
3. [Python API](./api.md) — using the engine from your own code.
4. [Architecture](./architecture.md) — how every piece works internally.
5. [PDF extraction](./pdf.md) — the built-in PDF parser and the
   glyph-spacing healer.
6. [OCR + Drishti](./ocr.md) — the stdlib OCR pipeline and the
   transcription-free shape search.
7. [Grounding guards](./grounding.md) — abstention and citation
   verification against hallucinations.
8. [Evaluation methodology](./evaluation.md) — datasets, baselines,
   significance testing.
9. [Benchmark results](./results.md) — every measured number, no hype.
10. [Research notes](./research.md) — what is novel, what is prior art,
    what remains to be proven.
11. [Roadmap](./roadmap.md) — what comes next.

## Dependencies

| layer | dependencies |
|---|---|
| `veda` (core retrieval, PDF, OCR, image IO) | Python standard library only |
| `vedax` lexical-only mode | Python standard library only |
| `vedax` hybrid mode | `onnxruntime`, `tokenizers`, `numpy` |
| `vedax chat` | Python standard library (`urllib`) |
| `eval/` evaluation harness | same as `vedax` hybrid |

## One-line use

```sh
python veda.py "your question"
```

See [getting-started.md](./getting-started.md) for the full walkthrough.
