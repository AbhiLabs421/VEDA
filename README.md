# VEDA — Vectorless Embedding-free Document Architecture

Semantic search over arbitrarily large documents with:

- **No vector database** — the index is compact in-memory signatures +
  posting arrays; optionally saved as one portable file.
- **No trained embedding model** — the embedding table is replaced by a
  hash function: any token's vector is *regenerated on demand* from its bytes.
- **No external dependencies** — pure Python standard library. Not even numpy.

The document indexes itself: meaning is learned on the fly from the text
being ingested.

## Documentation

Full documentation lives in [`docs/`](./docs/README.md):

1. [Getting started](./docs/getting-started.md) — install and your first query
2. [CLI reference](./docs/cli.md) — every command and flag
3. [Python API](./docs/api.md) — using the engine from your own code
4. [Architecture](./docs/architecture.md) — how every piece works
5. [PDF extraction](./docs/pdf.md) — the built-in PDF parser
6. [OCR + Drishti](./docs/ocr.md) — stdlib OCR and shape search
7. [Grounding guards](./docs/grounding.md) — abstention + citation verification
8. [Evaluation methodology](./docs/evaluation.md) — datasets, metrics, protocol
9. [Benchmark results](./docs/results.md) — every measured number
10. [Research notes](./docs/research.md) — novel vs prior art
11. [Roadmap](./docs/roadmap.md) — what comes next

## Quick start: `python veda.py "your question"`

The whole stack — retrieval, optional grounded LLM chat over your files —
runs from a single script. Drop into any folder and ask:

```


# 1. just retrieval (current directory; works on .txt/.md/.pdf/.png scans)
python veda.py "kitne din ka penalty lagega late delivery pe"

# 2. point at a specific folder
python veda.py ~/Documents/contracts "what is the arbitration clause"

# 3. grounded chat with your LLM (Ollama gateway / OpenAI-compatible)
export VEDAX_LLM_URL=https://ollamagw.example.net
export VEDAX_LLM_MODEL=gpt-oss:20b
python veda.py "summarise the quarterly report"

# 4. interactive REPL (search or chat depending on env)
python veda.py
```

What happens, step by step:
  1. **Walk** the folder, pick up every `.txt`/`.md`/`.pdf`/`.png` etc.
  2. **Extract** text — PDFs via our pure-stdlib parser (`veda/pdftext.py`),
     page scans via our pure-stdlib OCR (`veda/ocr.py`).
  3. **Index** — chunk, encode with hash-generated hypervectors + a
     MiniLM embedding (if `onnxruntime` is installed); else fall back to
     lexical + hyperdimensional only.
  4. **Search** — BM25 + dense + hyperdimensional query expansion + PRF,
     fused with the RRF weights tuned on NFCorpus.
  5. **(optional) Chat** — top-k chunks are pasted into a system-prompted
     conversation with your LLM, which streams a cited answer.

### Pure low-level use (if you want the package, not the script)

```
pip install <nothing>                # the core is stdlib only

python -m veda ask big_file.txt "what was the final decision"
python -m veda index *.txt -o corpus.veda
python -m veda search corpus.veda "query" -k 5
python -m veda repl notes.txt

python demo.py                  # tiny semantic demo
python bench.py                 # 1,000,000-token benchmark
python -m unittest discover -s tests -v
```

```python
from veda import Veda

engine = Veda()
engine.add_file("big_report.txt")        # streams; RAM stays bounded
for hit in engine.search("revenue declined because of supply issues"):
    print(hit["score"], hit["snippet"])
```

## Measured: 10 lakh (1,000,000) tokens on one CPU core

Benchmark (`bench.py`): a 6.5 MB synthetic haystack with 12 distinctive
"needle" sentences hidden inside; queries are paraphrase-ish word subsets,
not exact strings.

| metric            | result                          |
|-------------------|---------------------------------|
| ingest            | ~50 s (~20,000 tokens/s)        |
| chunks indexed    | 19,236                          |
| query latency     | ~20 ms                          |
| recall@5          | **12/12 needles**               |
| index size        | ~21 MB (in RAM, no server)      |

Pure CPython, single process, zero dependencies. (A numpy fast path would
be ~50–100x faster, but the whole point is needing nothing.)

## How it works

```
                      token bytes
                          │ blake2b (deterministic, stateless)
                          ▼
              sparse ternary hypervector            ← "embedding without a model"
                          │
        co-occurrence accumulation (random indexing,
        sampled, bounded memory)                     ← "semantics without training"
                          │ superposition (bundling)
                          ▼
            chunk signature (int8-quantised,
            ~3 bytes/entry) over (doc,start,end)     ← "the document is the storage"
                          │
        anchor coordinates → posting arrays          ← candidate generation
        query probes anchors of its words,
        stems and learned context → votes
                          │
                          ▼
            full cosine rescore of best-voted        ← ranking
```

1. **Hash-generated hypervectors** (`veda/hypervector.py`). Each token maps
   to a sparse ternary vector (64 of 2048 coordinates, ±1) derived from
   `blake2b(token)`. Near-orthogonal by construction — they behave like
   embedding rows, but there is no table to store or train.

2. **On-the-fly semantics** (`veda/encoder.py`). Words accumulate the
   hypervectors of their neighbours (sampled under a fixed budget, slab-
   flushed, pruned and evicted — memory stays constant for any input
   size). A query for `doctor` expands into the coordinates of
   `physician` if they ever co-occurred in the ingested text. The
   expansion happens on the **query side**, so ingest stays cheap.

3. **Anchor-voting index** (`veda/index.py`). Every chunk posts the
   leading hash coordinates ("anchors") of its informative words into
   stdlib posting arrays; stopwords are frequency-damped out. A query
   probes the anchors of its words, their stems and their learned context;
   best-voted chunks get a full cosine rescore against their quantised
   holographic signature. Sublinear search, and a rare "needle" chunk is
   found through the exact coordinates the query probes — no dilution.

4. **Streaming ingest** (`veda/engine.py`). `add_file` reads block by
   block; only `(doc, start, end)` spans and signatures are kept, snippets
   are read back straight off the disk.

5. **Built-in PDF extraction** (`veda/pdftext.py`) — also pure stdlib, no
   pypdf/pdfminer. Implements the needed slice of the PDF spec: object
   scanning (xref-free), ObjStm expansion, Flate/LZW/A85/AHx/RunLength
   filters with PNG predictors, page-tree walk, ToUnicode CMaps for
   subset fonts, and a text-operator interpreter. On a real 14-page
   academic paper it recovers 98.7% of pypdf's words in 0.2s. (Scanned
   image-only PDFs need OCR — out of scope; encrypted PDFs unsupported.)

## Built-in OCR + Drishti: search scans without transcribing them

`veda/imageio.py` + `veda/ocr.py`, pure stdlib (Python has no image
decoder — so PNG/BMP/PGM/PPM decoding is implemented from the specs,
PNG via zlib + hand-undone row filters):

- **OCR**: Otsu binarization → despeckle → line/word/glyph segmentation
  (connected components with vertical merge) → scale-normalized template
  matching against a 5x7 bitmap font embedded in the code. Perfect on
  clean machine-printed renders at any scale, ~90%+ characters at 1%
  salt-and-pepper noise. `python demo_ocr.py` to see it.

- **Drishti (the distinctive piece)**: searches a scanned page **without
  transcribing it**. Every word image gets a holographic signature from
  its visual shape (quantized glyph silhouettes + ink profiles bundled
  into hypervectors); the query is rendered with the embedded font and
  matched by cosine. Where OCR misreads a noisy word and grep then finds
  nothing, shape matching just loses a little score: in the demo, a line
  OCR mangles into `PA,,ME:(T TER::S` is still hit by its query at 0.97+.
  (Word spotting is an established research field; a zero-dependency
  implementation unified with hyperdimensional retrieval is ours.)

- Scanned page images (.png/.bmp/.pgm/.ppm) are first-class documents in
  `vedax` — they are OCR'd and indexed next to your .txt and .pdf files.

Honest scope: clean machine-printed text in the embedded-font style.
Handwriting and arbitrary typefaces genuinely need trained models — that
is the line where zero-dependency ends, and it is stated rather than
hidden. JPEG scans need a JPEG decoder (roadmap).

## Use it on YOUR documents (txt / md / pdf / scanned images)

The benchmark-winning pipeline is packaged as a tool — point it at your
own files and see for yourself:

```
pip install onnxruntime tokenizers numpy   # once (PDF support is built in, zero-dep)

# index + ask, one shot (files, folders, PDFs — mix freely)
python -m vedax ask  ~/my_documents "what is the penalty for late delivery"

# THE killer command: plain RAG vs VEDA-X on the same query, side by side
python -m vedax compare ~/my_documents "kitna penalty lagega late delivery pe"

# build once, query many times
python -m vedax index ~/my_documents -o my.vedax
python -m vedax search my.vedax "arbitration clause"

# interactive session
python -m vedax repl ~/my_documents

# fully offline (no model download, lexical + hyperdimensional only)
python -m vedax --no-dense ask ~/my_documents "health checkup limit"
```

The first hybrid run downloads the MiniLM model (~90 MB) once to /tmp.
Everything runs locally on CPU — your documents never leave the machine.

## Chat with your files (any Ollama / OpenAI-compatible LLM)

VEDA-X retrieves the most relevant chunks from your folder, then streams
a grounded answer from the LLM you point it at — entirely over stdlib
`urllib`, no SDK, no LangChain.

```
export VEDAX_LLM_URL=your ollama url 
export VEDAX_LLM_MODEL=model           # default also gpt-oss:20b

# one-shot
python -m vedax chat . "what does the contract say about penalties"

# interactive chat loop over a folder
python -m vedax chat ~/my_documents

# OpenAI-compatible gateway instead
python -m vedax chat . "summarise quarterly report" \
    --llm-url https://api.openai.com --llm-api openai --llm-model gpt-4o-mini
```

The system prompt instructs the model to answer only from the retrieved
context and cite sources `[1] [2]`. Both Ollama (`/api/chat`) and OpenAI
(`/v1/chat/completions`) streaming protocols are supported and unit-tested.

## FinanceBench (PageIndex's home turf): VEDA-X vs BM25 vs MiniLM

PageIndex (Vectorless RAG) markets itself on FinanceBench — SEC 10-K
filings, the most structured PDFs in the wild. We ran our **page-level
retrieval** harness on all 150 open-source questions with content-aligned
gold pages, against the same baselines:

| system | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| BM25 | 0.113 | 0.187 | 0.207 |
| all-MiniLM-L6-v2 (RAG retriever) | 0.147 | 0.240 | 0.313 |
| **VEDA-X** | **0.153** | **0.267** | **0.320** |

VEDA-X beats both baselines on **every metric**, on PageIndex's chosen
benchmark — without any LLM in the retrieval loop, on one CPU core, with
weights tuned on a different dataset (NFCorpus). Reproduce with
`python -m eval.financebench`.

Honest framing: absolute numbers are modest (~15% R@1) because
FinanceBench questions are heavily paraphrased ("FY2018 capital
expenditure") while the answer page just shows a table line
("Purchases of property, plant and equipment (PP&E) 1,577") — a
genuinely hard retrieval setting. PageIndex's marketing 98.7% is a
different metric (whole-pipeline answer accuracy with their LLM
evaluator), not Recall@k, so the two numbers aren't directly comparable.
What we can say with evidence: **on retrieval against the standard RAG
baseline, on PageIndex's chosen benchmark, VEDA-X still wins**.

## VEDA-X: beating the RAG retriever — on three benchmarks, not one

The honest research question — does VEDA-X beat the standard RAG
retriever beyond a single dataset? — is settled by running the same
pipeline on three different BEIR domains, with all fusion weights tuned
on each dataset's validation split only:

| dataset | domain | BM25 | MiniLM (RAG) | **VEDA-X** | gain | p |
|---|---|---|---|---|---|---|
| NFCorpus | medical lay Q&A | 0.3062 | 0.3195 | **0.3522** | +0.0327 | <0.0001 |
| SciFact  | scientific claims | 0.8352 | 0.8177 | **0.8578** | +0.0401 | 0.0004 |
| FiQA     | financial Q&A    | 0.2309 | 0.3687 | **0.3799** | +0.0112 | 0.0082 |

nDCG@10, held-out test split. **VEDA-X beats the dense RAG retriever on
3/3 datasets** — each gain statistically significant by paired bootstrap.
On NFCorpus we beat the RAG retriever by **+10.2%** relative; on SciFact
by **+4.9%**; on FiQA by **+3.0%**. Reproduce with `python -m eval.generalize`.

The distinctive component is the **hyperdimensional query expansion**
(`eval/veda_x.py`): feedback terms are selected by the cosine between a
term's corpus-local hypervector context (VEDA's random-indexing
semantics) and the encoded query — RM3-style feedback where term
selection is semantic rather than purely frequency-based. On NFCorpus
alone it lifts BM25 from 0.3062 to 0.3330, beating the neural retriever
with no neural network in the loop.

Honest caveats: three datasets is a strong signal, not a final claim;
BEIR has 18 and the next step is to keep widening. PRF and rank fusion
are established techniques — the new piece is the hypervector-guided
term selection. Baselines reproduce published literature numbers (BM25
NFCorpus ≈ 0.32, MiniLM FiQA ≈ 0.37), so the harness is trustworthy.
The eval needs `pip install onnxruntime tokenizers numpy`; the VEDA
core itself stays zero-dependency.

## Honest positioning vs. the RAG stack

| | classic RAG (embeddings + vector DB) | VEDA |
|---|---|---|
| infra | model server / API + DB | one `.py` package, stdlib |
| cold start | embed entire corpus through a model | ~20k tokens/s on one core |
| storage | vectors persisted in a DB | nothing persisted (optional single file) |
| general semantics | strong (pretrained) | corpus-local only |
| works offline / air-gapped | usually no | always |

VEDA does **not** beat pretrained embeddings at general-knowledge
semantics — `car ↔ automobile` is only bridged if the corpus itself links
them. Where it wins is everything around that: zero setup, zero network,
zero storage, instant cold start, tiny footprint, full auditability.
The honest research claim to chase is **quality-per-byte-of-index and
quality-per-dependency**, benchmarked on BEIR against BM25 (must beat) and
MiniLM (must approach).

## Prior art

The ingredients are classical and we stand on them openly:
hyperdimensional computing / VSA (Kanerva), random indexing (Sahlgren),
SimHash/LSH, inverted indexes. The combination — hash-generated
zero-storage embeddings + streaming bounded-memory semantics + anchor
voting over holographic signatures, as one dependency-free engine — is
the contribution to validate.

## Limitations

- Corpus-local semantics (see above).
- Spans are byte offsets; snippets of non-ASCII text can clip at edges.
- Index is ~2–3x the raw text in RAM at default settings (tunable down
  via `leaf_top`; it buys the 12/12 recall).
- `light_stem` is English-ish; Devanagari tokenises but has no stemming.

## Roadmap

1. BEIR / Monash-style evaluation vs BM25 and MiniLM.
2. Incremental postings (no rebuild on first search after ingest).
3. Multiprocessing ingest (stdlib `multiprocessing`, keeps zero-dep).
4. Proper multilingual stemming; cross-document semantic decay.
