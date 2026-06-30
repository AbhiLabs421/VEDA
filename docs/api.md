# Python API

Two entry classes, depending on whether you want the bare zero-dependency
core or the full hybrid pipeline.

## `vedax.VedaX` — the hybrid retrieval engine

The class behind every benchmark win in this repo.

```python
from vedax import VedaX

engine = VedaX(use_dense=True)        # set False for fully offline mode
engine.add("docs/contracts")          # walks files / directories / PDFs / images
engine.add("docs/notes.md", "extra/data.txt")

for hit in engine.search("late delivery penalty"):
    print(hit["score"], hit["file"], hit["snippet"])
```

### Construction

```python
VedaX(use_dense=True, chunk_tokens=120, overlap_tokens=20)
```

- `use_dense` — enable the MiniLM stage. Falls back automatically if
  `onnxruntime` is missing.
- `chunk_tokens` / `overlap_tokens` — chunk sizing in word tokens.

### Ingest

- `engine.add(*paths)` — index files or whole folders. Supported
  extensions: text-like (`.txt`, `.md`, `.csv`, `.json`, `.html`, `.log`,
  `.rst`), PDF (`.pdf`), images (`.png`, `.bmp`, `.pgm`, `.ppm`).
- Unsupported files are skipped silently.

### Query

- `engine.search(query, k=5)` — VEDA-X ranked list. Each result is a
  dict: `{"file": str, "snippet": str}`.
- `engine.search_plain_rag(query, k=5)` — plain dense retrieval only, for
  side-by-side comparisons.
- `engine.compare(query, k=5)` — returns `{"plain_rag": [...],
  "veda_x": [...]}`.

### Grounded chat

```python
from vedax.llm import llm_settings_from_env
settings = llm_settings_from_env()       # picks up VEDAX_LLM_URL, etc.

for kind, payload in engine.chat("what is the arbitration clause?", settings):
    if kind == "hits":
        print("retrieved:", [h["file"] for h in payload])
    else:                                # kind == "token"
        print(payload, end="", flush=True)
```

`settings` is a dict: `{"url": str, "model": str, "api": "ollama" |
"openai", "token": str | None}`.

### Persistence

```python
engine.save("contracts.vedax")           # single portable file
engine = VedaX.load("contracts.vedax")
```

## `veda.Veda` — the zero-dependency core

Pure-stdlib retrieval. No PDF, no images, no neural stage.

```python
from veda import Veda

engine = Veda()
engine.add_file("big_report.txt")        # streams; RAM stays bounded
for hit in engine.search("revenue decline supply issues"):
    print(hit["score"], hit["snippet"])
```

## Lower-level helpers

| function | module | purpose |
|---|---|---|
| `extract_pdf_text(path \| bytes)` | `veda.pdftext` | PDF -> text, stdlib only |
| `ocr_image(path \| (w,h,pixels))` | `veda.ocr` | OCR a page scan |
| `Drishti()` | `veda.ocr` | transcription-free shape search |
| `load_image(path \| bytes)` | `veda.imageio` | PNG / BMP / PGM / PPM |
| `save_png(w, h, pixels, path)` | `veda.imageio` | grayscale PNG writer |
| `BM25()` | `vedax.bm25` | the BM25 baseline |
| `MiniLM()` | `vedax.dense` | the ONNX dense encoder |
| `stream_chat(url, model, messages, ...)` | `vedax.llm` | stdlib LLM stream |

Every public helper is covered by the test suite (`python -m unittest
discover -s tests -v`, 41 tests at the time of writing).
