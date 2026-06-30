# Built-in PDF extraction

`veda/pdftext.py` is a self-contained PDF text extractor written against
the format specification, using only Python's standard library.

```python
from veda.pdftext import extract_pdf_text
text = extract_pdf_text("report.pdf")          # path or raw bytes
```

## Why reimplement it

The whole project's promise is a zero-dependency core. A third-party
PDF library would break that. Implementing the needed slice of the spec
turned out to be small enough to be worth doing in-house.

## What is implemented

| feature | status |
|---|---|
| Object scanner without xref | yes |
| `ObjStm` (compressed object streams) | yes |
| `FlateDecode` filter (+ PNG predictors 10..14) | yes |
| `LZWDecode` filter | yes |
| `ASCIIHexDecode`, `ASCII85Decode`, `RunLengthDecode` | yes |
| Page-tree walk in document order | yes |
| Inherited `/Resources` | yes |
| `ToUnicode` CMap, `bfchar` and `bfrange` sections | yes |
| Subset-font byte codes | yes (1- and 2-byte) |
| Text operators `Tj`, `'`, `"`, `TJ` | yes |
| Position operators `Td`, `TD`, `Tm`, `T*` | yes (see healer) |
| Inline images (`BI ... EI`) | skipped intelligently |
| `DCTDecode` (JPEG-in-PDF), `CCITTFaxDecode` | not text-bearing, returned empty |
| Encrypted PDFs | not supported |
| Scanned (image-only) PDFs | no text layer — use OCR (see [ocr.md](./ocr.md)) |

## The glyph-spacing healer

Industrial typesetters (LaTeX `pdftex`, the typesetter SEC uses for
10-K filings) often emit one `Tj` operator per glyph and position the
next glyph with a small `Td` hop. A naive extractor turns
`"Consolidated Statement"` into `"C o n s o l i d a t e d S t a t e m e n t"`.

`_heal_glyph_spacing` detects the pattern per line:

- If a line is dominated by single-character tokens, treat runs of
  two-or-more spaces as **word boundaries** (the wider `Td` hop) and
  runs of one space as **intra-word glyph spacing** (the small `Td`
  hop), and fuse the glyphs back into words.
- Otherwise the line is left untouched, so well-typeset PDFs are not
  damaged.

Example:

```
input   : 'T a b le  o f Contents'   # double space at the word break
output  : 'Table ofContents'         # glyphs fused; one merge slipped
```

The heuristic is intentionally conservative — it occasionally fuses two
adjacent words when the typesetter used only a single-space gap between
them (visible above: `ofContents` instead of `of Contents`). The
retrieval pipeline does not rely on exact tokenisation, and BM25 / dense
queries still match on the recovered tokens.

## Validated against real PDFs

| document | size | result |
|---|---|---|
| Mozilla pdf.js test "tracemonkey" academic paper | 14 pages, 1 MB | 80,405 chars in 0.2 s; 98.7% word overlap with `pypdf` |
| SEC 10-K filings (FinanceBench corpus) | 84 documents | every page extracted; numbers like `1,577` recoverable post-heal |

## API

```python
from veda.pdftext import PdfDocument, extract_pdf_text

# High-level
text = extract_pdf_text("file.pdf")

# Low-level for custom processing
with open("file.pdf", "rb") as f:
    doc = PdfDocument(f.read())
for page, inherited_resources in doc.pages():
    contents = page.get("Contents")          # PDF object refs
    # ...
```

Tests live in `tests/test_pdftext.py`. They build small PDFs by hand
(no library!) and exercise every filter and text operator.
