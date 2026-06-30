# Built-in OCR and Drishti shape search

`veda/ocr.py` is a pure-stdlib OCR engine, plus an experimental search
mode that finds words on a scanned page **without transcribing them**.

```python
from veda.ocr import ocr_image, Drishti
from veda.imageio import load_image

# Classic OCR
text = ocr_image("scanned_page.png")

# Transcription-free shape search
drishti = Drishti()
drishti.add_page("scan_1", "scanned_page.png")
for hit in drishti.search("PENALTY"):
    print(hit["page"], hit["box"], hit["score"])
```

## What "pure stdlib" means here

Python's standard library has no image decoder, so this layer ships
with one (`veda/imageio.py`):

- PNG via `zlib` and hand-undone row filters; grayscale, RGB, palette
  and alpha channels all map to 8-bit grayscale.
- BMP (8/24/32-bit uncompressed).
- PGM and PPM (ASCII P2/P3, binary P5/P6).

Anything OCR needs from there is built in: histogram, threshold,
connected components, normalised templates.

## OCR pipeline

1. **Binarization** (Otsu) plus a one-pass salt-and-pepper despeckle.
2. **Line segmentation** by ink projection on the vertical axis.
3. **Component extraction**: 8-neighbour connected components within
   each line.
4. **Vertical merge** for dot-and-stem characters (`!`, `?`, `:` etc.).
5. **Word grouping** by horizontal gap relative to the line's modal
   character width.
6. **Recognition**: each glyph is normalised onto a 10×14 grid and
   matched against the templates with an aspect-ratio gate.

Templates are derived from a 5×7 bitmap font embedded in code. Because
the font is also used as the **renderer** for tests and demos, the
system carries its own ground truth: every test is end-to-end without
any third-party asset.

### Honest scope

- Clean machine-printed text in the embedded-font style: roundtrip is
  exact at any rendering scale; about 90%+ character accuracy under 1%
  salt-and-pepper noise.
- Handwriting and arbitrary typefaces (Times, Arial scans) genuinely
  require trained models — that is the line where zero-dependency ends.
- JPEG scans need a JPEG decoder (not yet implemented; on the roadmap).

## Drishti — search without transcribing

Drishti (दृष्टि, "sight") is the experimental shape-search layer:

- Every word image on a page is converted into a holographic
  signature that bundles a coarse glyph silhouette and an ink-density
  profile into a hypervector.
- The query string is **rendered** with the embedded font into its own
  word image and converted into a signature the same way.
- Pages are searched by cosine similarity in hypervector space.

### Why this matters

When the OCR text layer reads `PA,,ME:(T TER::S ARE NET 30 DAYS` from
a noisy scan, `grep "PAYMENT"` finds nothing. Drishti is matching the
**shapes** of the words, so a few wrong glyphs only cost a little score:

```
OCR output line:    PA,,ME:(T TER::S ARE NET 30 DAYS FROM TNVOICE
Drishti search:
    PENALTY      score=1.000  -> FOUND
    ARBITRATION  score=0.976  -> FOUND
    ELEPHANT     score=0.297  -> not on page
```

Word spotting is an established research area; the contribution here is
a zero-dependency implementation unified with the same hyperdimensional
machinery that drives text retrieval, so a single index can rank scanned
images and text files together.

## Integration with `vedax`

Scanned page images (`.png`, `.bmp`, `.pgm`, `.ppm`) are first-class
documents in `vedax`: `VedaX.add()` OCRs them automatically and the
resulting text is indexed alongside `.txt`, `.md` and `.pdf` files.

```sh
python veda.py "what is the invoice total"
# indexed 47 chunks ... scan.png ... contract.pdf ... notes.txt
```

Tests live in `tests/test_ocr.py` (OCR roundtrip, noise tolerance,
multi-line text, Drishti behaviour) and `demo_ocr.py` for an
end-to-end demonstration.
