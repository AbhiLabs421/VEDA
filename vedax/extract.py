"""Document text extraction: .txt / .md / .pdf / scanned images /
.xlsx / .docx.

Everything is pure stdlib: PDFs via veda.pdftext, scanned page images
(PNG/BMP/PGM/PPM) via the built-in OCR in veda.ocr, Office Open XML
(.xlsx / .docx) via veda.officetext (zipfile + ElementTree).
"""

import os

TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".log", ".json", ".html"}
IMAGE_EXTENSIONS = {".png", ".bmp", ".pgm", ".ppm"}
OFFICE_EXTENSIONS = {".xlsx", ".docx"}


def extract_text(path):
    """Plain text of one file; raises ValueError for unsupported types."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from veda.pdftext import extract_pdf_text
        return extract_pdf_text(path)
    if ext in IMAGE_EXTENSIONS:
        from veda.ocr import ocr_image
        return ocr_image(path)
    if ext == ".xlsx":
        from veda.officetext import extract_xlsx
        return extract_xlsx(path)
    if ext == ".docx":
        from veda.officetext import extract_docx
        return extract_docx(path)
    if ext in TEXT_EXTENSIONS or ext == "":
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"unsupported file type: {path}")


def iter_documents(paths):
    """Yield (doc_id, text) for files and recursively for directories.
    Unsupported/empty files are skipped with a warning."""
    import sys
    seen = set()
    stack = list(paths)
    while stack:
        path = stack.pop()
        if os.path.isdir(path):
            for name in sorted(os.listdir(path), reverse=True):
                if not name.startswith("."):
                    stack.append(os.path.join(path, name))
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            text = extract_text(path)
        except ValueError:
            continue
        except Exception as exc:  # unreadable/corrupt file: skip, keep going
            print(f"  ! skipping {path}: {exc}", file=sys.stderr)
            continue
        if text.strip():
            yield path, text
