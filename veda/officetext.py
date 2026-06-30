"""Pure-stdlib readers for office formats: .xlsx and .docx.

Both Microsoft Office formats are ZIP archives containing XML — no
proprietary binary, no third-party library needed.  We use Python's
``zipfile`` + ``xml.etree.ElementTree`` directly.

Public entry points:

    extract_xlsx(path)  -> str   (one block per sheet, then per row)
    extract_docx(path)  -> str   (paragraphs in document order; tables
                                  are rendered cell-by-cell with tabs)

Errors are not silently swallowed — corrupted files raise.
"""

import os
import re
import zipfile
import xml.etree.ElementTree as ET


# Namespaces used by Office Open XML.  We treat unknown namespaces
# leniently because Excel writes a moving target.
_X = {
    "xlsx_main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "docx_main": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def _localname(tag: str) -> str:
    """Drop the {namespace} from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


# ────────────────────────────────────────────────────────────────────
#  XLSX
# ────────────────────────────────────────────────────────────────────

def _col_letter_to_index(letter: str) -> int:
    n = 0
    for ch in letter:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def _load_shared_strings(z: zipfile.ZipFile) -> list:
    """Return list of strings indexed by sharedStringsTable position."""
    try:
        with z.open("xl/sharedStrings.xml") as f:
            tree = ET.parse(f)
    except KeyError:
        return []
    out = []
    for si in tree.getroot():
        # <si> may be a single <t> or several <r><t>...</t></r> runs.
        text_chunks = [
            (e.text or "")
            for e in si.iter()
            if _localname(e.tag) == "t"
        ]
        out.append("".join(text_chunks))
    return out


def _row_cells(row_elem, shared_strings: list):
    """Yield (col_index, value) for each <c> child of a <row>."""
    for c in row_elem:
        if _localname(c.tag) != "c":
            continue
        ref = c.get("r", "")
        col = _col_letter_to_index(ref) if ref else 0
        cell_type = c.get("t", "")
        # Find <v> or <is><t> child
        value = None
        for child in c:
            tag = _localname(child.tag)
            if tag == "v":
                value = (child.text or "")
                break
            if tag == "is":
                value = "".join((t.text or "") for t in child.iter()
                                if _localname(t.tag) == "t")
                break
        if value is None:
            continue
        if cell_type == "s":   # shared string
            try:
                value = shared_strings[int(value)]
            except (ValueError, IndexError):
                pass
        elif cell_type == "b":  # bool
            value = "true" if value == "1" else "false"
        # other types ('str', 'inlineStr', 'n', '', etc.) come through as-is
        yield col, value


def _sheet_names(z: zipfile.ZipFile):
    """Return list of (sheet_name, internal_relationship_target)."""
    try:
        with z.open("xl/workbook.xml") as f:
            tree = ET.parse(f)
    except KeyError:
        return []
    sheets = []
    for sheet in tree.getroot().iter():
        if _localname(sheet.tag) == "sheet":
            sheets.append((sheet.get("name") or "Sheet",
                           sheet.get("sheetId") or ""))
    return sheets


def extract_xlsx(path) -> str:
    """Read every sheet of an .xlsx into a single readable string."""
    with zipfile.ZipFile(path) as z:
        shared = _load_shared_strings(z)
        sheets = _sheet_names(z) or [("Sheet1", "1")]
        sheet_files = sorted(
            n for n in z.namelist()
            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        )
        out = []
        for idx, sheet_file in enumerate(sheet_files):
            sheet_name = sheets[idx][0] if idx < len(sheets) \
                                        else f"Sheet{idx+1}"
            with z.open(sheet_file) as f:
                tree = ET.parse(f)
            out.append(f"=== Sheet: {sheet_name} ===")
            for elem in tree.getroot().iter():
                if _localname(elem.tag) != "row":
                    continue
                cells = list(_row_cells(elem, shared))
                if not cells:
                    continue
                # Pad gaps with empty strings, then join with tabs
                last_col = max(c for c, _ in cells)
                row = [""] * (last_col + 1)
                for col, val in cells:
                    row[col] = str(val)
                line = "\t".join(row).rstrip()
                if line:
                    out.append(line)
            out.append("")
        return "\n".join(out).strip()


# ────────────────────────────────────────────────────────────────────
#  DOCX
# ────────────────────────────────────────────────────────────────────

def _docx_paragraph_text(p) -> str:
    """Concatenate text runs within a <w:p>, respecting tab/br runs."""
    pieces = []
    for child in p.iter():
        tag = _localname(child.tag)
        if tag == "t":
            pieces.append(child.text or "")
        elif tag == "tab":
            pieces.append("\t")
        elif tag == "br":
            pieces.append("\n")
    return "".join(pieces).strip()


def _docx_table_text(tbl) -> str:
    rows = []
    for tr in tbl:
        if _localname(tr.tag) != "tr":
            continue
        cells = []
        for tc in tr:
            if _localname(tc.tag) != "tc":
                continue
            cell_text = "\n".join(_docx_paragraph_text(p)
                                  for p in tc
                                  if _localname(p.tag) == "p")
            cells.append(cell_text.strip())
        rows.append("\t".join(cells))
    return "\n".join(rows).strip()


def extract_docx(path) -> str:
    with zipfile.ZipFile(path) as z:
        with z.open("word/document.xml") as f:
            tree = ET.parse(f)
    root = tree.getroot()
    # Body is the first <w:body> child
    body = None
    for child in root:
        if _localname(child.tag) == "body":
            body = child
            break
    if body is None:
        return ""
    out = []
    for child in body:
        tag = _localname(child.tag)
        if tag == "p":
            txt = _docx_paragraph_text(child)
            if txt:
                out.append(txt)
        elif tag == "tbl":
            t = _docx_table_text(child)
            if t:
                out.append(t)
    return "\n\n".join(out).strip()
