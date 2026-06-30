"""Smoke tests for the pure-stdlib .xlsx / .docx text extractors.

We synthesise a minimal but VALID Office Open XML archive in memory
(zip of the required XML parts) and confirm the extractor reads it.
No external library, no precanned binary fixture.
"""

import io
import os
import sys
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veda.officetext import extract_xlsx, extract_docx
from vedax.extract import extract_text


XLSX_SHARED_STRINGS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    ' count="3" uniqueCount="3">'
    '<si><t>Step</t></si><si><t>Owner</t></si>'
    '<si><t>Trade Cancel Procedure</t></si>'
    '</sst>'
)
XLSX_WORKBOOK = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<sheets><sheet name="Steps" sheetId="1" /></sheets>'
    '</workbook>'
)
XLSX_SHEET1 = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<sheetData>'
    '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
    '<row r="2"><c r="A2" t="s"><v>2</v></c></row>'
    '</sheetData></worksheet>'
)


def _make_xlsx(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", XLSX_SHARED_STRINGS)
        z.writestr("xl/workbook.xml", XLSX_WORKBOOK)
        z.writestr("xl/worksheets/sheet1.xml", XLSX_SHEET1)


DOCX_DOCUMENT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/'
    'wordprocessingml/2006/main"><w:body>'
    '<w:p><w:r><w:t>Hello docx</w:t></w:r></w:p>'
    '<w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>'
    '</w:body></w:document>'
)


def _make_docx(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", DOCX_DOCUMENT)


class TestXlsx(unittest.TestCase):
    def test_extract(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.xlsx")
            _make_xlsx(p)
            text = extract_xlsx(p)
            self.assertIn("Sheet: Steps", text)
            self.assertIn("Step", text)
            self.assertIn("Owner", text)
            self.assertIn("Trade Cancel Procedure", text)

    def test_dispatched_through_extract_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.xlsx")
            _make_xlsx(p)
            self.assertIn("Trade Cancel", extract_text(p))


class TestDocx(unittest.TestCase):
    def test_extract(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.docx")
            _make_docx(p)
            text = extract_docx(p)
            self.assertIn("Hello docx", text)
            self.assertIn("Second paragraph", text)

    def test_dispatched_through_extract_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.docx")
            _make_docx(p)
            self.assertIn("Hello docx", extract_text(p))


if __name__ == "__main__":
    unittest.main()
