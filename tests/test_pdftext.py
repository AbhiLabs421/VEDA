"""Tests for the pure-stdlib PDF text extractor.

PDFs are assembled by hand here (no PDF library anywhere): a minimal
document is a catalog, a page tree, a page, a content stream and fonts.
Our parser scans objects directly, so no xref table is needed.
"""

import unittest
import zlib

from veda.pdftext import extract_pdf_text


def build_pdf(objects):
    """Assemble numbered objects into a PDF byte string."""
    out = bytearray(b"%PDF-1.4\n")
    for num, body in objects:
        out += b"%d 0 obj\n" % num
        out += body
        out += b"\nendobj\n"
    out += b"trailer << /Root 1 0 R >>\n%%EOF\n"
    return bytes(out)


def stream_obj(meta, data):
    return (meta + b"\nstream\n" + data + b"\nendstream")


CATALOG = (1, b"<< /Type /Catalog /Pages 2 0 R >>")
PAGES = (2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")


def page_obj(extra=b""):
    return (3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
               b"/Contents 4 0 R " + extra + b">>")


class TestPlainContent(unittest.TestCase):
    def test_uncompressed_literal_strings(self):
        content = (b"BT /F1 12 Tf 72 720 Td (Hello sansaar!) Tj "
                   b"0 -20 Td (Second line with \\(escapes\\)) Tj ET")
        pdf = build_pdf([
            CATALOG, PAGES, page_obj(),
            (4, stream_obj(b"<< /Length %d >>" % len(content), content)),
        ])
        text = extract_pdf_text(pdf)
        self.assertIn("Hello sansaar!", text)
        self.assertIn("Second line with (escapes)", text)

    def test_tj_array_with_kerning_spaces(self):
        content = b"BT [(Hel) -50 (lo) -400 (world)] TJ ET"
        pdf = build_pdf([
            CATALOG, PAGES, page_obj(),
            (4, stream_obj(b"<< >>", content)),
        ])
        text = extract_pdf_text(pdf)
        # Small kern joins, large kern becomes a space.
        self.assertIn("Hello world", text)


class TestFlateDecode(unittest.TestCase):
    def test_compressed_content_stream(self):
        content = b"BT (The compressed truth) Tj ET"
        packed = zlib.compress(content)
        pdf = build_pdf([
            CATALOG, PAGES, page_obj(),
            (4, stream_obj(b"<< /Filter /FlateDecode /Length %d >>"
                           % len(packed), packed)),
        ])
        self.assertIn("The compressed truth", extract_pdf_text(pdf))


class TestToUnicode(unittest.TestCase):
    def test_subset_font_cmap(self):
        """2-byte codes mapped through a bfchar/bfrange ToUnicode CMap —
        the way modern subset fonts work."""
        cmap = (b"/CIDInit /ProcSet findresource begin\n"
                b"begincmap\n"
                b"2 beginbfchar\n"
                b"<0001> <0056>\n"   # V
                b"<0002> <0045>\n"   # E
                b"endbfchar\n"
                b"1 beginbfrange\n"
                b"<0003> <0004> <0044>\n"  # D, (0x45 already taken: E->A)
                b"endbfrange\n"
                b"endcmap\n")
        content = b"BT /F1 12 Tf <00010002000300010004> Tj ET"
        pdf = build_pdf([
            CATALOG, PAGES,
            page_obj(b"/Resources << /Font << /F1 5 0 R >> >> "),
            (4, stream_obj(b"<< >>", content)),
            (5, b"<< /Type /Font /Subtype /Type0 /ToUnicode 6 0 R >>"),
            (6, stream_obj(b"<< /Length %d >>" % len(cmap), cmap)),
        ])
        # codes: 1->V 2->E 3->D(range lo) 1->V 4->E(range lo+1)
        self.assertIn("VEDVE", extract_pdf_text(pdf))


class TestRobustness(unittest.TestCase):
    def test_garbage_returns_empty_not_crash(self):
        self.assertEqual(extract_pdf_text(b"%PDF-1.4 garbage no objects"), "")

    def test_image_only_page_yields_no_text(self):
        content = b"q 612 0 0 792 0 0 cm /Im1 Do Q"
        pdf = build_pdf([
            CATALOG, PAGES, page_obj(),
            (4, stream_obj(b"<< >>", content)),
        ])
        self.assertEqual(extract_pdf_text(pdf), "")


if __name__ == "__main__":
    unittest.main()
