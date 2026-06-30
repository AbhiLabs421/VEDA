"""Tests for pure-stdlib image IO, OCR and Drishti shape search."""

import os
import tempfile
import unittest

from veda.imageio import load_image, save_png
from veda.ocr import Drishti, ocr_image, otsu_threshold, render_text


class TestImageIO(unittest.TestCase):
    def test_png_roundtrip(self):
        w, h, px = render_text("PNG TEST", scale=2)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            save_png(w, h, px, path)
            w2, h2, px2 = load_image(path)
            self.assertEqual((w, h), (w2, h2))
            self.assertEqual(bytes(px), bytes(px2))
        finally:
            os.unlink(path)

    def test_pgm_binary(self):
        data = b"P5\n# comment\n3 2\n255\n" + bytes([0, 128, 255, 10, 20, 30])
        w, h, px = load_image(data)
        self.assertEqual((w, h), (3, 2))
        self.assertEqual(list(px), [0, 128, 255, 10, 20, 30])

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            load_image(b"GIF89a not supported")


class TestOtsu(unittest.TestCase):
    def test_bimodal_split(self):
        pixels = bytes([10] * 50 + [240] * 50)
        t = otsu_threshold(pixels)
        self.assertTrue(10 <= t < 240)


class TestOCR(unittest.TestCase):
    def test_clean_roundtrip(self):
        img = render_text("HELLO WORLD 2026", scale=2)
        self.assertEqual(ocr_image(img), "HELLO WORLD 2026")

    def test_larger_scale(self):
        img = render_text("SCALE THREE", scale=3)
        self.assertEqual(ocr_image(img), "SCALE THREE")

    def test_multiline(self):
        img = render_text("FIRST LINE\nSECOND LINE", scale=2)
        self.assertEqual(ocr_image(img), "FIRST LINE\nSECOND LINE")

    def test_noise_tolerance(self):
        """Under 1% salt-and-pepper noise OCR is statistical: require
        >= 90% character accuracy rather than perfection."""
        import difflib
        expected = "NOISY SCAN TEST"
        img = render_text(expected, scale=2, noise=0.01, seed=7)
        ratio = difflib.SequenceMatcher(
            None, ocr_image(img), expected).ratio()
        self.assertGreaterEqual(ratio, 0.9)

    def test_punctuation_and_digits(self):
        img = render_text("NET 30 DAYS, OK?", scale=2)
        self.assertEqual(ocr_image(img), "NET 30 DAYS, OK?")

    def test_from_png_file(self):
        w, h, px = render_text("FROM A FILE", scale=2)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            save_png(w, h, px, path)
            self.assertEqual(ocr_image(path), "FROM A FILE")
        finally:
            os.unlink(path)


class TestDrishti(unittest.TestCase):
    def setUp(self):
        page = render_text(
            "THE VENDOR SHALL DELIVER MILESTONES\n"
            "PENALTY OF 2 PERCENT FOR DELAYS\n"
            "ARBITRATION HAPPENS IN NEW DELHI",
            scale=2, noise=0.005, seed=3)
        self.d = Drishti()
        self.d.add_page("scan1", page)

    def test_finds_word_without_transcription(self):
        hits = self.d.search("ARBITRATION", k=1)
        # ARBITRATION is on the third text line.
        self.assertEqual(hits[0]["page"], "scan1")
        self.assertGreater(hits[0]["score"], 0.55)

    def test_present_beats_absent(self):
        present = self.d.search("PENALTY", k=1)[0]["score"]
        absent = self.d.search("ELEPHANT", k=1)[0]["score"]
        self.assertGreater(present, absent + 0.2)


class TestVedaxImageIntegration(unittest.TestCase):
    def test_scanned_page_is_searchable(self):
        from vedax import VedaX
        with tempfile.TemporaryDirectory() as d:
            w, h, px = render_text(
                "INVOICE TOTAL IS 4500 RUPEES\nDUE IN 30 DAYS", scale=2)
            save_png(w, h, px, os.path.join(d, "scan.png"))
            with open(os.path.join(d, "notes.txt"), "w") as f:
                f.write("the meeting is on monday about hiring")
            engine = VedaX(use_dense=False).add(d)
            hits = engine.search("invoice total rupees", k=1)
            self.assertIn("scan.png", hits[0]["file"])


if __name__ == "__main__":
    unittest.main()
