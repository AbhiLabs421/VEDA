"""Demo: pure-stdlib OCR + transcription-free shape search (Drishti).

Creates a noisy synthetic "scanned page" PNG, then shows three things:
  1. OCR: the image becomes text again (no libraries, no models)
  2. Drishti: find a word ON the scan by shape, without transcribing
  3. vedax integration: the scan is searchable next to .txt/.pdf files

Run: python demo_ocr.py
"""

import os
import tempfile

from veda.imageio import save_png
from veda.ocr import Drishti, ocr_image, render_text

PAGE = ("AGREEMENT BETWEEN SHARMA ENTERPRISES AND VENDOR\n"
        "PENALTY OF 2 PERCENT APPLIES FOR LATE DELIVERY\n"
        "ARBITRATION SHALL HAPPEN IN NEW DELHI\n"
        "PAYMENT TERMS ARE NET 30 DAYS FROM INVOICE")


def main():
    width, height, pixels = render_text(PAGE, scale=2, noise=0.008, seed=11)
    scan_path = os.path.join(tempfile.gettempdir(), "contract_scan.png")
    save_png(width, height, pixels, scan_path)
    print(f"synthetic noisy scan written -> {scan_path} "
          f"({width}x{height})\n")

    print("1) OCR (stdlib only):")
    for line in ocr_image(scan_path).split("\n"):
        print("   ", line)

    print("\n2) Drishti — search the scan WITHOUT transcribing it:")
    drishti = Drishti()
    drishti.add_page("contract_scan", scan_path)
    for query in ("PENALTY", "ARBITRATION", "ELEPHANT"):
        hit = drishti.search(query, k=1)[0]
        verdict = "FOUND" if hit["score"] > 0.5 else "not on page"
        print(f"    {query:12s} score={hit['score']:.3f} "
              f"box={hit['box']}  -> {verdict}")

    print("\n3) Same scan inside a vedax document index:")
    from vedax import VedaX
    with tempfile.TemporaryDirectory() as d:
        os.link(scan_path, os.path.join(d, "scan.png"))
        with open(os.path.join(d, "notes.txt"), "w") as f:
            f.write("Meeting notes: discuss hiring and the office move.")
        engine = VedaX(use_dense=False).add(d)
        for hit in engine.search("penalty for late delivery", k=1):
            print(f"    [{os.path.basename(hit['file'])}] "
                  f"{hit['snippet'][:70]}...")


if __name__ == "__main__":
    main()
