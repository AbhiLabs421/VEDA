"""Generate architecture & comparison diagrams as SVG (zero deps).

Produces three mentor-ready diagrams in docs/diagrams/:

  1. traditional_rag.svg       - what mainstream RAG looks like
  2. vedax_architecture.svg    - VEDA-X stack
  3. comparison_side_by_side.svg - both stacks on one page, with
                                   honest side-by-side comparison

SVG is the right choice: scales cleanly, opens in any browser, can be
embedded in slides, GitHub renders it inline, and we need no library to
emit it.
"""

import os
from xml.sax.saxutils import escape


# ──────────────────────────────────────────────────────────────────────
# Tiny SVG helper
# ──────────────────────────────────────────────────────────────────────

class SVG:
    def __init__(self, w, h, title=""):
        self.w = w
        self.h = h
        self.body = []
        self.defs = []
        self.title = title

    def add(self, s):
        self.body.append(s)

    # primitives ------------------------------------------------------
    def box(self, x, y, w, h, fill="#fff", stroke="#222",
            stroke_width=1.5, rx=8, opacity=1.0):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{stroke_width}" opacity="{opacity}"/>')

    def text(self, x, y, txt, size=14, fill="#111", anchor="middle",
             weight="normal", family="Inter,Segoe UI,Arial,sans-serif"):
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" '
                 f'font-family="{family}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}">'
                 f'{escape(txt)}</text>')

    def lines_in_box(self, x, y, w, h, lines, size=12, color="#222"):
        line_h = size + 4
        block_h = len(lines) * line_h
        start_y = y + (h - block_h) / 2 + size
        for i, line in enumerate(lines):
            self.text(x + w / 2, start_y + i * line_h, line,
                      size=size, fill=color)

    def labeled_box(self, x, y, w, h, title, lines=(), fill="#fff",
                    stroke="#222", title_color="#111", body_color="#444",
                    title_size=14, body_size=11, stroke_width=1.5,
                    rx=10):
        self.box(x, y, w, h, fill=fill, stroke=stroke,
                 stroke_width=stroke_width, rx=rx)
        self.text(x + w / 2, y + 22, title, size=title_size,
                  fill=title_color, weight="700")
        if lines:
            line_h = body_size + 3
            block_h = len(lines) * line_h
            start_y = y + 30 + (h - 30 - block_h) / 2 + body_size
            for i, line in enumerate(lines):
                self.text(x + w / 2, start_y + i * line_h, line,
                          size=body_size, fill=body_color)

    def arrow(self, x1, y1, x2, y2, color="#222", width=2,
              dashed=False, label=None):
        dash = ' stroke-dasharray="6,5"' if dashed else ''
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{color}" stroke-width="{width}"{dash} '
                 f'marker-end="url(#arrow)"/>')
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 - 5
            # white halo so the label is readable on top of any line
            self.add(f'<rect x="{mx-len(label)*3.5-4}" y="{my-12}" '
                     f'width="{len(label)*7+8}" height="16" rx="3" '
                     f'fill="#fff" opacity="0.95"/>')
            self.text(mx, my, label, size=11, fill=color, weight="600")

    def section_header(self, x, y, w, text, color="#111", fill="#f6f7fb"):
        self.box(x, y, w, 38, fill=fill, stroke="#dcdfe6",
                 stroke_width=1, rx=6)
        self.text(x + w / 2, y + 25, text, size=16, weight="700",
                  fill=color)

    def render(self):
        defs = """
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#222"/>
  </marker>
  <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f6feb"/>
  </marker>
  <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#c63d3d"/>
  </marker>
  <linearGradient id="paper" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#fafbfd"/>
    <stop offset="100%" stop-color="#eef1f7"/>
  </linearGradient>
</defs>
        """.strip()
        title_tag = (f'<title>{escape(self.title)}</title>'
                     if self.title else '')
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '
            f'{self.w} {self.h}" width="{self.w}" height="{self.h}" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif">'
            f'{title_tag}{defs}'
            f'<rect width="{self.w}" height="{self.h}" fill="url(#paper)"/>'
            + "".join(self.body) + '</svg>'
        )


# ──────────────────────────────────────────────────────────────────────
# Diagram 1: Traditional RAG
# ──────────────────────────────────────────────────────────────────────

def diagram_traditional_rag(path):
    s = SVG(1100, 640, title="Traditional RAG")
    s.text(550, 38, "Traditional RAG  —  the standard stack",
           size=22, weight="700")
    s.text(550, 62,
           "Embedding model + vector DB + fixed top-k + LLM "
           "(with hallucination risk)",
           size=13, fill="#555")

    # Pipeline row
    boxes = [
        ("Documents",      "PDF, TXT, web",         "#eef5ff"),
        ("Chunker",        "fixed-size chunks",     "#eef5ff"),
        ("Embedding\nModel",
                           "all-MiniLM-L6-v2,\nOpenAI-Ada, BGE…",
                                                    "#fff3e0"),
        ("Vector DB",      "Pinecone, Weaviate,\nChroma, Faiss",
                                                    "#fff3e0"),
        ("Fixed top-K\nretrieval",
                           "ALWAYS K chunks\n(K = 6 typical)",
                                                    "#fde9e9"),
        ("LLM",            "GPT-4, Claude,\nGemini",
                                                    "#fde9e9"),
    ]
    x = 30
    y = 140
    w = 165
    h = 95
    gap = 14
    for i, (title, body, fill) in enumerate(boxes):
        s.labeled_box(x, y, w, h, title,
                      tuple(body.split("\n")), fill=fill,
                      stroke="#a8b1bf")
        if i < len(boxes) - 1:
            s.arrow(x + w + 1, y + h / 2, x + w + gap - 1,
                    y + h / 2)
        x += w + gap

    # Problem callouts
    s.section_header(30, 280, 1040, "Where the problems live")

    issues = [
        ("Embedding API cost",
         "Every doc + every query pays for a remote model call.\n"
         "Inference cost is permanent and per-token.",
         "#fff3e0", 30),
        ("Vector DB infra",
         "Index server, replication, backups.\n"
         "Drift between document version and index.",
         "#fff3e0", 290),
        ("Fixed top-K dilution",
         "K=6 always — 5 noise chunks dilute the LLM's\n"
         "attention even when answer lives in 1.",
         "#fde9e9", 550),
        ("Hallucination",
         "LLM may invent facts; no built-in citation\n"
         "verification — answer can be confidently wrong.",
         "#fde9e9", 810),
    ]
    for title, body, fill, xi in issues:
        s.labeled_box(xi, 340, 250, 130, title,
                      tuple(body.split("\n")), fill=fill,
                      stroke="#a8b1bf",
                      title_color="#7a3500" if fill == "#fff3e0"
                      else "#933",
                      title_size=13, body_size=11)

    # Bottom row
    s.section_header(30, 500, 1040, "Dependencies")
    deps = ("Python 3.x   ·   PyTorch / Transformers   ·   Embedding model "
            "(100 – 500 MB)   ·   Vector DB server   ·   "
            "Internet for the LLM provider")
    s.text(550, 583, deps, size=13, fill="#333")
    s.text(550, 612,
           "Online dependencies, paid inference, no in-built grounding "
           "guard against hallucination.",
           size=12, fill="#666")

    with open(path, "w", encoding="utf-8") as f:
        f.write(s.render())


# ──────────────────────────────────────────────────────────────────────
# Diagram 2: VEDA-X Architecture
# ──────────────────────────────────────────────────────────────────────

def diagram_vedax(path):
    s = SVG(1200, 820, title="VEDA-X architecture")
    s.text(600, 38, "VEDA-X  —  intent-aware, adaptive, grounded RAG",
           size=22, weight="700")
    s.text(600, 62,
           "Hyperdimensional retrieval + query intent + adaptive cutoff + "
           "grounding guards.  Pure stdlib core.",
           size=13, fill="#555")

    # ── Section 1: indexing path ─────────────────────────────────────
    s.section_header(30, 90, 1140, "INDEXING  (one pass, no model, no DB)")

    idx_boxes = [
        ("Documents",
         "PDF · TXT · MD ·\nScanned images",     "#e7f3ff"),
        ("Built-in extractor",
         "veda.pdftext (stdlib)\nveda.ocr (stdlib)", "#e7f3ff"),
        ("Tokenizer",
         "+ light stemmer\n+ acronym normaliser",  "#e7f3ff"),
        ("Hash-hypervectors",
         "blake2b → sparse\nternary, zero storage",
                                                  "#e8f5e9"),
        ("Random indexing",
         "Corpus-local\nsemantic neighbours",     "#e8f5e9"),
        ("Anchor-vote\nposting index",
         "Sublinear search,\nstdlib arrays only",  "#e8f5e9"),
    ]
    x = 30
    y = 152
    w = 175
    h = 95
    gap = 14
    for i, (title, body, fill) in enumerate(idx_boxes):
        s.labeled_box(x, y, w, h, title,
                      tuple(body.split("\n")), fill=fill,
                      stroke="#869a9d")
        if i < len(idx_boxes) - 1:
            s.arrow(x + w + 1, y + h / 2,
                    x + w + gap - 1, y + h / 2,
                    color="#1f6feb", width=2)
        x += w + gap

    # ── Section 2: query path ────────────────────────────────────────
    s.section_header(30, 290, 1140, "QUERY  (per question, all in-process)")

    # 1. Intent decompose
    s.labeled_box(40, 360, 220, 130,
                  "1.  Intent decomposer",
                  ("Detects: define / list /",
                   "    procedure / yes-no / explain",
                   "Extracts: subject (real topic)",
                   "Strips: 'in single word',",
                   "    'plz', 'bhai', 'kya hai'…",
                   "Acronym-first detection",
                   "+ typo rescue (Levenshtein)"),
                  fill="#fff8e1", stroke="#a8b1bf",
                  title_color="#6a4500",
                  title_size=14, body_size=11)

    # 2. Hybrid retrieval
    s.labeled_box(280, 360, 220, 130,
                  "2.  Hybrid retrieval",
                  ("BM25 lexical",
                   "+ Hyperdimensional semantic",
                   "    (catches paraphrases like",
                   "     'hot' ↔ 'temperature')",
                   "+ optional MiniLM dense"),
                  fill="#e8f5e9", stroke="#869a9d",
                  title_color="#1b5e20",
                  title_size=14, body_size=11)

    # 3. Subject-focused rescore
    s.labeled_box(520, 360, 220, 130,
                  "3.  Subject-focused rescore",
                  ("Subject weight × 3",
                   "Fillers weight = 0",
                   "'single', 'define', 'briefly'",
                   "    CAN NOT hijack ranking"),
                  fill="#e8f5e9", stroke="#869a9d",
                  title_color="#1b5e20",
                  title_size=14, body_size=11)

    # 4. Adaptive cutoff
    s.labeled_box(760, 360, 220, 130,
                  "4.  Adaptive cutoff",
                  ("Replaces fixed top-K",
                   "Score-plateau detection",
                   "  clear winner → k = 1",
                   "  spread answer → k = 3-5",
                   "  off-topic     → k = 0"),
                  fill="#fde9e9", stroke="#c97e7e",
                  title_color="#933",
                  title_size=14, body_size=11)

    # 5. Grounding guards
    s.labeled_box(1000, 360, 170, 130,
                  "5.  Grounding",
                  ("Abstention guard",
                   "   subject-coverage",
                   "Citation verifier",
                   "   per-sentence",
                   "   support check"),
                  fill="#fde9e9", stroke="#c97e7e",
                  title_color="#933",
                  title_size=14, body_size=11)

    # arrows between query steps
    for x1, x2 in ((260, 280), (500, 520), (740, 760), (980, 1000)):
        s.arrow(x1, 425, x2, 425, color="#c63d3d", width=2)

    # ── Section 3: output / LLM ──────────────────────────────────────
    s.section_header(30, 520, 1140, "OUTPUT")
    s.labeled_box(40, 590, 320, 130,
                  "Grounded answer (no LLM)",
                  ("Single best snippet,",
                   "marked with file + span.",
                   "Zero hallucination by",
                   "construction."),
                  fill="#eef5ff", stroke="#869a9d",
                  title_size=14, body_size=12)
    s.text(200, 757, "for offline / air-gapped use",
           size=11, fill="#555")

    s.labeled_box(400, 590, 360, 130,
                  "Chat with grounded LLM",
                  ("System prompt forces 'answer only",
                   "from context, cite [1] [2]…'.",
                   "Streamed tokens, then citation",
                   "verifier marks each sentence",
                   "✓ supported / ✗ ungrounded."),
                  fill="#fff8e1", stroke="#a8b1bf",
                  title_size=14, body_size=12)
    s.text(580, 757,
           "for offline LLM (Ollama) or any OpenAI-style endpoint",
           size=11, fill="#555")

    s.labeled_box(800, 590, 370, 130,
                  "Abstain",
                  ("Confidence + subject-coverage",
                   "below threshold:",
                   "  → 'Not in the provided documents.'",
                   "  → LLM is NEVER called",
                   "Off-topic and prompt-injection safe."),
                  fill="#fde9e9", stroke="#c97e7e",
                  title_color="#933",
                  title_size=14, body_size=12)
    s.text(985, 757,
           "the safety net that makes RAG production-grade",
           size=11, fill="#555")

    # Footer summary
    s.text(600, 793,
           "Zero external libraries in the core  ·  zero vector database "
           " ·  zero pretrained embedding model required",
           size=12, weight="700", fill="#1f6feb")
    with open(path, "w", encoding="utf-8") as f:
        f.write(s.render())


# ──────────────────────────────────────────────────────────────────────
# Diagram 3: Side-by-side comparison
# ──────────────────────────────────────────────────────────────────────

def diagram_comparison(path):
    s = SVG(1400, 980, title="Traditional RAG vs VEDA-X")
    s.text(700, 40,
           "Traditional RAG   vs   VEDA-X — honest side-by-side",
           size=24, weight="700")
    s.text(700, 65,
           "Same input, same goal.  Where each stack wins / loses.",
           size=13, fill="#555")

    # Two columns
    col_w = 660
    col_x_left = 30
    col_x_right = 710

    s.section_header(col_x_left, 95, col_w,
                     "TRADITIONAL RAG", color="#7a3500")
    s.section_header(col_x_right, 95, col_w,
                     "VEDA-X (this repo)", color="#1b5e20")

    # Stages
    stages = [
        ("Document ingest",
         ("Run docs through a 100–500 MB",
          "embedding model API.",
          "Inference cost per token."),
         ("Pure-stdlib parser + OCR.",
          "Hash hypervectors — zero storage,",
          "regenerated from bytes on demand."),
         "#fff3e0", "#e8f5e9"),

        ("Storage",
         ("Vector DB (Pinecone / Weaviate /",
          "Chroma).  Separate server,",
          "backups, drift risk."),
         ("Compact signatures in RAM",
          "+ optional single portable file.",
          "Document is the storage."),
         "#fff3e0", "#e8f5e9"),

        ("Query understanding",
         ("Embed the query as one vector.",
          "No structure: 'define X in single",
          "word' ≈ 'X single word'."),
         ("Decompose: intent + subject + fillers.",
          "Acronym + Hinglish + typo aware.",
          "'plz', 'bhai', 'briefly' cannot",
          "hijack scoring."),
         "#fff3e0", "#e8f5e9"),

        ("Retrieval shape",
         ("FIXED top-K (usually 6).",
          "Always six chunks — even when",
          "answer lives in one."),
         ("ADAPTIVE cutoff.",
          "Clear winner → k=1.  Spread → k=3.",
          "Off-topic → k=0 (abstain).",
          "LLM context tightens with the truth."),
         "#fde9e9", "#e8f5e9"),

        ("Hallucination control",
         ("Hope the LLM behaves.",
          "Optional human review.",
          "No structural guarantee."),
         ("Confidence + subject-coverage guard:",
          "LLM is NEVER called on off-topic.",
          "Per-sentence citation verifier",
          "flags unsupported claims."),
         "#fde9e9", "#e8f5e9"),

        ("Dependencies",
         ("torch, transformers, sentence-",
          "transformers, faiss/chroma client,",
          "optional langchain, internet."),
         ("Core: Python stdlib only.",
          "Optional dense stage: onnxruntime",
          "+ tokenizers + numpy."),
         "#fff3e0", "#e8f5e9"),

        ("Cost / latency",
         ("Per-query embedding call.",
          "Per-query LLM tokens × top-K.",
          "Network RTT to provider."),
         ("Retrieval ≈ 2 ms (CPU).",
          "Smaller LLM prompts (adaptive k).",
          "Runs entirely offline."),
         "#fff3e0", "#e8f5e9"),
    ]

    y = 140
    h = 100
    for label, left_lines, right_lines, lfill, rfill in stages:
        # row label (centre)
        s.box(col_x_left + col_w + 4, y + 8, 700 - 2 * (col_w - 658),
              0, fill="none", stroke="none")
        s.text(700, y + h / 2 + 5, label, size=14, weight="700",
               fill="#222")

        # left = traditional
        s.labeled_box(col_x_left, y, col_w - 130, h, "",
                      tuple(left_lines),
                      fill=lfill, stroke="#a8b1bf",
                      title_size=0, body_size=12)
        # right = vedax
        s.labeled_box(col_x_right + 130, y, col_w - 130, h, "",
                      tuple(right_lines),
                      fill=rfill, stroke="#869a9d",
                      title_size=0, body_size=12)
        y += h + 8

    # Footer summary
    s.section_header(30, y + 8, 1340, "Honest bottom line")
    s.text(370, y + 60,
           "Modern RAG (Traditional)",
           size=14, weight="700", fill="#7a3500")
    s.text(370, y + 84,
           "Strong general semantics, slow + paid, "
           "hallucination by default, online infra.",
           size=12, fill="#444")
    s.text(1030, y + 60,
           "VEDA-X",
           size=14, weight="700", fill="#1b5e20")
    s.text(1030, y + 84,
           "Cheap, fast, offline, structurally cannot hallucinate; "
           "general semantics weaker than a big embedder — "
           "but for one document it does not need it.",
           size=12, fill="#444")

    with open(path, "w", encoding="utf-8") as f:
        f.write(s.render())


# ──────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "docs", "diagrams")
    os.makedirs(out, exist_ok=True)
    diagram_traditional_rag(os.path.join(out, "traditional_rag.svg"))
    diagram_vedax(os.path.join(out, "vedax_architecture.svg"))
    diagram_comparison(os.path.join(out, "comparison_side_by_side.svg"))
    print(f"wrote 3 SVGs into {out}")


if __name__ == "__main__":
    main()
