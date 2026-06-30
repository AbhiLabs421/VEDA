"""Pure-stdlib OCR + transcription-free shape search ("drishti").

No external libraries, no trained models. Three layers:

1. A bitmap font embedded in code (5x7 glyphs). It serves both as the
   renderer (tests, demos, query synthesis) and as the recognition
   templates — the system carries its own ground truth.
2. Classic OCR: Otsu binarization -> line segmentation by ink
   projection -> connected-component glyphs (with vertical merge for
   dots of '!', '?', ':') -> scale-normalized template matching.
3. The distinctive part, ``drishti``: search a scanned page WITHOUT
   transcribing it. Every word image gets a holographic signature from
   its visual shape (quantized glyph silhouettes + ink profiles bundled
   into a hypervector); the query is rendered with the embedded font and
   matched by cosine. Recognition errors that break classic OCR+grep
   degrade gracefully here, because shapes are matched, not labels.
   (Word spotting is an established research area; a zero-dependency
   implementation unified with hyperdimensional retrieval is ours.)

Honest scope: clean machine-printed text. Handwriting and arbitrary
typefaces need trained models; scanned-image quality OCR is exactly
where that line sits.
"""

import math
import random

from .hypervector import l2_dense, new_dense, token_hv

# ------------------------------------------------------------- the font
# 5x7 bitmaps, one int per row, bit 4 = leftmost column.

FONT = {
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F),
    "3": (0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    ",": (0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    "?": (0x0E, 0x11, 0x01, 0x06, 0x04, 0x00, 0x04),
    "!": (0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    "(": (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "/": (0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10),
}

GLYPH_W, GLYPH_H = 5, 7
NORM_W, NORM_H = 10, 14  # normalization grid for matching


def render_text(text, scale=2, margin=8, noise=0.0, seed=0):
    """Render text with the embedded font -> (w, h, gray pixels).
    Multi-line via '\\n'. ``noise`` flips that fraction of pixels."""
    lines = text.upper().split("\n")
    cols = max(len(line) for line in lines)
    width = margin * 2 + cols * (GLYPH_W + 1) * scale
    height = margin * 2 + len(lines) * (GLYPH_H + 3) * scale
    pixels = bytearray(b"\xff" * (width * height))
    for li, line in enumerate(lines):
        y0 = margin + li * (GLYPH_H + 3) * scale
        for ci, ch in enumerate(line):
            glyph = FONT.get(ch)
            if glyph is None:
                continue
            x0 = margin + ci * (GLYPH_W + 1) * scale
            for gy in range(GLYPH_H):
                row = glyph[gy]
                for gx in range(GLYPH_W):
                    if row & (1 << (GLYPH_W - 1 - gx)):
                        for dy in range(scale):
                            base = (y0 + gy * scale + dy) * width
                            for dx in range(scale):
                                pixels[base + x0 + gx * scale + dx] = 0
    if noise > 0:
        rng = random.Random(seed)
        for _ in range(int(noise * width * height)):
            i = rng.randrange(width * height)
            pixels[i] = 255 - pixels[i]
    return width, height, pixels


# --------------------------------------------------------- binarization

def otsu_threshold(pixels):
    hist = [0] * 256
    for p in pixels:
        hist[p] += 1
    total = len(pixels)
    total_sum = sum(i * h for i, h in enumerate(hist))
    sum_b = 0.0
    w_b = 0
    best_t, best_var = 128, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (total_sum - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def binarize(width, height, pixels):
    """-> set of (x, y) ink pixels (dark = ink)."""
    t = otsu_threshold(pixels)
    ink = set()
    for y in range(height):
        base = y * width
        for x in range(width):
            if pixels[base + x] <= t:
                ink.add((x, y))
    return _despeckle(ink)


def _despeckle(ink):
    """One salt-and-pepper pass: drop isolated ink, fill enclosed holes."""
    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1))
    cleaned = set()
    candidates = set()
    for x, y in ink:
        n = sum(1 for dx, dy in neighbours if (x + dx, y + dy) in ink)
        if n >= 1:
            cleaned.add((x, y))
        for dx, dy in neighbours:
            candidates.add((x + dx, y + dy))
    for x, y in candidates - ink:
        n = sum(1 for dx, dy in neighbours if (x + dx, y + dy) in ink)
        if n >= 7:
            cleaned.add((x, y))
    return cleaned


# --------------------------------------------------------- segmentation

def _lines(ink, height):
    rows = [0] * height
    for _, y in ink:
        rows[y] += 1
    lines = []
    start = None
    for y in range(height):
        if rows[y] and start is None:
            start = y
        elif not rows[y] and start is not None:
            lines.append((start, y))
            start = None
    if start is not None:
        lines.append((start, height))
    return lines


def _components(ink_line):
    """Connected components (8-neighbour) -> list of pixel sets."""
    remaining = set(ink_line)
    comps = []
    while remaining:
        seed_px = next(iter(remaining))
        stack = [seed_px]
        remaining.discard(seed_px)
        comp = {seed_px}
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    p = (x + dx, y + dy)
                    if p in remaining:
                        remaining.discard(p)
                        comp.add(p)
                        stack.append(p)
        comps.append(comp)
    return comps


def _bbox(comp):
    xs = [p[0] for p in comp]
    ys = [p[1] for p in comp]
    return min(xs), min(ys), max(xs), max(ys)


def _merge_vertical(boxes):
    """Merge components that overlap in x (dots of !, ?, :, etc.)."""
    boxes = sorted(boxes, key=lambda b: b[0][0])
    merged = []
    for box, comp in boxes:
        if merged:
            (px0, py0, px1, py1), pcomp = merged[-1]
            x0 = box[0]
            overlap = min(px1, box[2]) - max(px0, box[0]) + 1
            if overlap >= 0.5 * min(px1 - px0 + 1, box[2] - box[0] + 1):
                merged[-1] = ((min(px0, box[0]), min(py0, box[1]),
                               max(px1, box[2]), max(py1, box[3])),
                              pcomp | comp)
                continue
        merged.append((box, comp))
    return merged


def segment(width, height, pixels):
    """-> list of lines; each line is a list of words; each word is a
    list of glyphs (bbox, pixelset)."""
    ink = binarize(width, height, pixels)
    out = []
    for y0, y1 in _lines(ink, height):
        line_ink = {p for p in ink if y0 <= p[1] < y1}
        comps = [c for c in _components(line_ink) if len(c) > 2]
        glyphs = _merge_vertical([(_bbox(c), c) for c in comps])
        if not glyphs:
            continue
        widths = sorted(b[2] - b[0] + 1 for b, _ in glyphs)
        typical = widths[len(widths) // 2]
        words = [[glyphs[0]]]
        for prev, cur in zip(glyphs, glyphs[1:]):
            gap = cur[0][0] - prev[0][2]
            if gap > 0.9 * typical:
                words.append([cur])
            else:
                words[-1].append(cur)
        out.append(words)
    return out


# -------------------------------------------------------- recognition

def _normalize(comp, box):
    """Resample a glyph to the NORM_W x NORM_H binary grid (area max)."""
    x0, y0, x1, y1 = box
    gw = x1 - x0 + 1
    gh = y1 - y0 + 1
    grid = [0] * (NORM_W * NORM_H)
    for x, y in comp:
        nx = min(NORM_W - 1, (x - x0) * NORM_W // gw)
        ny = min(NORM_H - 1, (y - y0) * NORM_H // gh)
        grid[ny * NORM_W + nx] = 1
    return tuple(grid)


def _templates():
    out = {}
    for ch, glyph in FONT.items():
        comp = set()
        for gy in range(GLYPH_H):
            for gx in range(GLYPH_W):
                if glyph[gy] & (1 << (GLYPH_W - 1 - gx)):
                    comp.add((gx, gy))
        box = _bbox(comp)
        out[ch] = (_normalize(comp, box),
                   (box[2] - box[0] + 1) / (box[3] - box[1] + 1))
    return out


_TEMPLATES = _templates()


def classify_glyph(comp, box):
    """-> (char, confidence 0..1) by normalized template agreement."""
    grid = _normalize(comp, box)
    aspect = (box[2] - box[0] + 1) / (box[3] - box[1] + 1)
    best_ch, best_score = "?", -1.0
    for ch, (tgrid, taspect) in _TEMPLATES.items():
        agree = sum(1 for a, b in zip(grid, tgrid) if a == b)
        score = agree / len(grid)
        ratio = aspect / taspect if taspect else 1.0
        if ratio > 1:
            ratio = 1 / ratio
        score *= 0.7 + 0.3 * ratio  # aspect-ratio gate
        if score > best_score:
            best_ch, best_score = ch, score
    return best_ch, best_score


def ocr_image(source):
    """Image (path/bytes/(w,h,pixels)) -> recognized text."""
    if isinstance(source, tuple):
        width, height, pixels = source
    else:
        from .imageio import load_image
        width, height, pixels = load_image(source)
    lines_out = []
    for words in segment(width, height, pixels):
        line = []
        for word in words:
            line.append("".join(classify_glyph(c, b)[0] for b, c in word))
        lines_out.append(" ".join(line))
    return "\n".join(lines_out)


# ------------------------------------------- drishti: shape search

def _word_signature(word):
    """Holographic signature of a word IMAGE — no transcription.
    Bundles per-glyph silhouettes (position-bound) and coarse ink
    profiles, so noisy glyphs degrade the match instead of breaking it."""
    dense = new_dense()
    for i, (box, comp) in enumerate(word):
        grid = _normalize(comp, box)
        # Coarse 5x7 silhouette code: robust to small pixel noise.
        coarse = []
        for cy in range(7):
            for cx in range(5):
                count = 0
                for yy in range(2):
                    for xx in range(2):
                        count += grid[(cy * 2 + yy) * NORM_W + cx * 2 + xx]
                coarse.append(1 if count >= 2 else 0)
        code = "".join(map(str, coarse))
        for pos, sign in token_hv(f"{i}|{code}", role=3):
            dense[pos] += sign
        for pos, sign in token_hv(code, role=4):  # order-free fallback
            dense[pos] += 0.5 * sign
        # Ink profile of the glyph (column densities, 3 levels).
        col = [0] * 5
        for x, y in comp:
            col[min(4, (x - box[0]) * 5 // (box[2] - box[0] + 1))] += 1
        peak = max(col) or 1
        profile = "".join(str(min(2, c * 3 // (peak + 1))) for c in col)
        for pos, sign in token_hv(f"{i}~{profile}", role=5):
            dense[pos] += 0.7 * sign
    return dense


class Drishti:
    """Search scanned pages by shape, without transcribing them."""

    def __init__(self):
        self.words = []  # (page_id, bbox, signature, norm)

    def add_page(self, page_id, source):
        if isinstance(source, tuple):
            width, height, pixels = source
        else:
            from .imageio import load_image
            width, height, pixels = load_image(source)
        for words in segment(width, height, pixels):
            for word in words:
                x0 = min(b[0] for b, _ in word)
                y0 = min(b[1] for b, _ in word)
                x1 = max(b[2] for b, _ in word)
                y1 = max(b[3] for b, _ in word)
                sig = _word_signature(word)
                self.words.append((page_id, (x0, y0, x1, y1), sig,
                                   l2_dense(sig) or 1.0))

    def search(self, query_text, k=5):
        """Render the query with the embedded font, match by shape."""
        w, h, pixels = render_text(query_text, scale=2)
        segs = segment(w, h, pixels)
        if not segs or not segs[0]:
            return []
        qsig = _word_signature(segs[0][0])
        qnorm = l2_dense(qsig) or 1.0
        scored = []
        for page_id, box, sig, norm in self.words:
            dot = sum(a * b for a, b in zip(qsig, sig))
            scored.append((dot / (qnorm * norm), page_id, box))
        scored.sort(reverse=True)
        return [{"score": round(s, 3), "page": p, "box": b}
                for s, p, b in scored[:k]]
