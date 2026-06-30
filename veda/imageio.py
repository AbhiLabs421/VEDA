"""Pure-stdlib image reading: PNG, BMP, PGM/PPM -> grayscale.

Python's standard library has no image decoder, so this implements the
needed ones from the format specs. PNG inflation uses stdlib zlib; the
five PNG row filters are undone by hand.

An image is (width, height, pixels) with pixels a bytearray of
width*height grayscale values, row-major.
"""

import struct
import zlib


def load_image(source):
    """Load PNG/BMP/PGM/PPM from a path or bytes -> (w, h, gray bytes)."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        with open(source, "rb") as f:
            data = f.read()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _load_png(data)
    if data[:2] == b"BM":
        return _load_bmp(data)
    if data[:2] in (b"P2", b"P3", b"P5", b"P6"):
        return _load_pnm(data)
    raise ValueError("unsupported image format (PNG/BMP/PGM/PPM supported)")


def _gray(r, g, b):
    return (r * 299 + g * 587 + b * 114) // 1000


# ----------------------------------------------------------------- PNG

def _load_png(data):
    pos = 8
    width = height = None
    bit_depth = color_type = None
    palette = b""
    idat = []
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos : pos + 8])
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(
                ">IIBB", chunk[:10])
        elif ctype == b"PLTE":
            palette = chunk
        elif ctype == b"IDAT":
            idat.append(chunk)
        elif ctype == b"IEND":
            break
    if width is None:
        raise ValueError("PNG missing IHDR")
    if bit_depth != 8:
        raise ValueError(f"PNG bit depth {bit_depth} unsupported (8 only)")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"PNG color type {color_type} unsupported")
    raw = zlib.decompress(b"".join(idat))
    bpp = channels
    stride = width * channels
    pixels = bytearray(width * height)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        row = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += 1 + stride
        if ftype == 1:  # Sub
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + (left + prev[i]) // 2) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = row[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else b if pb <= pc else c
                row[i] = (row[i] + pred) & 0xFF
        prev = row
        base = y * width
        if color_type == 0:
            pixels[base : base + width] = row
        elif color_type == 2:
            for x in range(width):
                pixels[base + x] = _gray(row[3 * x], row[3 * x + 1],
                                         row[3 * x + 2])
        elif color_type == 3:
            for x in range(width):
                pi = row[x] * 3
                pixels[base + x] = _gray(palette[pi], palette[pi + 1],
                                         palette[pi + 2])
        elif color_type == 4:
            for x in range(width):
                g, a = row[2 * x], row[2 * x + 1]
                pixels[base + x] = (g * a + 255 * (255 - a)) // 255
        elif color_type == 6:
            for x in range(width):
                r, g, b, a = row[4 * x : 4 * x + 4]
                gray = _gray(r, g, b)
                pixels[base + x] = (gray * a + 255 * (255 - a)) // 255
    return width, height, pixels


def save_png(width, height, pixels, path):
    """Minimal grayscale PNG writer (for tests and demos)."""
    def chunk(ctype, body):
        out = struct.pack(">I", len(body)) + ctype + body
        return out + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF)

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: None
        raw.extend(pixels[y * width : (y + 1) * width])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw)))
            + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)


# ----------------------------------------------------------------- BMP

def _load_bmp(data):
    offset = struct.unpack("<I", data[10:14])[0]
    hsize = struct.unpack("<I", data[14:18])[0]
    width, height = struct.unpack("<ii", data[18:26])
    bpp = struct.unpack("<H", data[28:30])[0]
    compression = struct.unpack("<I", data[30:34])[0]
    if compression != 0 or bpp not in (8, 24, 32):
        raise ValueError("only uncompressed 8/24/32-bit BMP supported")
    flip = height > 0
    height = abs(height)
    palette = data[14 + hsize : offset] if bpp == 8 else b""
    rowsize = (width * bpp // 8 + 3) // 4 * 4
    pixels = bytearray(width * height)
    for y in range(height):
        src_y = height - 1 - y if flip else y
        row = data[offset + src_y * rowsize :]
        base = y * width
        if bpp == 8:
            for x in range(width):
                pi = row[x] * 4
                pixels[base + x] = _gray(palette[pi + 2], palette[pi + 1],
                                         palette[pi])
        else:
            step = bpp // 8
            for x in range(width):
                b, g, r = row[x * step : x * step + 3]
                pixels[base + x] = _gray(r, g, b)
    return width, height, pixels


# ------------------------------------------------------------- PGM/PPM

def _load_pnm(data):
    tokens = []
    pos = 0
    while len(tokens) < 4 and pos < len(data):
        # skip whitespace and comments
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        if pos < len(data) and data[pos] == 0x23:  # '#'
            while pos < len(data) and data[pos] not in b"\r\n":
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    magic = tokens[0]
    width, height = int(tokens[1]), int(tokens[2])
    maxval = int(tokens[3])
    pos += 1  # single whitespace after maxval
    pixels = bytearray(width * height)
    if magic == b"P5":
        body = data[pos : pos + width * height]
        for i in range(width * height):
            pixels[i] = body[i] * 255 // maxval
    elif magic == b"P6":
        body = data[pos : pos + 3 * width * height]
        for i in range(width * height):
            pixels[i] = _gray(body[3 * i], body[3 * i + 1], body[3 * i + 2])
    else:  # ASCII P2/P3
        values = data[pos:].split()
        if magic == b"P2":
            for i in range(width * height):
                pixels[i] = int(values[i]) * 255 // maxval
        else:
            for i in range(width * height):
                pixels[i] = _gray(int(values[3 * i]), int(values[3 * i + 1]),
                                  int(values[3 * i + 2]))
    return width, height, pixels
