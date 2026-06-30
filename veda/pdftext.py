"""Pure-stdlib PDF text extraction. No external libraries.

A PDF is a graph of objects; page content lives in (usually
zlib-compressed) streams of drawing operators, and text is shown by the
Tj / ' / " / TJ operators. Subset fonts map byte codes to characters via
ToUnicode CMaps. This module implements the subset of the spec needed to
get the text back out:

  * object scanner (no xref needed) + object-stream (ObjStm) expansion
  * PDF value parser: dicts, arrays, names, strings, numbers, refs
  * stream filters: FlateDecode (+PNG predictors), LZWDecode,
    ASCIIHexDecode, ASCII85Decode, RunLengthDecode
  * page-tree walk for correct page order, inherited resources
  * ToUnicode CMap parsing (bfchar / bfrange) for subset fonts
  * content-stream interpreter for the text operators

Scanned (image-only) PDFs have no text layer — that needs OCR and is out
of scope. Encrypted PDFs are not supported.
"""

import re
import zlib

_OBJ_RE = re.compile(rb"(?<![0-9])(\d+)\s+(\d+)\s+obj\b")
_WS = b"\x00\t\n\x0c\r "
_DELIM = b"()<>[]{}/%"


class Ref:
    __slots__ = ("num", "gen")

    def __init__(self, num, gen):
        self.num = num
        self.gen = gen

    def __repr__(self):
        return f"Ref({self.num},{self.gen})"


class Name(str):
    """A PDF /Name (distinct from text strings)."""


# --------------------------------------------------------------- values

def _skip_ws(data, pos):
    n = len(data)
    while pos < n:
        c = data[pos]
        if c in _WS:
            pos += 1
        elif c == 0x25:  # % comment
            while pos < n and data[pos] not in b"\r\n":
                pos += 1
        else:
            break
    return pos


def _parse_name(data, pos):
    pos += 1  # past '/'
    out = bytearray()
    n = len(data)
    while pos < n:
        c = data[pos]
        if c in _WS or c in _DELIM:
            break
        if c == 0x23 and pos + 2 < n:  # #xx hex escape
            try:
                out.append(int(data[pos + 1 : pos + 3], 16))
                pos += 3
                continue
            except ValueError:
                pass
        out.append(c)
        pos += 1
    return Name(out.decode("latin-1")), pos


_ESCAPES = {0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12,
            0x28: 40, 0x29: 41, 0x5C: 92}


def _parse_literal_string(data, pos):
    pos += 1  # past '('
    out = bytearray()
    depth = 1
    n = len(data)
    while pos < n:
        c = data[pos]
        if c == 0x5C:  # backslash
            pos += 1
            if pos >= n:
                break
            e = data[pos]
            if e in _ESCAPES:
                out.append(_ESCAPES[e])
                pos += 1
            elif 0x30 <= e <= 0x37:  # octal \ddd
                octal = 0
                for _ in range(3):
                    if pos < n and 0x30 <= data[pos] <= 0x37:
                        octal = octal * 8 + (data[pos] - 0x30)
                        pos += 1
                    else:
                        break
                out.append(octal & 0xFF)
            elif e in b"\r\n":  # line continuation
                pos += 1
                if e == 0x0D and pos < n and data[pos] == 0x0A:
                    pos += 1
            else:
                out.append(e)
                pos += 1
        elif c == 0x28:
            depth += 1
            out.append(c)
            pos += 1
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                return bytes(out), pos + 1
            out.append(c)
            pos += 1
        else:
            out.append(c)
            pos += 1
    return bytes(out), pos


def _parse_hex_string(data, pos):
    end = data.find(b">", pos + 1)
    if end < 0:
        end = len(data)
    digits = re.sub(rb"[^0-9A-Fa-f]", b"", data[pos + 1 : end])
    if len(digits) % 2:
        digits += b"0"
    return bytes.fromhex(digits.decode("ascii")), end + 1


_NUM_RE = re.compile(rb"[+-]?(?:\d+\.?\d*|\.\d+)")


def parse_value(data, pos):
    """Parse one PDF value at ``pos`` -> (value, new_pos)."""
    pos = _skip_ws(data, pos)
    if pos >= len(data):
        return None, pos
    c = data[pos]
    if data.startswith(b"<<", pos):
        pos += 2
        out = {}
        while True:
            pos = _skip_ws(data, pos)
            if data.startswith(b">>", pos) or pos >= len(data):
                return out, pos + 2
            if data[pos] != 0x2F:  # malformed: bail out of the dict
                return out, pos + 1
            key, pos = _parse_name(data, pos)
            val, pos = parse_value(data, pos)
            out[key] = val
    if c == 0x3C:  # '<' hex string
        return _parse_hex_string(data, pos)
    if c == 0x28:  # '('
        return _parse_literal_string(data, pos)
    if c == 0x2F:  # '/'
        return _parse_name(data, pos)
    if c == 0x5B:  # '['
        pos += 1
        out = []
        while True:
            pos = _skip_ws(data, pos)
            if pos >= len(data) or data[pos] == 0x5D:
                return out, pos + 1
            val, pos = parse_value(data, pos)
            out.append(val)
    if data.startswith(b"true", pos):
        return True, pos + 4
    if data.startswith(b"false", pos):
        return False, pos + 5
    if data.startswith(b"null", pos):
        return None, pos + 4
    m = _NUM_RE.match(data, pos)
    if m:
        text = m.group(0)
        pos = m.end()
        # Lookahead for "gen R" -> indirect reference.
        if b"." not in text:
            m2 = re.compile(rb"\s+(\d+)\s+R\b").match(data, pos)
            if m2:
                return Ref(int(text), int(m2.group(1))), m2.end()
            return int(text), pos
        return float(text), pos
    return None, pos + 1  # unknown byte: skip


# -------------------------------------------------------------- filters

def _png_unpredict(data, columns, colors=1, bpc=8):
    bpp = max(1, colors * bpc // 8)
    rowlen = bpp * columns
    out = bytearray()
    prev = bytearray(rowlen)
    pos = 0
    while pos + 1 + rowlen <= len(data) + rowlen and pos < len(data):
        ftype = data[pos]
        row = bytearray(data[pos + 1 : pos + 1 + rowlen])
        pos += 1 + rowlen
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            up = prev[i]
            ul = prev[i - bpp] if i >= bpp else 0
            if ftype == 1:
                row[i] = (row[i] + left) & 0xFF
            elif ftype == 2:
                row[i] = (row[i] + up) & 0xFF
            elif ftype == 3:
                row[i] = (row[i] + (left + up) // 2) & 0xFF
            elif ftype == 4:
                p = left + up - ul
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - ul)
                pred = left if pa <= pb and pa <= pc else \
                    up if pb <= pc else ul
                row[i] = (row[i] + pred) & 0xFF
        out.extend(row)
        prev = row
    return bytes(out)


def _lzw_decode(data):
    """PDF LZWDecode: 9..12-bit codes, 256=clear, 257=EOD."""
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    out = bytearray()
    bitbuf = bits = 0
    width = 9
    prev = None
    for byte in data:
        bitbuf = (bitbuf << 8) | byte
        bits += 8
        while bits >= width:
            bits -= width
            code = (bitbuf >> bits) & ((1 << width) - 1)
            if code == 256:
                table = table[:258]
                width = 9
                prev = None
                continue
            if code == 257:
                return bytes(out)
            if prev is None:
                entry = table[code]
            elif code < len(table):
                entry = table[code]
                table.append(prev + entry[:1])
            else:
                entry = prev + prev[:1]
                table.append(entry)
            out.extend(entry)
            prev = entry
            if len(table) + 1 >= (1 << width) and width < 12:
                width += 1
    return bytes(out)


def _rl_decode(data):
    out = bytearray()
    pos = 0
    while pos < len(data):
        length = data[pos]
        if length == 128:
            break
        if length < 128:
            out.extend(data[pos + 1 : pos + 2 + length])
            pos += 2 + length
        else:
            if pos + 1 < len(data):
                out.extend(data[pos + 1 : pos + 2] * (257 - length))
            pos += 2
    return bytes(out)


def decode_stream(meta, raw, resolve):
    """Apply the stream's /Filter chain."""
    filters = resolve(meta.get("Filter"))
    if filters is None:
        return raw
    if not isinstance(filters, list):
        filters = [filters]
    parms = resolve(meta.get("DecodeParms") or meta.get("DP"))
    if not isinstance(parms, list):
        parms = [parms] * len(filters)
    data = raw
    for filt, parm in zip(filters, parms):
        filt = str(resolve(filt) or "")
        parm = resolve(parm) or {}
        if filt == "FlateDecode":
            try:
                data = zlib.decompress(data)
            except zlib.error:
                try:
                    data = zlib.decompressobj().decompress(data)
                except zlib.error:
                    return b""
        elif filt == "LZWDecode":
            data = _lzw_decode(data)
        elif filt == "ASCIIHexDecode":
            digits = re.sub(rb"[^0-9A-Fa-f]", b"",
                            data.split(b">")[0])
            if len(digits) % 2:
                digits += b"0"
            data = bytes.fromhex(digits.decode("ascii"))
        elif filt == "ASCII85Decode":
            import base64
            body = data.split(b"~>")[0].translate(None, _WS)
            try:
                data = base64.a85decode(body, adobe=False)
            except ValueError:
                return b""
        elif filt == "RunLengthDecode":
            data = _rl_decode(data)
        else:
            return b""  # image filters (DCT, CCITT, ...): no text inside
        predictor = resolve(parm.get("Predictor")) if parm else None
        if predictor and predictor >= 10:
            data = _png_unpredict(
                data,
                resolve(parm.get("Columns")) or 1,
                resolve(parm.get("Colors")) or 1,
                resolve(parm.get("BitsPerComponent")) or 8,
            )
    return data


# -------------------------------------------------------------- document

class PdfDocument:
    def __init__(self, data):
        self.data = data
        self.raw = {}      # (num, gen) -> (meta_src_bytes, stream_bytes|None)
        self.parsed = {}   # (num, gen) -> python value
        self._scan_objects()
        self._expand_object_streams()

    # -- object table

    def _scan_objects(self):
        data = self.data
        for m in _OBJ_RE.finditer(data):
            num, gen = int(m.group(1)), int(m.group(2))
            start = m.end()
            end = data.find(b"endobj", start)
            if end < 0:
                end = len(data)
            body = data[start:end]
            stream = None
            s = body.find(b"stream")
            if s >= 0:
                head = body[:s]
                after = body[s + 6 :]
                if after.startswith(b"\r\n"):
                    after = after[2:]
                elif after.startswith(b"\n") or after.startswith(b"\r"):
                    after = after[1:]
                e = after.rfind(b"endstream")
                stream = after[:e] if e >= 0 else after
                body = head
            self.raw[(num, gen)] = (body, stream)

    def object(self, num, gen=0):
        key = (num, gen)
        if key in self.parsed:
            return self.parsed[key]
        if key not in self.raw:
            return None
        value, _ = parse_value(self.raw[key][0], 0)
        self.parsed[key] = value
        return value

    def resolve(self, value, depth=0):
        while isinstance(value, Ref) and depth < 32:
            value = self.object(value.num, value.gen)
            depth += 1
        return value

    def stream_bytes(self, ref):
        """Decoded stream content of an object."""
        value = ref
        if isinstance(value, Ref):
            key = (value.num, value.gen)
            meta = self.resolve(value)
            raw = self.raw.get(key, (b"", None))[1]
        else:
            return b""
        if raw is None or not isinstance(meta, dict):
            return b""
        return decode_stream(meta, raw, self.resolve)

    def _expand_object_streams(self):
        for key in list(self.raw):
            meta = self.object(*key)
            if not (isinstance(meta, dict)
                    and str(meta.get("Type")) == "ObjStm"):
                continue
            data = decode_stream(meta, self.raw[key][1] or b"",
                                 self.resolve)
            first = self.resolve(meta.get("First")) or 0
            count = self.resolve(meta.get("N")) or 0
            header = data[:first].split()
            for i in range(count):
                try:
                    num = int(header[2 * i])
                    off = int(header[2 * i + 1])
                except (IndexError, ValueError):
                    break
                value, _ = parse_value(data, first + off)
                self.parsed.setdefault((num, 0), value)

    # -- page tree

    def pages(self):
        """Page dicts in document order, with inherited resources."""
        root = None
        for key in self.raw:
            value = self.object(*key)
            if isinstance(value, dict) and \
                    str(value.get("Type")) == "Catalog":
                root = self.resolve(value.get("Pages"))
                break
        out = []
        if isinstance(root, dict):
            self._walk(root, {}, out, 0)
        if not out:  # fallback: any /Type /Page object, file order
            for key in sorted(set(self.raw) | set(self.parsed)):
                value = self.object(*key)
                if isinstance(value, dict) and \
                        str(value.get("Type")) == "Page":
                    out.append((value, {}))
        return out

    def _walk(self, node, inherited, out, depth):
        if depth > 64 or not isinstance(node, dict):
            return
        resources = self.resolve(node.get("Resources"))
        if isinstance(resources, dict):
            inherited = resources
        if str(node.get("Type")) == "Page":
            out.append((node, inherited))
            return
        kids = self.resolve(node.get("Kids")) or []
        for kid in kids:
            self._walk(self.resolve(kid), inherited, out, depth + 1)

    # -- fonts

    def font_cmaps(self, resources):
        """{font_name: (code_bytes, {code: text})} from /ToUnicode CMaps."""
        out = {}
        fonts = self.resolve((resources or {}).get("Font")) or {}
        if not isinstance(fonts, dict):
            return out
        for name, font_ref in fonts.items():
            font = self.resolve(font_ref)
            if not isinstance(font, dict):
                continue
            tu = font.get("ToUnicode")
            if isinstance(tu, Ref):
                cmap_src = self.stream_bytes(tu)
                if cmap_src:
                    out[name] = _parse_cmap(cmap_src)
        return out


def _parse_cmap(src):
    """Parse bfchar/bfrange sections -> (code_byte_length, mapping)."""
    mapping = {}
    code_len = 1

    def utf16(b):
        try:
            return b.decode("utf-16-be")
        except UnicodeDecodeError:
            return ""

    for m in re.finditer(rb"beginbfchar(.*?)endbfchar", src, re.S):
        for src_hex, dst_hex in re.findall(
                rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", m.group(1)):
            code = int(src_hex, 16)
            code_len = max(code_len, len(src_hex) // 2)
            mapping[code] = utf16(bytes.fromhex(dst_hex.decode().zfill(
                (len(dst_hex) + 3) // 4 * 4)))
    for m in re.finditer(rb"beginbfrange(.*?)endbfrange", src, re.S):
        body = m.group(1)
        for lo_h, hi_h, rest in re.findall(
                rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*"
                rb"(<[0-9A-Fa-f]*>|\[[^\]]*\])", body):
            lo, hi = int(lo_h, 16), int(hi_h, 16)
            code_len = max(code_len, len(lo_h) // 2)
            if rest.startswith(b"["):
                dsts = re.findall(rb"<([0-9A-Fa-f]*)>", rest)
                for i, dst in enumerate(dsts):
                    if lo + i <= hi and dst:
                        mapping[lo + i] = utf16(bytes.fromhex(
                            dst.decode().zfill((len(dst) + 3) // 4 * 4)))
            else:
                base_hex = rest[1:-1].decode()
                if base_hex:
                    base = int(base_hex, 16)
                    for code in range(lo, min(hi, lo + 65535) + 1):
                        mapping[code] = utf16(
                            (base + code - lo).to_bytes(
                                max(2, len(base_hex) // 2), "big"))
    return code_len, mapping


# ----------------------------------------------------- content streams

_OPERATOR_RE = re.compile(rb"[A-Za-z'\"][A-Za-z0-9*'\"]*")

_PRINTABLE = {i: chr(i) for i in range(32, 127)}
_PRINTABLE.update({i: bytes([i]).decode("latin-1") for i in range(160, 256)})


def _decode_shown(raw, cmap):
    if cmap:
        code_len, mapping = cmap
        out = []
        for i in range(0, len(raw) - code_len + 1, code_len):
            code = int.from_bytes(raw[i : i + code_len], "big")
            out.append(mapping.get(code, ""))
        return "".join(out)
    return "".join(_PRINTABLE.get(b, "") for b in raw)


def _extract_content_text(content, cmaps):
    """Run the text operators of one content stream.

    Td/TD/Tm shift the text cursor without ending a logical line; only a
    significant vertical movement, T* or ET starts a new line. Otherwise
    each kerning hop in a typeset 10-K would split every glyph onto its
    own line.
    """
    out = []
    operands = []
    font = None
    pos = 0
    n = len(content)
    while pos < n:
        pos = _skip_ws(content, pos)
        if pos >= n:
            break
        c = content[pos]
        if c in b"(<[/+-.0123456789":
            value, pos = parse_value(content, pos)
            operands.append(value)
            continue
        m = _OPERATOR_RE.match(content, pos)
        if not m:
            pos += 1
            continue
        op = m.group(0)
        pos = m.end()

        def _is_num(x):
            return isinstance(x, (int, float))

        if op == b"BI":  # inline image: skip to EI
            e = content.find(b"EI", pos)
            pos = e + 2 if e >= 0 else n
        elif op == b"Tf" and len(operands) >= 2:
            font = operands[-2]
        elif op in (b"Tj", b"'") and operands:
            if isinstance(operands[-1], bytes):
                out.append(_decode_shown(operands[-1], cmaps.get(font)))
            if op == b"'":
                out.append("\n")
        elif op == b'"' and operands:
            if isinstance(operands[-1], bytes):
                out.append("\n")
                out.append(_decode_shown(operands[-1], cmaps.get(font)))
        elif op == b"TJ" and operands and isinstance(operands[-1], list):
            for element in operands[-1]:
                if isinstance(element, bytes):
                    out.append(_decode_shown(element, cmaps.get(font)))
                elif _is_num(element) and element < -180:
                    out.append(" ")
        elif op in (b"Td", b"TD") and len(operands) >= 2 \
                and _is_num(operands[-1]) and _is_num(operands[-2]):
            ty = operands[-1]
            tx = operands[-2]
            # Typesetters position individual glyphs with small Td hops
            # (often 1-3 units). Only meaningful jumps warrant separators.
            if abs(ty) > 3:
                out.append("\n")
            elif tx > 5:
                out.append(" ")
        elif op == b"Tm" and len(operands) >= 6 \
                and all(_is_num(o) for o in operands[-6:]):
            # Tm only acts as a line break when the y-translation changes.
            if abs(operands[-1]) > 0.1:
                out.append("\n")
        elif op == b"T*":
            out.append("\n")
        operands = []
    return "".join(out)


# ------------------------------------------------------------- public

def extract_pdf_text(source):
    """Text of a PDF given a path or raw bytes. Pure stdlib."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        with open(source, "rb") as f:
            data = f.read()
    if b"/Encrypt" in data[-2048:] or b"/Encrypt" in data[:4096]:
        # Best effort: encrypted documents are out of scope.
        pass
    doc = PdfDocument(data)
    pages_text = []
    for page, inherited in doc.pages():
        resources = doc.resolve(page.get("Resources")) or inherited
        cmaps = doc.font_cmaps(resources if isinstance(resources, dict)
                               else {})
        contents = page.get("Contents")
        refs = contents if isinstance(contents, list) else [contents]
        chunks = []
        for ref in refs:
            if isinstance(ref, Ref):
                chunks.append(doc.stream_bytes(ref))
        text = _extract_content_text(b"\n".join(chunks), cmaps)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{2,}", "\n", text).strip()
        text = _heal_glyph_spacing(text)
        if text:
            pages_text.append(text)
    return "\n\n".join(pages_text)


def _heal_glyph_spacing(text):
    """Some typesetters position every glyph with a separate Td, leaving
    a single space between glyphs in the same word and a wider gap
    between words. If a line is dominated by single-character tokens,
    treat multi-space gaps as word boundaries and single-space gaps as
    intra-word glyph spacing."""
    out_lines = []
    for line in text.split("\n"):
        tokens = [t for t in line.split(" ") if t != ""]
        singles = sum(1 for t in tokens if len(t) == 1)
        if not (len(tokens) >= 3 and singles >= 0.6 * len(tokens)):
            out_lines.append(line)
            continue
        # Split into word-groups on runs of 2+ spaces, then fuse glyphs
        # within each group.
        word_groups = re.split(r" {2,}", line)
        words = []
        for group in word_groups:
            joined = group.replace(" ", "")
            if joined:
                words.append(joined)
        out_lines.append(" ".join(words))
    return "\n".join(out_lines)
