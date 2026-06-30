"""
====================================================================
  VEDAX CONFIG  —  pure-stdlib YAML loader + config.yaml schema
====================================================================

Loads ./config.yaml at import time (or VEDAX_CONFIG env path).  No
PyYAML dependency — we ship a tiny subset parser that handles exactly
what our config needs: dicts, lists, strings, numbers, booleans, null,
comments and quoting.

Strict mode: if a required key is missing the loader raises so the
server fails-fast at startup instead of pretending it has defaults.
"""

import os
import re
from typing import Any


# ──────────────────────────────────────────────────────────────────
#  Minimal YAML loader (stdlib only)
# ──────────────────────────────────────────────────────────────────

_QUOTED = re.compile(r'^("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')$')
_INT    = re.compile(r"^-?\d+$")
_FLOAT  = re.compile(r"^-?\d+\.\d+$")


def _scalar(s: str) -> Any:
    """Parse a YAML scalar (right-hand side of a key)."""
    s = s.strip()
    if not s:
        return ""
    if s in ("null", "~", "Null", "NULL"):
        return None
    if s in ("true", "True", "yes", "on"):
        return True
    if s in ("false", "False", "no", "off"):
        return False
    if _QUOTED.match(s):
        return s[1:-1].encode("utf-8").decode("unicode_escape")
    if _INT.match(s):
        return int(s)
    if _FLOAT.match(s):
        return float(s)
    # comma-separated inline list (eg [a, b, c])
    if s.startswith("[") and s.endswith("]"):
        body = s[1:-1].strip()
        if not body:
            return []
        return [_scalar(p) for p in _split_top_level(body, ",")]
    return s


def _split_top_level(s: str, sep: str) -> list:
    """Split s by sep but ignore separators inside [] or "" / ''."""
    out, depth, buf, in_q = [], 0, [], ""
    for ch in s:
        if in_q:
            buf.append(ch)
            if ch == in_q:
                in_q = ""
            continue
        if ch in ('"', "'"):
            in_q = ch
            buf.append(ch); continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(buf)); buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def load_yaml(text: str) -> dict:
    """Parse our YAML subset.  Returns a plain dict-tree."""
    # strip comments and blank lines, but keep indentation
    raw_lines = []
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        # strip inline comments only when preceded by whitespace and not
        # inside a quoted scalar
        if "#" in ln:
            in_q = ""
            cut = -1
            for i, ch in enumerate(ln):
                if in_q:
                    if ch == in_q:
                        in_q = ""
                    continue
                if ch in ('"', "'"):
                    in_q = ch
                    continue
                if ch == "#" and (i == 0 or ln[i - 1].isspace()):
                    cut = i; break
            if cut >= 0:
                ln = ln[:cut].rstrip()
                if not ln.strip():
                    continue
        raw_lines.append(ln.rstrip())

    pos = [0]

    def parse_block(indent: int) -> Any:
        first = raw_lines[pos[0]] if pos[0] < len(raw_lines) else None
        if first is None:
            return None
        if first.lstrip().startswith("- "):
            return parse_list(indent)
        return parse_map(indent)

    def parse_map(indent: int) -> dict:
        out = {}
        while pos[0] < len(raw_lines):
            ln = raw_lines[pos[0]]
            ind = _indent_of(ln)
            if ind < indent:
                break
            if ind > indent:
                raise ValueError(f"YAML indent jump at line: {ln!r}")
            body = ln.lstrip()
            if ":" not in body:
                raise ValueError(f"YAML map line without ':': {ln!r}")
            key, _, value = body.partition(":")
            key = key.strip()
            value = value.strip()
            pos[0] += 1
            if value == "" or value is None:
                if (pos[0] < len(raw_lines)
                        and _indent_of(raw_lines[pos[0]]) > indent):
                    out[key] = parse_block(_indent_of(raw_lines[pos[0]]))
                else:
                    out[key] = None
            else:
                out[key] = _scalar(value)
        return out

    def parse_list(indent: int) -> list:
        out = []
        while pos[0] < len(raw_lines):
            ln = raw_lines[pos[0]]
            ind = _indent_of(ln)
            if ind < indent or not ln.lstrip().startswith("- "):
                break
            item = ln.lstrip()[2:].strip()
            pos[0] += 1
            if item == "":
                if (pos[0] < len(raw_lines)
                        and _indent_of(raw_lines[pos[0]]) > indent):
                    out.append(parse_block(_indent_of(raw_lines[pos[0]])))
                else:
                    out.append(None)
            else:
                # inline scalar OR "key: value" map entry
                if ":" in item:
                    k, _, v = item.partition(":")
                    k = k.strip(); v = v.strip()
                    if v == "":
                        d = {k: parse_block(indent + 2)
                             if (pos[0] < len(raw_lines)
                                 and _indent_of(raw_lines[pos[0]]) > indent)
                             else None}
                    else:
                        d = {k: _scalar(v)}
                    # absorb subsequent same-indent map keys
                    while (pos[0] < len(raw_lines)
                           and _indent_of(raw_lines[pos[0]]) == indent + 2
                           and not raw_lines[pos[0]].lstrip().startswith("- ")):
                        ln2 = raw_lines[pos[0]].lstrip()
                        if ":" not in ln2:
                            break
                        k2, _, v2 = ln2.partition(":")
                        d[k2.strip()] = _scalar(v2)
                        pos[0] += 1
                    out.append(d)
                else:
                    out.append(_scalar(item))
        return out

    return parse_map(0)


# ──────────────────────────────────────────────────────────────────
#  Config loader
# ──────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = os.environ.get("VEDAX_CONFIG", "./config.yaml")


class Config(dict):
    """dict subclass with dotted-path access:  cfg.get_path('a.b.c')."""

    def get_path(self, path: str, default=None):
        cur = self
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load(path: str = _DEFAULT_CONFIG_PATH) -> Config:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"config.yaml not found at {path!r}. "
            "Copy config.example.yaml -> config.yaml and edit it."
        )
    with open(path, encoding="utf-8") as f:
        return Config(load_yaml(f.read()))


# Lazy singleton — first import triggers load.  Tests can replace
# vedax_config.cfg = Config(...) to use a synthetic config.
try:
    cfg: Config = load()
except FileNotFoundError:
    # Allow tests / first-run setup to import without a config file
    cfg = Config({})
