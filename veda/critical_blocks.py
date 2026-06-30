"""Critical-block parsing and atomic chunking helpers.

Some SOPs in a regulated environment cannot tolerate a partial-retrieval
failure mode: an incident-response runbook, a settlement-rollback
procedure, a regulatory disclosure list.  If the retriever returns
"Step 2 - Step 5" without "Step 1" the operator may take a destructive
action and the institution gets fined.

This module gives the indexer two ways to mark "this is atomic, never
split":

1. **Inline markers** anywhere inside an ordinary document::

       [[CRITICAL: Trade Cancel Procedure]]
       Step 1: Freeze the settlement queue
       Step 2: Notify risk desk
       Step 3: Get dual approval from CRO + CFO
       [[/CRITICAL]]

   The span between the opening and closing tags is treated as one
   indivisible unit.  Any chunk boundary that would fall inside the
   span is expanded so the whole block is in a single chunk.

2. **Critical folder convention**: every file dropped into
   ``./critical_sops/`` is treated as 100% critical — the whole file
   becomes one atomic chunk regardless of length.  The folder is
   configurable via ``documents.critical_fetch_dir`` in ``config.yaml``.

Public entry points:

    parse_critical_spans(text)        -> [(start, end, title), ...]
    is_critical_file(path, root)      -> bool
    expand_to_critical(start, end,
                       spans, n)      -> (start, end)
"""

import os
import re


_OPEN_RE = re.compile(
    r"\[\[\s*CRITICAL\s*:\s*(?P<title>[^\]\n]+?)\s*\]\]",
    re.IGNORECASE,
)
_CLOSE_RE = re.compile(
    r"\[\[\s*/\s*CRITICAL\s*\]\]",
    re.IGNORECASE,
)


def parse_critical_spans(text):
    """Return ``[(start_char, end_char, title), ...]`` for every
    ``[[CRITICAL: ...]] ... [[/CRITICAL]]`` block.

    Unclosed markers are tolerated (the unclosed block is ignored and
    a warning span list still returns what closed correctly) — we err
    on the side of indexing the document rather than refusing it.
    Nested blocks are flattened: the outermost block wins; inner
    markers are kept as literal text inside the outer span.
    """
    spans = []
    i = 0
    n = len(text)
    while i < n:
        m = _OPEN_RE.search(text, i)
        if not m:
            break
        title = m.group("title").strip()
        # find matching close — non-nested.  If a second OPEN appears
        # before CLOSE, the OUTER block extends to whichever CLOSE
        # finally appears, swallowing the inner.
        close = _CLOSE_RE.search(text, m.end())
        if not close:
            # unclosed: bail
            break
        spans.append((m.start(), close.end(), title))
        i = close.end()
    return spans


def is_critical_file(path, critical_root):
    """True if ``path`` lives inside the critical-folder root."""
    if not critical_root:
        return False
    try:
        ap = os.path.realpath(path)
        ar = os.path.realpath(critical_root)
    except OSError:
        return False
    return ap == ar or ap.startswith(ar + os.sep)


def find_span_for_offset(spans, offset):
    """Return the critical span ``(s, e, title)`` covering ``offset``,
    or ``None`` if the offset is outside every span.
    """
    for s, e, title in spans:
        if s <= offset < e:
            return (s, e, title)
    return None


def expand_to_critical(start_char, end_char, spans):
    """Given a tentative chunk range ``[start_char, end_char)`` in the
    source text, expand it so the chunk fully contains any critical
    span that it overlaps.

    Returns ``(new_start, new_end, matched_title or None)``.  Caller
    treats the returned chunk as a single retrieval unit and tags it
    as critical when ``matched_title`` is not None.
    """
    if not spans:
        return start_char, end_char, None
    new_start, new_end = start_char, end_char
    matched = None
    for s, e, title in spans:
        if s < new_end and e > new_start:  # any overlap
            if s < new_start:
                new_start = s
            if e > new_end:
                new_end = e
            matched = title  # last overlapped title wins (deterministic)
    return new_start, new_end, matched


def strip_markers(text):
    """Remove ``[[CRITICAL: ...]]`` / ``[[/CRITICAL]]`` tags from the
    chunk text so the markers don't leak into LLM context or UI.
    """
    text = _OPEN_RE.sub("", text)
    text = _CLOSE_RE.sub("", text)
    return text.strip()
