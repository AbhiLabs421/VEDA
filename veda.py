#!/usr/bin/env python3
"""One-line VEDA-X: search and answer over the current directory.

Usage:
    python veda.py "your question here"          # search current dir
    python veda.py path/to/folder "question"     # explicit folder
    python veda.py                               # interactive REPL
    python veda.py --chat "question"             # grounded LLM answer

It walks the directory, indexes every .txt/.md/.pdf/.png/... it can read
(plain text, real PDFs, scanned page images via the built-in OCR), then
ranks the most relevant chunks. With --chat (or VEDAX_LLM_URL set) it
streams a grounded answer from your LLM, citing sources inline.
"""

import os
import sys
import time

# Make sure we import the local packages even when run from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vedax import VedaX
from vedax.llm import llm_settings_from_env


def _parse():
    args = list(sys.argv[1:])
    chat = False
    if "--chat" in args:
        chat = True
        args.remove("--chat")
    path = "."
    query = None
    if args:
        # If the first arg is a real path, treat it as the folder.
        if os.path.exists(args[0]) and not args[0].endswith("?"):
            path = args.pop(0)
        if args:
            query = " ".join(args)
    return path, query, chat


def _show(hits, width=110):
    if not hits:
        print("  (no results)")
        return
    for rank, hit in enumerate(hits, 1):
        snippet = " ".join(hit["snippet"].split())
        if len(snippet) > width:
            snippet = snippet[:width] + "..."
        rel = os.path.relpath(hit["file"])
        print(f"  {rank}. [{rel}] {snippet}")


def _build(path):
    print(f"indexing '{path}' ...", file=sys.stderr, flush=True)
    t0 = time.time()
    engine = VedaX(use_dense=True).add(path)
    engine._finalize()
    mode = "hybrid (neural + lexical + hyperdimensional)" \
        if engine.use_dense else "lexical + hyperdimensional"
    print(f"indexed {len(engine.chunks)} chunks in {time.time() - t0:.1f}s "
          f"({mode})\n", file=sys.stderr, flush=True)
    return engine


def _answer(engine, query, settings):
    print("\n--- retrieved (VEDA-X) ---", file=sys.stderr)
    hits_printed = False
    for kind, payload in engine.chat(query, settings, k=6):
        if kind == "hits":
            _show(payload, width=90)
            print("--- answer ---", file=sys.stderr, flush=True)
            hits_printed = True
        elif kind == "abstain":
            print(f"\n[abstain] confidence={payload['confidence']} "
                  f"({', '.join(payload['reasons']) or 'low'})",
                  file=sys.stderr)
            print(payload["message"])
        elif kind == "verification":
            g = payload["grounded_fraction"]
            badge = "OK" if g >= 0.8 else "WARN" if g >= 0.5 else "UNGROUNDED"
            print(f"\n[citation check: {badge} grounded={g * 100:.0f}%]",
                  file=sys.stderr)
            for s in payload["sentences"]:
                if not s["supported"]:
                    print(f"  ! unsupported (cited {s['citations']}, "
                          f"support={s['support']}): {s['sentence']}",
                          file=sys.stderr)
        else:  # token
            sys.stdout.write(payload)
            sys.stdout.flush()
    print()
    return hits_printed


def main():
    path, query, want_chat = _parse()
    engine = _build(path)
    settings = llm_settings_from_env()
    use_chat = want_chat or bool(settings["url"])

    if query:
        if use_chat:
            if not settings["url"]:
                sys.exit("error: --chat needs VEDAX_LLM_URL (or --llm-url)")
            _answer(engine, query, settings)
        else:
            _show(engine.search(query, k=5))
        return

    # REPL
    label = "chat" if use_chat else "search"
    print(f"{label} mode — type a question (empty to quit)", file=sys.stderr)
    while True:
        try:
            q = input(f"\n{label}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        t0 = time.time()
        if use_chat:
            _answer(engine, q, settings)
        else:
            _show(engine.search(q, k=5))
        print(f"  ({(time.time() - t0) * 1000:.0f} ms)", file=sys.stderr)


if __name__ == "__main__":
    main()
