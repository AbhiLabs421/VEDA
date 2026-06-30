"""VEDA-X command line — hybrid search and grounded chat over your own files.

  python -m vedax ask <files/folders...> "query"      index + search, one shot
  python -m vedax compare <files/folders...> "query"  plain RAG vs VEDA-X
  python -m vedax index <files/folders...> -o my.vedax
  python -m vedax search my.vedax "query"
  python -m vedax repl <files/folders...>             interactive search loop

  python -m vedax chat <files/folders...> "query" \\
       --llm-url https://ollamagw.example.net --llm-model gpt-oss:20b
  python -m vedax chat <files/folders...>             interactive chat loop

Add --no-dense to run fully offline (lexical + hyperdimensional only).
The LLM URL / model / api / token can also be set via VEDAX_LLM_URL,
VEDAX_LLM_MODEL, VEDAX_LLM_API (ollama|openai), VEDAX_LLM_TOKEN.
"""

import argparse
import sys
import time

from .engine import VedaX
from .llm import llm_settings_from_env


def _print_hits(hits, width=110):
    if not hits:
        print("  (no results)")
        return
    for rank, hit in enumerate(hits, 1):
        snippet = hit["snippet"]
        if len(snippet) > width:
            snippet = snippet[:width] + "..."
        print(f"  {rank}. [{hit['file']}] {snippet}")


def _build(paths, use_dense):
    engine = VedaX(use_dense=use_dense)
    t0 = time.time()
    engine.add(*paths)
    engine._finalize()
    print(f"indexed {len(engine.chunks)} chunks from your documents "
          f"in {time.time() - t0:.1f}s "
          f"({'hybrid' if engine.use_dense else 'lexical-only'} mode)",
          file=sys.stderr)
    return engine


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vedax", description=__doc__)
    parser.add_argument("--no-dense", action="store_true",
                        help="skip the neural stage (fully offline)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("ask", "compare", "repl"):
        p = sub.add_parser(name)
        p.add_argument("paths", nargs="+")
        if name != "repl":
            p.add_argument("query")
        p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("index")
    p.add_argument("paths", nargs="+")
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("search")
    p.add_argument("index")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("chat", help="retrieve with VEDA-X and stream a "
                                    "grounded answer from your LLM")
    p.add_argument("paths", nargs="+")
    p.add_argument("query", nargs="?", help="one-shot; omit for chat loop")
    p.add_argument("-k", type=int, default=6)
    p.add_argument("--llm-url",   help="env: VEDAX_LLM_URL")
    p.add_argument("--llm-model", help="env: VEDAX_LLM_MODEL")
    p.add_argument("--llm-api",   choices=("ollama", "openai"),
                   help="env: VEDAX_LLM_API")
    p.add_argument("--llm-token", help="env: VEDAX_LLM_TOKEN")

    args = parser.parse_args(argv)
    use_dense = not args.no_dense

    if args.cmd == "ask":
        engine = _build(args.paths, use_dense)
        _print_hits(engine.search(args.query, k=args.k))
    elif args.cmd == "compare":
        engine = _build(args.paths, use_dense)
        both = engine.compare(args.query, k=args.k)
        print("\n--- plain RAG (dense retriever only) ---")
        _print_hits(both["plain_rag"])
        print("\n--- VEDA-X (hybrid + hyperdimensional expansion) ---")
        _print_hits(both["veda_x"])
    elif args.cmd == "index":
        engine = _build(args.paths, use_dense)
        engine.save(args.out)
        print(f"saved -> {args.out}", file=sys.stderr)
    elif args.cmd == "search":
        engine = VedaX.load(args.index)
        _print_hits(engine.search(args.query, k=args.k))
    elif args.cmd == "repl":
        engine = _build(args.paths, use_dense)
        print("type a query (empty line to quit)")
        while True:
            try:
                query = input("vedax> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                break
            t0 = time.time()
            _print_hits(engine.search(query, k=args.k))
            print(f"  ({(time.time() - t0) * 1000:.0f} ms)")
    elif args.cmd == "chat":
        settings = llm_settings_from_env(args)
        if not settings["url"]:
            sys.exit("error: --llm-url or VEDAX_LLM_URL is required for chat")
        engine = _build(args.paths, use_dense)
        print(f"LLM: {settings['model']} @ {settings['url']} "
              f"({settings['api']})", file=sys.stderr)

        def one_turn(query):
            hits_printed = False
            for kind, payload in engine.chat(query, settings, k=args.k):
                if kind == "hits":
                    print("\n--- retrieved (VEDA-X) ---", file=sys.stderr)
                    _print_hits(payload, width=80)
                    print("--- answer ---", file=sys.stderr)
                    hits_printed = True
                elif kind == "abstain":
                    print(f"\n[abstain] confidence={payload['confidence']} "
                          f"({', '.join(payload['reasons']) or 'low'})",
                          file=sys.stderr)
                    print(payload["message"])
                elif kind == "verification":
                    grounded = payload["grounded_fraction"]
                    badge = "OK" if grounded >= 0.8 else (
                        "WARN" if grounded >= 0.5 else "UNGROUNDED")
                    print(f"\n[citation check: {badge} "
                          f"grounded={grounded * 100:.0f}%]",
                          file=sys.stderr)
                    for s in payload["sentences"]:
                        if not s["supported"]:
                            print(f"  ! unsupported (citing {s['citations']}, "
                                  f"support={s['support']}): {s['sentence']}",
                                  file=sys.stderr)
                else:  # token
                    sys.stdout.write(payload)
                    sys.stdout.flush()
            print()
            if not hits_printed:
                print("(no retrieval hits — answer may be ungrounded)",
                      file=sys.stderr)

        if args.query:
            one_turn(args.query)
        else:
            print("chat mode — type a question (empty line to quit)",
                  file=sys.stderr)
            while True:
                try:
                    q = input("\nyou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q:
                    break
                one_turn(q)


if __name__ == "__main__":
    main()
