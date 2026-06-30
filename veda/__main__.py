"""VEDA command line — semantic search with zero dependencies.

  python -m veda ask <file> "query"            one-shot: index + search
  python -m veda index <file...> -o my.veda    build a persistent index
  python -m veda search my.veda "query"        query a saved index
  python -m veda repl <file>                   interactive search loop
"""

import argparse
import sys
import time

from .engine import Veda


def _print_hits(hits, width=100):
    if not hits:
        print("  (no results)")
        return
    for rank, hit in enumerate(hits, 1):
        snippet = hit["snippet"]
        if len(snippet) > width:
            snippet = snippet[:width] + "..."
        print(f"  {rank}. [{hit['score']:+.3f}] ({hit['doc']}) {snippet}")


def _build(files):
    engine = Veda()
    t0 = time.time()
    for path in files:
        engine.add_file(path)
    stats = engine.stats()
    print(f"indexed {stats['documents']} file(s), {stats['chunks']} chunks, "
          f"{stats['index_bytes'] / 1024:.0f} KB index, "
          f"{time.time() - t0:.1f}s", file=sys.stderr)
    return engine


def main(argv=None):
    parser = argparse.ArgumentParser(prog="veda", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask", help="index a file and search it in one shot")
    p.add_argument("file")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("index", help="build and save a persistent index")
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("search", help="query a saved index")
    p.add_argument("index")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("repl", help="interactive search over a file")
    p.add_argument("file")
    p.add_argument("-k", type=int, default=5)

    args = parser.parse_args(argv)

    if args.cmd == "ask":
        engine = _build([args.file])
        _print_hits(engine.search(args.query, k=args.k))
    elif args.cmd == "index":
        engine = _build(args.files)
        engine.save(args.out)
        print(f"saved -> {args.out}", file=sys.stderr)
    elif args.cmd == "search":
        engine = Veda.load(args.index)
        t0 = time.time()
        hits = engine.search(args.query, k=args.k)
        _print_hits(hits)
        print(f"  ({(time.time() - t0) * 1000:.0f} ms)", file=sys.stderr)
    elif args.cmd == "repl":
        engine = _build([args.file])
        print("type a query (empty line to quit)")
        while True:
            try:
                query = input("veda> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                break
            t0 = time.time()
            _print_hits(engine.search(query, k=args.k))
            print(f"  ({(time.time() - t0) * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
