"""VEDA benchmark — 1,000,000-token (10 lakh) document, end to end.

Builds a synthetic ~1M-token haystack with Zipf-distributed filler words,
hides 12 distinctive "needle" sentences at random positions, then measures:
ingest speed, index size, query latency, and recall@5 on needle queries
(queries are paraphrase-ish word subsets, not exact strings).

Run:  python bench.py [--tokens 1000000]
"""

import argparse
import os
import random
import sys
import tempfile
import time

from veda import Veda

NEEDLES = [
    ("the violet meteor crashed into the ancient lighthouse while the keeper slept",
     "meteor hitting the old lighthouse"),
    ("a hidden waterfall behind the marble temple feeds the sacred turquoise pool",
     "waterfall behind temple sacred pool"),
    ("the chess grandmaster sacrificed his queen to deliver a stunning checkmate",
     "grandmaster queen sacrifice checkmate"),
    ("engineers cooled the quantum processor with liquid helium below two kelvin",
     "quantum processor liquid helium cooling"),
    ("the orchestra rehearsed the symphony under candlelight during the blackout",
     "orchestra playing symphony in blackout"),
    ("smugglers buried the emerald crown beneath the third oak in the cemetery",
     "emerald crown buried under oak tree"),
    ("the marathon runner collapsed two meters before the finish line ribbon",
     "runner collapsing near finish line"),
    ("volcanic ash grounded every flight across the northern hemisphere for days",
     "flights grounded by volcanic ash"),
    ("the librarian discovered a forgotten manuscript inside the hollow globe",
     "manuscript hidden inside the globe"),
    ("fishermen rescued the stranded whale calf during the midnight storm",
     "whale calf rescued in storm"),
    ("the spy memorized the launch codes reflected in the silver teapot",
     "spy reading codes in teapot reflection"),
    ("archaeologists unearthed bronze chariot wheels beneath the wheat field",
     "bronze chariot wheels found in field"),
]


def build_haystack(n_tokens, seed=7):
    rng = random.Random(seed)
    alphabet = "aeioubdgklmnprstvz"
    vocab = ["".join(rng.choice(alphabet) for _ in range(rng.randint(3, 9)))
             for _ in range(20000)]
    weights = [1.0 / rank for rank in range(1, len(vocab) + 1)]  # Zipf-ish
    words = rng.choices(vocab, weights=weights, k=n_tokens)

    # Splice needles in at random, well-separated positions.
    gap = n_tokens // (len(NEEDLES) + 1)
    for i, (needle, _) in enumerate(NEEDLES):
        at = gap * (i + 1) + rng.randint(0, gap // 2)
        words[at:at] = needle.split()
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1_000_000)
    args = parser.parse_args()

    print(f"building ~{args.tokens:,}-token haystack...", file=sys.stderr)
    text = build_haystack(args.tokens)
    size_mb = len(text) / 1e6
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(text)

    try:
        engine = Veda()
        t0 = time.time()
        engine.add_file(path, doc_id="haystack")
        t_ingest = time.time() - t0
        stats = engine.stats()

        # Warm-up build of the tree happens on first search.
        engine.search("warm up", k=1)

        found = 0
        latencies = []
        for needle, query in NEEDLES:
            t0 = time.time()
            hits = engine.search(query, k=5)
            latencies.append(time.time() - t0)
            marker = needle.split()[2]  # distinctive word from the needle
            if any(marker in h["snippet"] for h in hits):
                found += 1
            else:
                print(f"  MISS: {query!r}", file=sys.stderr)

        # Context: how long one naive full-text scan of the raw file takes.
        t0 = time.time()
        text.find("zzqx-not-present")
        t_scan = time.time() - t0

        avg_ms = 1000 * sum(latencies) / len(latencies)
        print()
        print(f"document        : {args.tokens:,} tokens ({size_mb:.1f} MB)")
        print(f"ingest          : {t_ingest:.1f} s "
              f"({args.tokens / t_ingest:,.0f} tokens/s)")
        print(f"chunks indexed  : {stats['chunks']:,}")
        print(f"index size      : {stats['index_bytes'] / 1e6:.1f} MB "
              f"({stats['index_bytes'] / len(text) * 100:.1f}% of text)")
        print(f"query latency   : {avg_ms:.1f} ms avg "
              f"(naive scan of raw text: {t_scan * 1000:.0f} ms)")
        print(f"recall@5        : {found}/{len(NEEDLES)}")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
