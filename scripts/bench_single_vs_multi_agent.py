"""Benchmark: single-agent vs multi-agent on the same query set.

Three variants compared on the same engine and queries:

  1. SINGLE     — current pipeline (1 retrieval)
  2. MULTI seq  — router -> specialist -> verifier, all sequential
  3. MULTI par  — N specialist agents (one per category) FIRED IN
                  PARALLEL through a ThreadPoolExecutor, then a merge
                  step picks the best.  This is what people usually
                  mean by 'multi-agent' in production — agents that
                  work concurrently, not one-after-another.

Three axes measured:

    1. Accuracy           — did the gold sentence appear in retrieved chunks?
    2. Latency            — total wall-clock per query
    3. Abstention quality — did the system correctly refuse on out-of-scope?

The multi-agent variant adds three extra steps:
  router    — picks the right category before retrieval
  specialist — runs the same retrieval but scoped to its category
  verifier  — second pass: re-runs retrieval with the LLM-style 'answer'
              as query to confirm the same chunks come back
"""

import concurrent.futures
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vedax_core as core
from vedax.intent import subject_coverage, decompose


# ───────────────────────────── single-agent
def single_agent(query: str) -> dict:
    t0 = time.time()
    res = core.do_retrieve(query, top_k=core.TOP_K)
    elapsed = time.time() - t0
    return {
        "elapsed_ms": round(elapsed * 1000, 1),
        "would_abstain": res.get("would_abstain", False),
        "chunks": res.get("chunks", []),
        "subject": res.get("subject", ""),
    }


# ───────────────────────────── multi-agent helpers
_CATEGORY_KEYWORDS = {
    "HR":         ["leave", "salary", "attendance", "policy", "employee",
                   "office hours", "bonus", "benefit", "casual", "sick"],
    "Finance":    ["invoice", "payment", "ledger", "accounting", "tax",
                   "audit", "budget", "expense"],
    "Compliance": ["regulation", "compliance", "kyc", "law", "audit",
                   "guideline", "rule"],
}


def router_agent(query: str) -> str | None:
    q = query.lower()
    scores = {}
    for cat, words in _CATEGORY_KEYWORDS.items():
        s = sum(w in q for w in words)
        if s:
            scores[cat] = s
    if not scores:
        return None
    return max(scores, key=scores.get)


def specialist_agent(query: str, category: str | None) -> dict:
    return core.do_retrieve(query, top_k=core.TOP_K, category=category)


def verifier_agent(query: str, first_pass: dict) -> bool:
    if not first_pass.get("chunks"):
        return True
    subject = first_pass.get("subject") or query
    second = core.do_retrieve(subject, top_k=3)
    if not second.get("chunks"):
        return False
    return first_pass["chunks"][0]["snippet"][:80] == \
           second["chunks"][0]["snippet"][:80]


def multi_agent_seq(query: str) -> dict:
    """Sequential — router -> specialist -> verifier."""
    t0 = time.time()
    category = router_agent(query)
    first = specialist_agent(query, category)
    verified = verifier_agent(query, first) if not first.get("would_abstain") \
                                            else True
    elapsed = time.time() - t0
    return {
        "elapsed_ms": round(elapsed * 1000, 1),
        "would_abstain": first.get("would_abstain", False),
        "chunks": first.get("chunks", []),
        "subject": first.get("subject", ""),
        "routed_to": category or "(none)",
        "verified": verified,
    }


# Shared executor so we measure steady-state cost, not thread spawn-up.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def multi_agent_par(query: str) -> dict:
    """Parallel — fire ALL specialist agents (one per category) plus the
    verifier concurrently in a thread pool, then merge in O(1).

    The merge picks whichever specialist returned the highest score.
    """
    t0 = time.time()
    categories = list(_CATEGORY_KEYWORDS.keys()) + [None]
    futures = {cat: _EXECUTOR.submit(specialist_agent, query, cat)
               for cat in categories}
    # also fire the verifier early using the raw query
    verifier_future = _EXECUTOR.submit(core.do_retrieve, query,
                                       core.TOP_K, None)

    # collect all specialists
    results = {}
    for cat, fut in futures.items():
        try:
            results[cat] = fut.result(timeout=2.0)
        except Exception:
            results[cat] = {"chunks": [], "would_abstain": True}
    # merge: pick the specialist whose top chunk has the highest implicit
    # signal (chunk text length is a fine proxy here since smart_search
    # returns adaptive cutoff)
    best_cat, best = None, None
    for cat, r in results.items():
        if r.get("would_abstain") or not r.get("chunks"):
            continue
        score = len(r["chunks"][0].get("snippet", ""))
        if best is None or score > best[0]:
            best = (score, cat, r); best_cat = cat
    chosen = best[2] if best else results.get(None, {})

    try:
        v = verifier_future.result(timeout=1.0)
        verified = bool(v.get("chunks"))
    except Exception:
        verified = False

    elapsed = time.time() - t0
    return {
        "elapsed_ms": round(elapsed * 1000, 1),
        "would_abstain": chosen.get("would_abstain", True),
        "chunks": chosen.get("chunks", []),
        "subject": chosen.get("subject", ""),
        "routed_to": best_cat or "(merged-global)",
        "verified": verified,
    }


# ───────────────────────────── benchmark
QUERIES = [
    ("casual leave kitne din",                 "casual leave",     False),
    ("how many sick leaves",                   "sick leave",       False),
    ("maternity leave",                        "maternity",        False),
    ("office hours",                           "9:30",             False),
    ("performance bonus",                      "bonus",            False),
    ("define casual leave in single word",     "casual leave",     False),
    ("yo bhai office hours kya hai",           "9:30",             False),
    ("you know who am i please explain bonus", "bonus",            False),
    ("what is the share price of Reliance",    None,               True),
    ("how do I cook biryani",                  None,               True),
]


SOP = """HR POLICY MANUAL 2024

Casual Leave: Every confirmed employee is entitled to 12 days of Casual
Leave per calendar year.
Sick Leave: Confirmed employees are entitled to 10 days of Sick Leave
per year.
Maternity Leave: 26 weeks of paid maternity leave is granted to female
employees.
Office hours are 9:30 AM to 6:30 PM, Monday through Friday.
Performance bonus ranges from 1 to 3 months of basic salary.
"""


def main():
    import tempfile
    fd, sop_path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(SOP)
    try:
        core.AUTO_FETCH_DIR = os.path.dirname(sop_path)
        core.store.engine = None
        core.store.documents = []
        core.store.add_document(sop_path, category="HR")

        # warm up the JIT-y bits (BM25 build etc.) so the first measured
        # query is not 5x slower than the rest
        for _ in range(2):
            single_agent("warmup"); multi_agent_seq("warmup"); multi_agent_par("warmup")

        agg = {"single": [0, 0.0], "multi_seq": [0, 0.0], "multi_par": [0, 0.0]}

        print(f"\n{'query':<44} {'1A':>4} {'1Ams':>6} "
              f"{'Ms':>4} {'Msms':>6} {'Mp':>4} {'Mpms':>6}")
        print("-" * 84)

        for q, gold, should_abstain in QUERIES:
            s = single_agent(q)
            ms = multi_agent_seq(q)
            mp = multi_agent_par(q)

            def _ok(res):
                if should_abstain:
                    return res["would_abstain"]
                joined = " ".join(c["snippet"] for c in res["chunks"]).lower()
                return gold.lower() in joined if gold else False

            so, mso, mpo = _ok(s), _ok(ms), _ok(mp)
            agg["single"][0]    += so;  agg["single"][1]    += s["elapsed_ms"]
            agg["multi_seq"][0] += mso; agg["multi_seq"][1] += ms["elapsed_ms"]
            agg["multi_par"][0] += mpo; agg["multi_par"][1] += mp["elapsed_ms"]

            mark = lambda b: "✓" if b else "✗"
            print(f"{q[:42]:<44} {mark(so):>4} {s['elapsed_ms']:>5.1f} "
                  f"{mark(mso):>4} {ms['elapsed_ms']:>5.1f} "
                  f"{mark(mpo):>4} {mp['elapsed_ms']:>5.1f}")

        total = len(QUERIES)
        print("-" * 84)
        print(f"{'TOTAL':<44} {agg['single'][0]}/{total:<2} "
              f"{agg['single'][1]/total:>5.1f} "
              f"{agg['multi_seq'][0]}/{total:<2} "
              f"{agg['multi_seq'][1]/total:>5.1f} "
              f"{agg['multi_par'][0]}/{total:<2} "
              f"{agg['multi_par'][1]/total:>5.1f}")
        print()

        s_ms = agg["single"][1]/total
        ms_ms = agg["multi_seq"][1]/total
        mp_ms = agg["multi_par"][1]/total

        print(f"Single-agent      : {agg['single'][0]}/{total} correct, "
              f"avg {s_ms:.1f} ms          (baseline)")
        print(f"Multi-agent seq   : {agg['multi_seq'][0]}/{total} correct, "
              f"avg {ms_ms:.1f} ms          ({ms_ms/s_ms:.1f}x baseline)")
        print(f"Multi-agent par   : {agg['multi_par'][0]}/{total} correct, "
              f"avg {mp_ms:.1f} ms          ({mp_ms/s_ms:.1f}x baseline)")
    finally:
        os.unlink(sop_path)
        _EXECUTOR.shutdown(wait=False)


if __name__ == "__main__":
    main()
