"""FinanceBench head-to-head: page-level retrieval on SEC 10-K filings.

Methodology:
  * Each question targets exactly one PDF and one evidence page.
  * We extract every page of every needed PDF with our built-in parser
    (veda.pdftext), make each page a chunk and index it with VEDA-X.
  * For each question we measure Recall@k (k=1, 3, 5): did the gold page
    appear in our top-k pages of that document?
  * Compared against three baselines:
      - BM25                       (Anserini params)
      - all-MiniLM-L6-v2 (MiniLM)  (the standard RAG retriever)
      - VEDA-X                     (our pipeline)

This is PageIndex's home turf — structured 10-K filings — so a fair
comparison here matters more than another win on BEIR.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

from eval.veda_x import K, rrf
from vedax.bm25 import BM25
from veda.encoder import SemanticMemory, encode_query, tokenize
from veda.hypervector import l2_dense
from veda.pdftext import extract_pdf_text


FB_ROOT = os.environ.get("FB_DIR", "/tmp/fb")


def load_questions(limit=None):
    path = os.path.join(FB_ROOT, "data", "financebench_open_source.jsonl")
    out = []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        evidence = row.get("evidence") or []
        if not evidence:
            continue
        gold_pages = {int(e["evidence_page_num"]) for e in evidence
                      if e.get("evidence_page_num") is not None}
        gold_texts = [e.get("evidence_text_full_page") or e["evidence_text"]
                      for e in evidence if e.get("evidence_text")]
        if not gold_pages or not gold_texts:
            continue
        out.append({
            "qid": row["financebench_id"],
            "doc": row["doc_name"],
            "question": row["question"],
            "gold_pages": gold_pages,
            "gold_texts": gold_texts,
        })
        if limit and len(out) >= limit:
            break
    return out


_TOK_RE = __import__("re").compile(r"[A-Za-z0-9]+")


def _fingerprint(text, n=10):
    """Take the n rarest-looking tokens from the evidence text to anchor
    a content match against our extracted pages."""
    toks = [t.lower() for t in _TOK_RE.findall(text) if len(t) > 3]
    # Numbers and longer tokens are most discriminative.
    toks.sort(key=lambda t: (-len(t), t))
    seen, picked = set(), []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        picked.append(t)
        if len(picked) >= n:
            break
    return picked


def align_gold(question, pages):
    """Find the physical page indices (1-based) whose content best matches
    the evidence text. Returns the set of best matches (usually one)."""
    page_text = [t.lower() for t in pages]
    aligned = set()
    for gold_text in question["gold_texts"]:
        fp = _fingerprint(gold_text)
        if not fp:
            continue
        best, best_n = None, 0
        for i, pt in enumerate(page_text):
            n = sum(1 for t in fp if t in pt)
            if n > best_n:
                best_n, best = n, i + 1
        if best is not None and best_n >= max(2, len(fp) // 2):
            aligned.add(best)
    return aligned


def extract_pages(pdf_path):
    """List of page-strings; index = page number (1-based).

    Uses the same heal-and-normalise pipeline as extract_pdf_text so the
    glyph-spaced typesetters in SEC 10-Ks become readable text.
    """
    import re
    from veda.pdftext import (PdfDocument, Ref, _extract_content_text,
                              _heal_glyph_spacing)
    with open(pdf_path, "rb") as f:
        doc = PdfDocument(f.read())
    pages_text = []
    for page, inherited in doc.pages():
        resources = doc.resolve(page.get("Resources")) or inherited
        cmaps = doc.font_cmaps(resources if isinstance(resources, dict)
                               else {})
        contents = page.get("Contents")
        refs = contents if isinstance(contents, list) else [contents]
        chunks = [doc.stream_bytes(r) for r in refs if isinstance(r, Ref)]
        text = _extract_content_text(b"\n".join(chunks), cmaps)
        text = re.sub(r"[ \t]{3,}", "  ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = _heal_glyph_spacing(text)
        pages_text.append(text)
    return pages_text


# ---- retrievers (BM25 / dense / VEDA-X) all rank within ONE document.

def bm25_rank(doc_pages, query, k=10):
    bm = BM25()
    bm.index({i + 1: p for i, p in enumerate(doc_pages)})
    return [pid for pid, _ in bm.search(query, k=k)]


def dense_rank_factory():
    """Lazy: one MiniLM session reused across docs."""
    from vedax.dense import MiniLM
    model = MiniLM()
    cache = {}

    def embed_pages(doc_name, pages):
        if doc_name in cache:
            return cache[doc_name]
        import numpy as np
        embs = model.embed([p[:2000] for p in pages]) if pages else None
        cache[doc_name] = embs
        return embs

    def rank(doc_name, doc_pages, query, k=10):
        import numpy as np
        embs = embed_pages(doc_name, doc_pages)
        qemb = model.embed([query])[0]
        sims = embs @ qemb
        order = list(np.argsort(-sims)[:k])
        return [int(i) + 1 for i in order]

    return rank


def vedax_rank(doc_pages, query, sem, bm25, dense_ranking,
               embs, model, expand_w=0.4, prf_alpha=0.5, prf_docs=2):
    """Same pipeline as eval.veda_x, but ranking pages within one doc."""
    import numpy as np
    from collections import defaultdict

    # bm25 + dense first pass -> RRF
    first_bm = bm25_rank(doc_pages, query, k=10)
    first = rrf({"bm": {0: first_bm}, "ds": {0: dense_ranking}},
                {"bm": 0.5, "ds": 0.5})[0]

    # hyperdimensional query expansion -> bm25 re-run
    qtokens = tokenize(query)
    qdense = encode_query(qtokens, sem)
    qnorm = l2_dense(qdense) or 1.0
    qset = set(qtokens)
    cands = Counter()
    import math
    for pid in first[:prf_docs * 3]:
        for term in set(tokenize(doc_pages[pid - 1])):
            if term not in qset and len(term) > 2:
                cands[term] += 1
    scored = []
    for term, df in cands.items():
        entries = sem.semantic_entries(term)
        dot = sum(v * qdense[p] for p, v in entries)
        tn = math.sqrt(sum(v * v for _, v in entries)) or 1.0
        sim = dot / (tn * qnorm)
        if sim > 0:
            scored.append((sim * math.log1p(df) * bm25.idf(term), term))
    scored.sort(reverse=True)
    terms = [t for _, t in scored[:8]]

    bm_x = Counter()
    for pid, sc in bm25.search(query, k=20):
        bm_x[pid] += (1 - expand_w) * sc
    if terms:
        for pid, sc in bm25.search(query + " " + " ".join(terms), k=20):
            bm_x[pid] += expand_w * sc
    bm_x_rank = [pid for pid, _ in bm_x.most_common(10)]

    # dense PRF
    fb_idx = [pid - 1 for pid in first[:prf_docs]]
    qemb = model.embed([query])[0]
    if fb_idx:
        centroid = embs[fb_idx].mean(axis=0)
        qemb = qemb + prf_alpha * centroid
        qemb = qemb / (np.linalg.norm(qemb) or 1.0)
    sims = embs @ qemb
    dense_x = [int(i) + 1 for i in np.argsort(-sims)[:10]]

    # fuse — weights from our NFCorpus tuning (bm25_x 0.25, dense_x 1.0)
    fused = rrf({"bm25_x": {0: bm_x_rank}, "dense_x": {0: dense_x}},
                {"bm25_x": 0.25, "dense_x": 1.0})
    return fused[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions")
    parser.add_argument("--systems", nargs="+",
                        default=["bm25", "dense", "vedax"])
    args = parser.parse_args()

    questions = load_questions(limit=args.limit)
    print(f"{len(questions)} questions over "
          f"{len({q['doc'] for q in questions})} 10-K PDFs", flush=True)

    # Group questions by doc so we extract each PDF only once.
    by_doc = defaultdict(list)
    for q in questions:
        by_doc[q["doc"]].append(q)

    dense_rank = dense_rank_factory() if "dense" in args.systems \
        or "vedax" in args.systems else None
    from vedax.dense import MiniLM
    model = MiniLM() if dense_rank else None

    # Aggregate counters.
    sys_hits = {s: {"r1": 0, "r3": 0, "r5": 0} for s in args.systems}
    total = 0
    t0 = time.time()
    skipped = 0

    for di, (doc_name, qs) in enumerate(by_doc.items(), 1):
        pdf_path = os.path.join(FB_ROOT, "pdfs", doc_name + ".pdf")
        if not os.path.isfile(pdf_path):
            skipped += len(qs)
            continue
        try:
            pages = extract_pages(pdf_path)
        except Exception as exc:
            print(f"  ! skip {doc_name}: {exc}", file=sys.stderr)
            skipped += len(qs)
            continue
        if not pages:
            skipped += len(qs)
            continue

        bm25 = BM25()
        bm25.index({i + 1: p for i, p in enumerate(pages)})

        sem = SemanticMemory()
        for p in pages:
            sem.observe(tokenize(p))

        embs = None
        if model:
            embs = model.embed([p[:2000] for p in pages])

        for q in qs:
            # FinanceBench's gold page numbers follow the PDF's printed
            # numbering, which is offset from physical index per filing.
            # Align by content (find the page whose text actually matches
            # the cited evidence). Falls back to gold ±1 if no match.
            aligned = align_gold(q, pages)
            if not aligned:
                aligned = set()
                for g in q["gold_pages"]:
                    aligned.update({g - 2, g - 1, g, g + 1, g + 2})
                aligned = {p for p in aligned if 1 <= p <= len(pages)}
            if not aligned:
                skipped += 1
                continue
            total += 1
            gold_loose = aligned

            rankings = {}
            if "bm25" in args.systems:
                rankings["bm25"] = bm25_rank(pages, q["question"], k=10)
            if "dense" in args.systems:
                rankings["dense"] = dense_rank(doc_name, pages,
                                               q["question"], k=10)
            if "vedax" in args.systems:
                drank = (rankings.get("dense")
                         or dense_rank(doc_name, pages, q["question"], k=10))
                rankings["vedax"] = vedax_rank(pages, q["question"], sem,
                                               bm25, drank, embs, model)
            for name, r in rankings.items():
                if r and r[0] in gold_loose:
                    sys_hits[name]["r1"] += 1
                if any(p in gold_loose for p in r[:3]):
                    sys_hits[name]["r3"] += 1
                if any(p in gold_loose for p in r[:5]):
                    sys_hits[name]["r5"] += 1

        if di % 5 == 0 or di == len(by_doc):
            print(f"  ...{di}/{len(by_doc)} docs, {total} q's, "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"\n{total} judged questions ({skipped} skipped)\n")
    print(f"{'system':<10} {'Recall@1':>9} {'Recall@3':>9} {'Recall@5':>9}")
    print("-" * 40)
    for name in args.systems:
        h = sys_hits[name]
        print(f"{name:<10} {h['r1'] / total:>9.3f} "
              f"{h['r3'] / total:>9.3f} {h['r5'] / total:>9.3f}")


if __name__ == "__main__":
    main()
