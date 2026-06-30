"""Grounding guards: abstention + citation verification.

These two pieces fight the most common failure modes of any RAG stack:

  * abstention   — refuse to answer when retrieval is weak. Today: if a
                   user asks something not in the corpus, the retriever
                   still returns the top-k chunks and the LLM willingly
                   makes up an answer.
  * verification — after the LLM has answered with inline ``[1]``,
                   ``[2]`` citations, check that each cited chunk
                   actually supports the sentence that cites it. A
                   "supported" answer is grounded; an unsupported claim
                   is a hallucination, even when cited.

Both are training-free, run on CPU, and use only the standard library.
"""

import math
import re
from collections import Counter

from veda.encoder import SemanticMemory, encode_query, light_stem, tokenize
from veda.hypervector import l2_dense


def _stem_set(tokens):
    """A token AND its light-stem live in the same equivalence class —
    'delivery' and 'delivered' should match."""
    out = set()
    for t in tokens:
        out.add(t)
        stem = light_stem(t)
        if stem:
            out.add(stem)
    return out


def _in_corpus(token, sem):
    """True if ``token`` or any morphological cousin has been observed.

    Plain match, light-stem match, and a prefix bridge that catches
    'delivery' <-> 'delivered' / 'penalties' <-> 'penalty'."""
    if sem.freq.get(token, 0):
        return True
    stem = light_stem(token)
    if stem and sem.freq.get(stem, 0):
        return True
    # Adaptive prefix match: keep all but the last ~3 characters, never
    # less than 5. 'delivery' bridges to 'delivered', 'microsoft' does
    # not bridge to 'microcode'.
    need = max(5, len(token) - 3)
    if len(token) < need:
        return False
    prefix = token[:need]
    return any(w.startswith(prefix) for w in sem.freq)


CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Question words and structural English stopwords that frequently appear
# in user queries but carry no topical signal. Excluded from the
# IDF-weighted coverage check so their absence from the corpus does not
# drown the actual content terms.
_QUERY_STOPWORDS = frozenset((
    "what", "which", "whose", "where", "when", "who", "whom", "why",
    "how", "the", "and", "for", "but", "are", "was", "were", "have",
    "has", "had", "you", "your", "yours", "this", "that", "these",
    "those", "with", "from", "about", "into", "over", "under", "between",
    "such", "than", "then", "they", "them", "their", "there", "any",
    "all", "some", "more", "most", "many", "much", "very", "just",
    "also", "can", "could", "should", "would", "shall", "will", "may",
    "might", "must", "does", "did", "doing", "done", "tell", "give",
    "show", "explain", "describe", "list", "outline", "summarise",
    "summarize", "summary", "details", "detail", "please", "kindly",
    # answer-style markers — these tell us HOW to answer, not WHAT
    # ('define X in single word' / 'X in one line' / 'briefly')
    "define", "single", "word", "line", "sentence", "briefly", "brief",
    "short", "shortly", "long", "longer", "longest", "in", "one",
    "few", "couple", "summarize", "meaning", "matlab",
))

# Common attribution preambles that the LLM sometimes emits as standalone
# lines. These are meta-statements, not claims — they declare WHERE the
# information came from, but make no factual assertion to verify.
_META_PREAMBLES = (
    "source", "sources", "reference", "references", "citations",
    "see also", "cited", "ref",
)


def _is_meta_citation_only(sentence):
    """A sentence that is just citation bookkeeping (e.g. 'Sources: [1],
    [5]', 'See also: [3]') is not a factual claim and should be skipped
    by the grounding check."""
    stripped = CITATION_RE.sub("", sentence).strip()
    cleaned = re.sub(r"[^A-Za-z]+", " ", stripped).strip().lower()
    if not cleaned:
        return True
    return cleaned in _META_PREAMBLES


# ----------------------------------------------------------- abstention

def retrieval_confidence(query, hits, sem):
    """Score 0..1 estimating whether the top chunks actually answer the
    query. Returned as ``(confidence, reasons)``.

    Heuristics combine three weak signals into a robust composite:
      * top score absolute magnitude       (sharpness vs noise floor)
      * query-token coverage in top chunks (lexical grounding)
      * score gap between rank 1 and 3     (peak vs flat)
    """
    if not hits:
        return 0.0, ["no_results"]

    reasons = []
    qtokens = [t for t in tokenize(query)
               if len(t) > 2 and t not in _QUERY_STOPWORDS]
    qset = set(qtokens)

    # 1. IDF-weighted, stem-aware coverage of query tokens in the top 3
    # chunks. A specific entity that appears (even via its stem)
    # nowhere in the corpus is the strongest signal that the answer is
    # absent. Generic words carry low weight by construction.
    if qset:
        top_words = set(tokenize(" ".join(h["snippet"] for h in hits[:3])))
        top_stems = _stem_set(top_words)
        covered_weight = 0.0
        total_weight = 0.0
        for t in qset:
            df = sem.freq.get(t, 0)
            idf = math.log(1 + 1.0 / (1 + df))
            total_weight += idf
            if t in top_words or (light_stem(t) or t) in top_stems:
                covered_weight += idf
                continue
            # Adaptive morphological bridge: only fuzz the last ~3 chars
            # (suffix variants), never collapse half the word away.
            need = max(5, len(t) - 3)
            if len(t) >= need:
                prefix = t[:need]
                if any(w.startswith(prefix) for w in top_words):
                    covered_weight += idf
        coverage = covered_weight / total_weight if total_weight else 0.5
        # Hard signal: a non-trivial query term that does not exist in
        # the corpus at all (after stemming) means the answer cannot be
        # there. Common short words ("was", "the") are ignored.
        unknown_specific = [
            t for t in qset
            if len(t) >= 5 and not _in_corpus(t, sem)
        ]
        if unknown_specific:
            reasons.append(f"unknown_terms({','.join(unknown_specific[:3])})")
            coverage *= 0.2
    else:
        coverage = 0.5  # query has no content words, neutral

    # 2. semantic alignment of best hit vs the encoded query
    qdense = encode_query(qtokens, sem)
    qnorm = l2_dense(qdense) or 1.0
    sims = []
    for h in hits[:3]:
        chunk_tokens = [t for t in tokenize(h["snippet"]) if len(t) > 2]
        if not chunk_tokens:
            continue
        cdense = encode_query(chunk_tokens, sem)
        cnorm = l2_dense(cdense) or 1.0
        dot = sum(a * b for a, b in zip(qdense, cdense))
        sims.append(dot / (qnorm * cnorm))
    top_sim = max(sims) if sims else 0.0

    # 3. gap between top hit and 3rd: large gap = peaked = confident
    if "score" in (hits[0] if hits else {}):
        scores = [h.get("score", 0.0) for h in hits[:5]]
        gap = scores[0] - (scores[2] if len(scores) >= 3 else 0)
    else:
        gap = 0.0

    if coverage < 0.25:
        reasons.append(f"low_query_term_coverage({coverage:.2f})")
    if top_sim < 0.10:
        reasons.append(f"weak_semantic_alignment({top_sim:.2f})")

    # Composite confidence. Weights tuned heuristically; coverage gates
    # the rest because a query whose words appear nowhere in the chunks
    # is almost always unanswerable from those chunks.
    composite = (
        0.5 * coverage
        + 0.4 * max(0.0, min(1.0, top_sim * 3))
        + 0.1 * max(0.0, min(1.0, gap * 5))
    )
    return composite, reasons


def should_abstain(query, hits, sem, threshold=0.3):
    """Bool helper: True if confidence is below ``threshold``."""
    conf, _ = retrieval_confidence(query, hits, sem)
    return conf < threshold


# ------------------------------------------------------- verification

def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _lexical_overlap(claim, chunk):
    """Length-weighted bag-of-content-words overlap. Numbers and longer
    tokens are the most discriminative signal that a specific
    quantitative claim is actually backed."""
    claim_tokens = Counter(t for t in tokenize(claim) if len(t) > 2)
    chunk_tokens = set(t for t in tokenize(chunk) if len(t) > 2)
    if not claim_tokens:
        return 0.0
    score = 0.0
    total = 0.0
    for token, count in claim_tokens.items():
        weight = math.log1p(len(token)) * count
        total += weight
        if token in chunk_tokens:
            score += weight
    return score / total if total else 0.0


def _semantic_overlap(claim, chunk):
    """Hypervector cosine between claim and chunk content vectors.
    Catches paraphrases the lexical check misses ("capital expenditure"
    vs "purchases of property plant and equipment")."""
    sem = SemanticMemory()
    sem.observe(tokenize(chunk))
    sem.observe(tokenize(claim))
    cdense = encode_query([t for t in tokenize(claim) if len(t) > 2], sem)
    kdense = encode_query([t for t in tokenize(chunk) if len(t) > 2], sem)
    cn = l2_dense(cdense) or 1.0
    kn = l2_dense(kdense) or 1.0
    return sum(a * b for a, b in zip(cdense, kdense)) / (cn * kn)


def _overlap_score(claim, chunk):
    """Composite grounding score: lexical and semantic must both think
    the claim is at least loosely connected to the chunk."""
    lex = _lexical_overlap(claim, chunk)
    sem = _semantic_overlap(claim, chunk)
    return 0.6 * lex + 0.4 * max(0.0, sem)


def _parse_citation(token):
    """'[1, 3]' or '[1]' -> [1, 3] or [1]."""
    return [int(n) for n in re.split(r"\s*,\s*", token)]


def verify_citations(answer, hits, support_threshold=0.25):
    """Sentence-level grounding check.

    Returns a list of dicts, one per sentence that carries a citation:

        {
            "sentence":  "Revenue was $5,363M [1].",
            "citations": [1],
            "support":   0.72,
            "supported": True,
        }

    plus a roll-up score in ``[0, 1]`` averaged over cited sentences."""
    sentences = split_sentences(answer)
    results = []
    supported_count = 0
    cited_count = 0
    for sentence in sentences:
        cite_tokens = CITATION_RE.findall(sentence)
        if not cite_tokens:
            continue
        if _is_meta_citation_only(sentence):
            # 'Sources: [1], [5]' carries no claim — skip both counter
            # and verdict, otherwise it falsely flags as UNGROUNDED.
            continue
        cited_count += 1
        nums = []
        for tok in cite_tokens:
            nums.extend(_parse_citation(tok))
        # Best supporting evidence among the cited chunks.
        best = 0.0
        for n in nums:
            if 1 <= n <= len(hits):
                best = max(best, _overlap_score(sentence, hits[n - 1]["snippet"]))
        supported = best >= support_threshold
        if supported:
            supported_count += 1
        results.append({
            "sentence": sentence,
            "citations": nums,
            "support": round(best, 3),
            "supported": supported,
        })
    roll_up = supported_count / cited_count if cited_count else 1.0
    return results, roll_up
