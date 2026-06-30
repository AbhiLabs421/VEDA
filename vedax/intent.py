"""Query Intent Decomposer + Adaptive Retrieval — beats fixed top-k.

The vedax_chatbot session showed a real production bug:

    Q: 'software'                       conf 0.90  -> answers correctly
    Q: 'define software in single word' conf 0.25  -> wrongly abstains

The retrieval actually finds the right chunk at rank 1.  What kills the
answer is the abstention guard, because:
  * 'define' is treated as an unknown corpus term (it is rare)
  * 'single' is everywhere in the corpus (dilutes coverage)
  * the real SUBJECT ('software') gets only 1/4 of the weight

Lesson:  'define', 'in single word', 'in one line', 'ka matlab', 'tell
me about', 'summarise' etc. are query INTENT markers, not topic words.
A user query has a STRUCTURE — they signal HOW they want the answer
('define', 'list', 'briefly') and WHAT they want it about ('software').
Treating every word the same is what makes BM25-style retrieval brittle
on natural-language questions.

This module:
  1. Decomposes a query into (intent, subject, fillers, language)
  2. Provides subject-focused scoring so filler words like 'single' can
     no longer hijack the ranking
  3. Returns an ADAPTIVE candidate set (replaces fixed top-k) chosen
     by score-plateau detection: when one chunk wins clearly, return
     just that; when the answer is spread, return more.
"""

import math
import re
from collections import Counter

from veda.encoder import tokenize


# ── 1. Intent markers ────────────────────────────────────────────────
#
# Every entry is (pattern, intent_name, "answer style") and may emit
# a CAPTURE group that names the subject.

_INTENT_PATTERNS = [
    # explicit definition / explanation
    (re.compile(r"^(?:please\s+)?define\s+(.+?)(?:\s+in\s+(?:single|one|few|short)?\s*"
                r"(?:word|line|sentence)s?)?\s*[?.]?$", re.I), "define"),
    (re.compile(r"^(?:please\s+)?what\s+(?:is|are|does)\s+(?:a\s+|an\s+|the\s+)?"
                r"(.+?)(?:\s+in\s+(?:single|one|few|short)?\s*"
                r"(?:word|line|sentence)s?)?\s*[?.]?$", re.I), "define"),
    (re.compile(r"^(?:full\s+form|meaning|definition)\s+of\s+(.+?)\s*[?.]?$",
                re.I), "define"),
    # Hinglish: 'X ka matlab', 'X ka matlab kya hai', 'X kya hai'
    (re.compile(r"^(.+?)\s+(?:ka\s+matlab|matlab|meaning)"
                r"(?:\s+(?:kya\s+hai|kya\s+hota\s+hai))?\s*[?.]?$",
                re.I), "define"),
    (re.compile(r"^(.+?)\s+(?:kya\s+hai|kya\s+hota\s+hai|"
                r"kya\s+matlab\s+hai)\s*[?.]?$", re.I), "define"),
    # listing / enumeration
    (re.compile(r"^list(?:\s+all)?\s+(?:the\s+)?(.+?)\s*[?.]?$", re.I), "list"),
    (re.compile(r"^(?:what\s+are\s+(?:the\s+)?(?:all\s+)?|"
                r"name\s+(?:the\s+|all\s+)?)(.+?)\s*[?.]?$", re.I), "list"),
    # how-to / procedure
    (re.compile(r"^how\s+(?:do|does|can|to)\s+(?:i\s+|you\s+|one\s+)?(.+?)\s*[?.]?$",
                re.I), "procedure"),
    (re.compile(r"^(?:procedure|steps?|process)\s+(?:for|to|of)\s+(.+?)\s*[?.]?$",
                re.I), "procedure"),
    # quantity / amount
    (re.compile(r"^how\s+(?:many|much|long|often)\s+(.+?)\s*[?.]?$",
                re.I), "quantity"),
    # yes/no
    (re.compile(r"^(?:is|are|can|does|do|will|should|may|must|has|have)\s+"
                r"(.+?)\s*[?.]?$", re.I), "yesno"),
    # general "tell me"
    (re.compile(r"^(?:tell\s+me|explain|describe|summari[sz]e|brief\s+me)\s+"
                r"(?:about\s+)?(.+?)\s*[?.]?$", re.I), "explain"),
]

# Words that are clearly *answer style* requests — they tell us how the
# answer should look, never what it is about.  Stripped before scoring.
_STYLE_FILLERS = frozenset("""
in single one few short brief long detailed quickly briefly please
kindly word words line lines sentence sentences paragraph
""".split())

# Conversational openers / closers / hedges / address terms.  None of
# these ever describes the topic — they are social glue that real users
# sprinkle around their question ('yo what's X bro', 'please tell me
# about X bhai').
_CONVERSATIONAL = frozenset("""
yo hey hi hello hai hii bro bhai bhaiya bhaiyya didi sir madam ma'am
yaar dost dude guys folks please plz pls kindly thanks thank kindly
just simply only actually basically really exactly um uh hmm okay ok
i me my mine you your we us they them he she it this that
know dont don't can't cant could would should will shall may might
tell me explain define describe define brief briefly show give
about regarding concerning anything something nothing somebody anybody
quick quickly fast slow slowly clearly properly thoroughly
help me out here there now then today briefly
what's whats whatis what's wassup wsup sup
""".split())

# Hinglish conversational glue: 'kya hai', 'ka matlab', 'ko samjhao',
# 'bata do', 'samajh nahi aaya', etc.
_HINGLISH_FILLERS = frozenset("""
kya hai hota hai matlab ka ke ki ko se me main mein bata batao
batayiye samjhao samjha samjhe samjhna samajh nahi aaya aaya nahi
chahiye chahta jaanna jaana jaan kar karo karta karte karna
ham hum hamko mujhe mujhko apko aapko apna apni aap tum tumko
aur kuch sab thoda thodi zyada bahut bilkul accha theek matlab
hota hai hain hai bhai bhaiya
""".split())

# Common stopwords that may be safely down-weighted
_LIGHT_STOP = frozenset("""
a an the of to in on at by for and or but with from into about as is are
was were be been being it its this that these those i you he she they
we my your his her their our what which who whom whose when where why
how do does did have has had can could should would may might must
""".split())

_ALL_FILLERS = (_STYLE_FILLERS | _CONVERSATIONAL | _HINGLISH_FILLERS
                | _LIGHT_STOP)

_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
_HYPHEN_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,5}-[A-Z][A-Z0-9]{1,5}\b")
_SPACED_LETTERS_RE = re.compile(r"\b(?:[a-zA-Z]\s+){2,}[a-zA-Z]\b")


def _detect_acronyms(query):
    """Find acronym-like tokens regardless of regex pattern.  Handles:
      * 'xyz', 'NDS-OM' (standard)
      * 'cclil' (typo-tolerant via prefix matching to corpus terms)
      * 'c c i l' (spaced letters)
    """
    acros = list(_HYPHEN_ACRONYM_RE.findall(query))
    acros += list(_ACRONYM_RE.findall(query))
    # spaced letters: "c c i l" -> "cxyz"
    for m in _SPACED_LETTERS_RE.finditer(query):
        joined = re.sub(r"\s+", "", m.group(0)).upper()
        if len(joined) >= 2:
            acros.append(joined)
    return acros


def _strip_fillers(query):
    """Remove conversational / answer-style fillers, leaving only the
    likely topic words."""
    tokens = re.findall(r"\b[A-Za-z0-9'\-]+\b", query)
    kept = []
    for t in tokens:
        low = t.lower()
        if low in _ALL_FILLERS:
            continue
        kept.append(t)
    return " ".join(kept).strip()


def decompose(query):
    """Return ``{intent, subject, raw, fillers, has_intent_marker}``.

    'define software in single word'
        -> intent='define', subject='software',
           fillers={'in','single','word'}, has_intent_marker=True
    'software'
        -> intent='general', subject='software',
           fillers=set(), has_intent_marker=False
    """
    q = query.strip()
    intent = "general"
    subject = q

    # ── STEP 1: Acronym shortcut ─────────────────────────────────────
    # If the query contains an acronym (xyz, NDS-OM, 'c c i l') treat
    # IT as the subject regardless of how the question is phrased.
    # Real users say 'yo what's xyz bro' — the acronym is the only
    # token that matters.
    acros = _detect_acronyms(q)
    matched_span = None

    # ── STEP 2: Intent classification via regex ──────────────────────
    for pat, name in _INTENT_PATTERNS:
        m = pat.match(q)
        if m:
            intent = name
            subject = m.group(1).strip().rstrip(",.?!")
            matched_span = m.group(0)
            break

    # ── STEP 3: Intent inference from filler markers ─────────────────
    # Even if no regex matched, presence of 'define', 'explain',
    # 'tell me', 'meaning' etc. signals a define-intent.
    if intent == "general":
        ql = q.lower()
        if re.search(r"\b(define|meaning|definition|explain|describe|"
                     r"matlab|kya\s+hai|kya\s+hota\s+hai|stand\s+for|"
                     r"full\s+form)\b", ql):
            intent = "define"
        elif re.search(r"\b(list|enumerate|all\s+the|name\s+all)\b", ql):
            intent = "list"
        elif re.search(r"\b(how\s+(to|do|does|can)|procedure|steps?|"
                       r"process)\b", ql):
            intent = "procedure"
        elif re.search(r"\b(tell|brief|about)\b", ql):
            intent = "explain"

    # ── STEP 4: Subject — prefer the acronym, else strip fillers ─────
    if acros:
        # the acronym dominates; keep at most two if multiple
        subject = " ".join(acros[:2])
    elif matched_span is None:
        # no regex matched — strip fillers from the whole query
        stripped = _strip_fillers(q)
        if stripped and len(stripped.split()) <= 6:
            subject = stripped
    else:
        # a regex matched; further strip its captured subject of any
        # leftover fillers ('plz', 'bro', 'briefly', 'kya hota hai')
        stripped = _strip_fillers(subject)
        if stripped:
            subject = stripped

    # ── STEP 5: Filler set — every non-subject token ─────────────────
    sub_tokens = set(t.lower() for t in re.findall(r"\b\w+\b", subject))
    q_tokens = tokenize(q)
    fillers = set()
    for t in q_tokens:
        if t in sub_tokens:
            continue
        if (t in _ALL_FILLERS or intent != "general" or t in _LIGHT_STOP):
            fillers.add(t)
    return {
        "intent": intent,
        "subject": subject,
        "raw": query,
        "fillers": fillers,
        "has_intent_marker": intent != "general",
    }


# ── 2. Subject-focused scoring ───────────────────────────────────────

def weighted_query_terms(parsed, *, subject_weight=3.0, filler_weight=0.0):
    """Term -> weight, used to rescore retrieval candidates so filler
    words like 'single' cannot hijack the ranking."""
    weights = Counter()
    for t in tokenize(parsed["subject"]):
        if len(t) >= 2 and t not in _LIGHT_STOP:
            weights[t] += subject_weight
    for t in tokenize(parsed["raw"]):
        if t in weights:
            continue
        if t in parsed["fillers"] or t in _STYLE_FILLERS:
            continue
        if t in _LIGHT_STOP:
            continue
        # leftover content words from the raw query (e.g. an adjective
        # the user added) get a small extra weight, not zero
        weights[t] += 1.0
    return weights


def rescore(hits, parsed):
    """Re-rank existing hits using subject-focused term weights.  Returns
    a new list of dicts with an added ``adj_score`` field.

    The rescore is monotone in subject overlap, so if the original
    retrieval put the right chunk at rank 1, it stays at rank 1; if
    fillers caused a wrong chunk to outrank the right one, this
    typically fixes it."""
    weights = weighted_query_terms(parsed)
    if not weights:
        for h in hits:
            h["adj_score"] = h.get("score", 0.0)
        return hits
    rescored = []
    for h in hits:
        text = (h.get("snippet") or "").lower()
        toks = tokenize(text)
        tf = Counter(toks)
        score = sum(w * math.log1p(tf.get(t, 0)) for t, w in weights.items())
        h = dict(h)
        h["adj_score"] = score
        rescored.append(h)
    rescored.sort(key=lambda x: x["adj_score"], reverse=True)
    return rescored


# ── 3. Adaptive cutoff: beat fixed top-k ──────────────────────────────

def adaptive_cutoff(scored_hits, *, score_key="adj_score",
                    min_keep=1, max_keep=12, plateau_drop=0.4):
    """Pick the natural number of chunks instead of a fixed top-k.

    Strategy: scan the sorted scores; stop as soon as one of the
    following is true:

      1. score drops by more than ``plateau_drop`` (relative to the
         CURRENT score) — a 'cliff' indicating the rest are noise;
      2. score drops below ``plateau_drop * top_score`` — the rest are
         qualitatively worse than the leader;
      3. ``max_keep`` reached.

    This way:

      * a uniquely-answered query returns ONE chunk (LLM context is
        tight, hallucination space shrinks);
      * a 'tell me about X' style query that has answers spread across
        many chunks returns more;
      * a query whose top-1 is a clear winner is not diluted by mediocre
        chunks below it.
    """
    if not scored_hits:
        return []
    ranked = sorted(scored_hits, key=lambda x: x.get(score_key, 0.0),
                    reverse=True)
    top = ranked[0].get(score_key, 0.0) or 1e-9
    keep = [ranked[0]]
    for i in range(1, min(len(ranked), max_keep)):
        cur = ranked[i].get(score_key, 0.0)
        prev = ranked[i - 1].get(score_key, 0.0) or 1e-9
        # relative cliff between consecutive items
        if cur < prev * (1.0 - plateau_drop):
            break
        # absolute distance from the leader
        if cur < top * plateau_drop:
            break
        keep.append(ranked[i])
    while len(keep) < min_keep and len(keep) < len(ranked):
        keep.append(ranked[len(keep)])
    return keep


# ── 4. End-to-end smart search ───────────────────────────────────────

def _typo_correct(token, corpus_vocab, max_dist=2):
    """Find the closest corpus token within Levenshtein distance
    ``max_dist``.  Used to rescue queries like 'cclil' -> 'xyz'.
    Returns the corrected token, or ``None`` if no close match."""
    if not token or len(token) < 3:
        return None
    low = token.lower()
    candidates = []
    for word in corpus_vocab:
        if abs(len(word) - len(low)) > max_dist:
            continue
        d = _levenshtein(low, word.lower(), max_dist)
        if d <= max_dist:
            candidates.append((d, len(word), word))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _levenshtein(a, b, max_d):
    """Standard Levenshtein with an early-exit when distance exceeds
    ``max_d``."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_d:
        return max_d + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > max_d:
            return max_d + 1
        prev = cur
    return prev[lb]


def smart_search(engine, query, *, max_keep=12):
    """Drop-in replacement for ``engine.search`` that uses the intent
    decomposer + subject-focused rescore + adaptive cutoff.

    Returns ``{hits, parsed, k_selected, dropped}``."""
    parsed = decompose(query)
    # We search on the SUBJECT — fillers do not hit the corpus at all.
    # Whenever an acronym or an intent marker was detected, trust the
    # parsed subject (not the raw query).  For bare queries with no
    # markers and no acronyms, fall back to the raw text.
    has_acro = bool(_detect_acronyms(query))
    if parsed["has_intent_marker"] or has_acro:
        base_query = parsed["subject"]
    else:
        base_query = query
    raw_hits = engine.search(base_query, k=max_keep)

    # Typo rescue: if retrieval came back weak, try correcting subject
    # tokens against the corpus vocabulary ('cclil' -> 'xyz').
    if not raw_hits or max(h.get("score", 0) for h in raw_hits) < 0.3:
        subj_tokens = [t for t in re.findall(r"\b\w+\b", base_query)
                       if len(t) >= 3 and t.lower() not in _ALL_FILLERS]
        if subj_tokens:
            vocab = set(engine.sem.freq) if hasattr(engine, "sem") else set()
            corrections = []
            for t in subj_tokens:
                fix = _typo_correct(t, vocab)
                corrections.append(fix or t)
            corrected = " ".join(corrections)
            if corrected.lower() != base_query.lower():
                rescue = engine.search(corrected, k=max_keep)
                if rescue and (not raw_hits or
                               max(h.get("score", 0) for h in rescue)
                               > max(h.get("score", 0) for h in raw_hits)):
                    raw_hits = rescue
                    parsed["subject"] = corrected
                    parsed["typo_corrected"] = True

    rescored = rescore(raw_hits, parsed)
    kept = adaptive_cutoff(rescored, max_keep=max_keep)
    return {
        "hits": kept,
        "parsed": parsed,
        "k_selected": len(kept),
        "dropped": len(rescored) - len(kept),
    }


def subject_coverage(parsed, hits):
    """Fraction of SUBJECT content terms that appear in the kept hits.
    A robust replacement for the old 'every query word matters equally'
    coverage signal, so 'define software in single word' is judged by
    whether the answer covers 'software', not by 'define' / 'single' /
    'word'."""
    subj = [t for t in tokenize(parsed["subject"])
            if t not in _LIGHT_STOP and len(t) >= 2]
    if not subj:
        return 1.0
    text = " ".join(h.get("snippet", "") for h in hits).lower()
    hit_tokens = set(tokenize(text))
    matched = sum(1 for t in subj if t in hit_tokens
                  or any(t in w for w in hit_tokens))
    return matched / len(subj)
