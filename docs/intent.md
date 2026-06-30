# Query Intent + Adaptive Retrieval (beats fixed top-k)

`vedax/intent.py`. A real production session showed this concrete bug:

```
Q: software                            conf 0.900  ✓ answers
Q: define software in single word      conf 0.386  ✗ wrongly abstains
```

The right chunk was retrieved at rank 1. The abstention guard killed it
because `define`, `single`, `word` were treated as ordinary query
terms — and `single` happened to be everywhere else in the corpus.

That is the deeper issue: every user query has a **structure**.

- *Subject*: what it is about (e.g. `software`).
- *Intent*: how the answer should look (define / list / how-to / yes-no).
- *Fillers*: answer-style markers (`in single word`, `briefly`, `ka matlab`).

If we ignore that structure, fillers drown the subject. So the smart
search layer does three things, all in one pure-Python module:

## 1. Decompose

```python
from vedax.intent import decompose

decompose("define software in single word")
# -> {intent: 'define', subject: 'software',
#     fillers: {'define', 'in', 'single', 'word'}, ...}

decompose("software ka matlab kya hai")          # Hinglish
# -> {intent: 'define', subject: 'software', ...}

decompose("list the eligibility criteria")
# -> {intent: 'list', subject: 'eligibility criteria', ...}

decompose("how do I prepare a release note")
# -> {intent: 'procedure', subject: 'prepare a release note', ...}
```

English **and** Hinglish patterns are recognised: `kya hai`, `ka matlab`,
`kya matlab hai`, `ka matlab kya hai`.

## 2. Subject-focused rescore

The subject gets ~3× the weight of any leftover content word. Fillers
get weight 0. A noisy chunk full of `single` cannot outrank the chunk
that defines `software`, no matter how many `single`s it contains.

## 3. Adaptive cutoff — better than fixed top-k

Instead of always returning a fixed `k`, the cutoff is chosen by
**score-plateau detection**:

- If one chunk wins clearly → return **just that one** (tighter LLM
  context = smaller hallucination surface).
- If the answer is genuinely spread across several chunks → return more.
- If everything drops off a cliff → return only what is above the
  plateau.

```python
adaptive_cutoff([{adj_score: 10}, {adj_score: 1}, {adj_score: 0.5}])
#   -> 1 hit  (clear winner)

adaptive_cutoff([{adj_score: 5}, {adj_score: 4.9}, {adj_score: 4.6}, {adj_score: 0.5}])
#   -> 3 hits  (plateau of 3, then a cliff)
```

This is the part that beats the rigid top-k convention. A `top-k=6`
chatbot always shoves six chunks at the LLM, even when one would do.
That dilutes the prompt and is the real reason LLMs sometimes pick the
wrong chunk to cite.

## End-to-end usage

```python
from vedax import VedaX

engine = VedaX(use_dense=False).add("docs/")

# drop-in replacement for engine.search:
res = engine.smart_search("define software in single word")
# res = {hits: [...], parsed: {...}, k_selected: 1, dropped: 5}
```

`VedaX.chat(..., smart=True)` (the default) uses this pipeline
automatically, so `vedax_chatbot.py` benefits with no code change.

## Why this is not just "another reranker"

A reranker re-orders the SAME candidate set with the SAME query. Smart
search restructures the QUERY — fillers cannot influence retrieval at
all — and replaces fixed top-k with score-aware adaptive selection.
It is honest about what part of the query mattered, and shows it
explicitly via the `parsed` field, so any debugging session can see
exactly what the engine thought you were asking about.

## Honest scope

- Patterns currently cover English + common Hinglish phrasings. A new
  language family needs a few more regexes.
- For genuinely off-topic queries (`price of bitcoin` on an HR policy)
  the subject is captured, retrieval still finds the lexically closest
  chunk, and `subject_coverage` (0.0 in that case) flags it for
  abstention. The LLM is never called.

Tests: `python -m unittest tests.test_intent` (19 cases).
