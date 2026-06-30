# Grounding guards: abstention and citation verification

The two failure modes that cause hallucinations in any RAG stack:

1. The user asks something that is **not in the documents**, the
   retriever still returns its top-k chunks, and the LLM willingly
   makes up an answer over them.
2. The LLM cites `[1]` after a claim, but the cited chunk **does not
   actually support** that claim — a confident-looking fabrication.

`vedax/grounding.py` adds two training-free, stdlib-only guards against
both, layered onto `VedaX.chat()`.

## Guard 1 — abstention before generation

`retrieval_confidence(query, hits, sem) -> (score, reasons)` returns a
composite confidence in `[0, 1]` made of three signals:

- **IDF-weighted, stem-aware query-term coverage** in the top three
  chunks. Generic words ("revenue", "what") carry low weight; rare
  query terms ("Microsoft", "FY2018") carry high weight. Specific
  query tokens that do not exist in the corpus at all — even via a
  light-stem prefix bridge — drop the score sharply with an explicit
  reason such as `unknown_terms(microsoft)`.
- **Hypervector cosine alignment** between the encoded query and the
  encoded top chunks.
- **Score gap** between the first and the third hit: a peaked
  distribution is more trustworthy than a flat one.

If the composite is below `abstain_threshold` (default `0.3`),
`VedaX.chat()` emits an `("abstain", {...})` event and **does not call
the LLM at all**. The user sees a clear refusal instead of a
fabricated answer.

### Example

```text
$ python veda.py "what was Microsoft revenue in 2023" (corpus is contracts)
[abstain] confidence=0.22 (unknown_terms(microsoft), low_query_term_coverage(0.08))
The retrieved chunks do not appear to answer this question; refusing
to answer rather than risk a fabrication.
```

```text
$ python veda.py "what is the penalty for late delivery" (same corpus)
[retrieved 6 chunks ranked by VEDA-X]
[answer]
The penalty is two percent of contract value per week of delay,
capped at ten percent overall [1].
[citation check: OK grounded=100%]
```

## Guard 2 — citation verification after generation

After the LLM's response has streamed in full, every sentence carrying
an inline `[k]` citation is checked against the chunk it cites:

- **Lexical overlap**: IDF-leaning bag-of-words score between the
  claim sentence and the cited chunk.
- **Semantic overlap**: hypervector cosine similarity in the same
  corpus-local random-indexing space the retriever uses.

A composite of the two is compared against `support_threshold`
(default `0.25`). Each sentence is tagged `supported: True/False` and
the rolled-up `grounded_fraction` is reported.

```python
from vedax.grounding import verify_citations

answer = ("Revenue grew eighteen percent [1]. The company invented "
          "the wheel [2].")
results, grounded = verify_citations(answer, hits)
# results[0]['supported'] = True   (claim matches chunk 1)
# results[1]['supported'] = False  (chunk 2 says nothing about wheels)
# grounded                = 0.5
```

The CLI prints a verdict line `[citation check: OK / WARN / UNGROUNDED]`
plus the offending sentences when the score is low, so the user can
see which specific claim should not be trusted.

## Why these two and not more

These two guards eliminate the two most damaging failure modes — *answer
to a question that has no answer in the corpus* and *cite a chunk that
does not support the cited claim* — without changing the retrieval
pipeline and without adding any dependency. Stronger guards exist
(cross-encoder rerankers, NLI fact-checkers, prompt-injection filters,
table-aware chunking, conversation memory, query decomposition); see
[roadmap.md](./roadmap.md). They are useful improvements but address
narrower or less common failure modes than the two above.

## Configuration

`VedaX.chat()` accepts:

- `abstain_threshold` — default `0.3`. Raise it for stricter refusal,
  lower it to tolerate weaker matches.
- `verify=True` — set to `False` to skip post-generation citation
  checking (faster but unmonitored).
