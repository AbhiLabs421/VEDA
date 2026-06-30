"""VEDA-X engine: the benchmark-winning pipeline over your own files.

Chunks every document, then per query:
  1. BM25 + dense (MiniLM) first pass over chunks
  2. hyperdimensional query expansion (term selection by cosine between a
     term's corpus-local hypervector context and the encoded query) ->
     BM25 re-run with the expanded query
  3. dense pseudo-relevance feedback re-run
  4. weighted reciprocal-rank fusion (weights as tuned on NFCorpus)

Dense stage is optional (``use_dense=False`` or onnxruntime missing) —
the lexical+hyperdimensional stack alone already beats plain BM25.
"""

import math
import os
import pickle
from collections import Counter, defaultdict

from veda.encoder import SemanticMemory, encode_query, tokenize, tokenize_spans
from veda.hypervector import l2_dense
from veda.critical_blocks import (
    parse_critical_spans,
    expand_to_critical,
    strip_markers,
)

from .bm25 import BM25
from .extract import iter_documents

# Fusion weights as tuned on the NFCorpus validation split.
FUSION = {"bm25_x": 0.25, "dense_x": 1.0}
RRF_K = 60


class VedaX:
    def __init__(self, use_dense=True, chunk_tokens=120, overlap_tokens=20):
        self.use_dense = use_dense
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        # 3-tuple per chunk: (doc_id, chunk_text, meta_dict).
        # ``meta_dict`` carries flags such as ``is_critical``,
        # ``critical_title`` and ``whole_file`` so retrievers can
        # surface compliance-grade context to the UI/LLM.
        self.chunks = []
        self.bm25 = BM25()
        self.sem = SemanticMemory()
        self.embs = None        # numpy array, set by _finalize
        self._model = None
        self._ready = False
        # Paths whose ENTIRE content is critical (whole-file atomic
        # chunk).  Populated by ``mark_critical_path`` before ``add``.
        self._critical_paths = set()

    # ------------------------------------------------------------ ingest

    def mark_critical_path(self, path):
        """Mark a path so that everything indexed from it becomes a
        single atomic chunk (compliance folder convention)."""
        self._critical_paths.add(path)
        return self

    def add(self, *paths):
        """Index files and/or directories (.txt/.md/.pdf/...)."""
        for doc_id, text in iter_documents(paths):
            self.sem.observe(tokenize(text))
            crit_spans = parse_critical_spans(text)
            whole_file_critical = doc_id in self._critical_paths

            if whole_file_critical:
                # Folder convention: whole file is one indivisible
                # chunk, regardless of token count.  Markers (if any)
                # are stripped from the visible chunk text.
                self.chunks.append((doc_id, strip_markers(text), {
                    "is_critical": True,
                    "critical_title": os.path.basename(doc_id),
                    "whole_file": True,
                }))
                continue

            tokens_spans = tokenize_spans(text)
            step = max(1, self.chunk_tokens - self.overlap_tokens)
            i, n = 0, len(tokens_spans)
            while i < n:
                j = min(n, i + self.chunk_tokens)
                start_char = tokens_spans[i][1]
                end_char = tokens_spans[j - 1][2]
                # Critical-span expansion: if this tentative chunk
                # overlaps any [[CRITICAL: ...]] span, swallow the
                # whole span so the block is never split.
                new_start, new_end, crit_title = expand_to_critical(
                    start_char, end_char, crit_spans)
                chunk_text = strip_markers(text[new_start:new_end])
                if chunk_text:
                    self.chunks.append((doc_id, chunk_text, {
                        "is_critical": crit_title is not None,
                        "critical_title": crit_title,
                        "whole_file": False,
                    }))
                if j >= n and not crit_title:
                    break
                if crit_title:
                    # Resume after the critical block; no overlap into
                    # it (the block was emitted whole).
                    while i < n and tokens_spans[i][1] < new_end:
                        i += 1
                    if i >= n:
                        break
                else:
                    i += step
        self._ready = False
        return self

    def _finalize(self):
        if self._ready:
            return
        self.bm25 = BM25()
        self.bm25.index({i: text for i, (_, text, _m) in enumerate(self.chunks)})
        if self.use_dense:
            try:
                self.embs = self._embed_chunks()
            except Exception as exc:
                import sys
                print(f"  ! dense stage unavailable ({exc}); "
                      f"running lexical-only", file=sys.stderr)
                self.use_dense = False
        self._ready = True

    def _embed_chunks(self):
        from .dense import MiniLM
        if self._model is None:
            self._model = MiniLM()
        return self._model.embed([text for _, text, _m in self.chunks])

    # ------------------------------------------------------------ search

    def search(self, query, k=5):
        """VEDA-X ranking: list of {file, score, snippet}."""
        self._finalize()
        runs = {"bm25_x": self._bm25_expanded(query)}
        if self.use_dense:
            runs["dense_x"] = self._dense_prf(query)
        fused = self._rrf(runs, FUSION)
        return self._format(fused, k)

    def smart_search(self, query, max_keep=12):
        """Intent-decomposed search that beats fixed top-k.

        Returns ``{hits, parsed, k_selected, dropped}`` where:
          * the query is decomposed into (intent, subject, fillers)
            so 'define X in single word' searches for X (not 'define'
            / 'single' / 'word');
          * results are rescored to give the SUBJECT the dominant
            weight;
          * the number of returned hits is chosen by score-plateau
            detection — a uniquely answered query gets ONE chunk, a
            distributed answer gets more.
        """
        from .intent import smart_search as _smart
        return _smart(self, query, max_keep=max_keep)

    def search_plain_rag(self, query, k=5):
        """Baseline for comparison: plain dense retrieval, nothing else —
        what a standard RAG stack does."""
        self._finalize()
        if not self.use_dense:
            raise RuntimeError("plain-RAG baseline needs the dense stage")
        return self._format(self._dense(query), k)

    def compare(self, query, k=5):
        return {"plain_rag": self.search_plain_rag(query, k),
                "veda_x": self.search(query, k)}

    def chat(self, query, llm_settings, k=6, system=None,
             abstain_threshold=0.3, verify=True, smart=True):
        """Retrieve with VEDA-X, then stream a grounded answer.

        Three grounding guards are layered on top:

          1. retrieval confidence — if the top chunks do not look like
             they contain the answer, the engine abstains BEFORE calling
             the LLM, preventing fabricated answers from being streamed.
          2. constrained system prompt — the model is forced to use only
             the provided context and to cite ``[1]``, ``[2]`` inline.
          3. citation verification — after generation, every cited
             sentence is checked for content overlap with its chunk.

        With ``smart=True`` (default) the query is first decomposed into
        (intent, subject, fillers) and an ADAPTIVE candidate set is sent
        to the LLM instead of a fixed top-``k`` — so 'define X in single
        word' searches for X (not 'define' / 'single' / 'word') and the
        LLM receives only as many chunks as are clearly relevant.

        Yields ``(kind, payload)`` events:

          ("hits",         [...])         the retrieved chunks
          ("parsed",       {...})         the intent / subject decomposition
          ("abstain",      {...})         emitted instead of "token" /
                                          "verification" when guard 1 fires
          ("token",        str)           one streamed token of the answer
          ("verification", {...})         per-sentence citation check
        """
        from .llm import stream_chat
        from .grounding import (retrieval_confidence, verify_citations)
        from .intent import subject_coverage

        parsed = None
        if smart:
            res = self.smart_search(query, max_keep=k)
            hits = res["hits"]
            parsed = res["parsed"]
            yield "parsed", parsed
        else:
            hits = self.search(query, k=k)
        yield "hits", hits

        confidence, reasons = retrieval_confidence(query, hits, self.sem)
        if smart and parsed:
            # Treat subject coverage as an alternative ('strong') signal
            # of grounding — if the subject is clearly present, do not
            # let filler-driven low coverage trigger an abstention.
            cov = subject_coverage(parsed, hits)
            if cov >= 0.5 and hits:
                confidence = max(confidence, 0.4 + 0.5 * cov)
                if reasons:
                    reasons = [r for r in reasons
                               if "unknown_terms" not in r
                               and "low_query_term_coverage" not in r]
        if confidence < abstain_threshold:
            yield "abstain", {
                "confidence": round(confidence, 3),
                "reasons": reasons,
                "message": ("The retrieved chunks do not appear to "
                            "answer this question; refusing to answer "
                            "rather than risk a fabrication."),
            }
            return

        context = "\n\n".join(
            f"[{i + 1}] {h['file']}\n{h['snippet']}"
            for i, h in enumerate(hits)
        )
        system_msg = system or (
            "You answer questions STRICTLY from the provided context. "
            "Cite sources as [1], [2] inline at the end of every claim. "
            "If a question cannot be answered from the context, reply "
            "exactly: 'Not in the provided documents.' "
            "Do not invent facts. Be concise."
        )
        user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        answer_parts = []
        for chunk in stream_chat(messages=messages, **llm_settings):
            answer_parts.append(chunk)
            yield "token", chunk

        if verify:
            answer = "".join(answer_parts)
            results, roll_up = verify_citations(answer, hits)
            yield "verification", {
                "sentences": results,
                "grounded_fraction": round(roll_up, 3),
            }

    # ------------------------------------------------------- the stages

    def _first_pass(self, query, depth=50):
        runs = {"bm25": [c for c, _ in self.bm25.search(query, k=depth)]}
        if self.use_dense:
            runs["dense"] = self._dense(query, depth)
        return self._rrf(runs, {"bm25": 0.5, "dense": 0.5})

    def _expansion_terms(self, query, feedback_ids, n_terms=10, fb_docs=10):
        qtokens = tokenize(query)
        qdense = encode_query(qtokens, self.sem)
        qnorm = l2_dense(qdense) or 1.0
        qset = set(qtokens)
        candidates = Counter()
        for cid in feedback_ids[:fb_docs]:
            for term in set(tokenize(self.chunks[cid][1])):
                if term not in qset and len(term) > 2:
                    candidates[term] += 1
        scored = []
        for term, df_fb in candidates.items():
            entries = self.sem.semantic_entries(term)
            dot = sum(val * qdense[pos] for pos, val in entries)
            tnorm = math.sqrt(sum(v * v for _, v in entries)) or 1.0
            sim = dot / (tnorm * qnorm)
            if sim > 0:
                scored.append((sim * math.log1p(df_fb) * self.bm25.idf(term),
                               term))
        scored.sort(reverse=True)
        return [term for _, term in scored[:n_terms]]

    def _bm25_expanded(self, query, depth=200, expand_weight=0.4):
        feedback = self._first_pass(query)
        terms = self._expansion_terms(query, feedback)
        merged = Counter()
        for cid, score in self.bm25.search(query, k=depth):
            merged[cid] += (1 - expand_weight) * score
        if terms:
            expanded = query + " " + " ".join(terms)
            for cid, score in self.bm25.search(expanded, k=depth):
                merged[cid] += expand_weight * score
        return [cid for cid, _ in merged.most_common(depth)]

    def _dense(self, query, depth=200):
        import numpy as np
        if self._model is None:
            from .dense import MiniLM
            self._model = MiniLM()
        qemb = self._model.embed([query])[0]
        sims = self.embs @ qemb
        return list(np.argsort(-sims)[:depth])

    def _dense_prf(self, query, depth=200, alpha=0.5, prf_docs=3):
        import numpy as np
        first = self._first_pass(query)[:prf_docs]
        qemb = self._model.embed([query])[0]
        if first:
            centroid = self.embs[first].mean(axis=0)
            qemb = qemb + alpha * centroid
            qemb = qemb / (np.linalg.norm(qemb) or 1.0)
        sims = self.embs @ qemb
        return list(np.argsort(-sims)[:depth])

    @staticmethod
    def _rrf(runs, weights):
        votes = defaultdict(float)
        for name, ranking in runs.items():
            w = weights.get(name, 0.0)
            for rank, cid in enumerate(ranking):
                votes[cid] += w / (RRF_K + rank)
        return [cid for cid, _ in
                sorted(votes.items(), key=lambda kv: kv[1], reverse=True)]

    def _format(self, ranking, k):
        results = []
        for cid in ranking[:k]:
            doc_id, text, meta = self.chunks[cid]
            results.append({
                "file": doc_id,
                "snippet": " ".join(text.split()),
                "is_critical": bool(meta.get("is_critical")),
                "critical_title": meta.get("critical_title"),
            })
        return results

    # ----------------------------------------------------------- persist

    def save(self, path):
        self._finalize()
        model, self._model = self._model, None  # session object: don't pickle
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=4)
        finally:
            self._model = model

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            engine = pickle.load(f)
        if not isinstance(engine, cls):
            raise TypeError(f"not a VedaX index file: {path!r}")
        return engine
