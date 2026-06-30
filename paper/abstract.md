# Abstract — VEDA-X

Retrieval-augmented generation (RAG) in regulated industries such as
financial clearing must satisfy three constraints that mainstream
designs handle poorly: (i) zero external dependency for auditability,
(ii) defence against prompt-injection and jailbreak attacks
including paraphrased variants, and (iii) guaranteed atomic
retrieval of compliance-critical procedures so that no operator
ever sees a partial step list.

We present **VEDA-X**, a production stack deployed at an Indian
financial-market clearing infrastructure that addresses all three.
Retrieval uses BM25 with hyperdimensional query expansion and
adaptive cutoff — no external embedding model is required.  The
guardrail is a five-layer defence-in-depth pipeline; its semantic
layer detects paraphrased attacks via deterministic `blake2b`
hash-hypervector centroids without any neural model.  A novel
*atomic critical-block* chunking scheme guarantees that
compliance-critical standard operating procedures are emitted as
single indivisible chunks by construction.

On an adversarial test suite drawn from public jailbreak research
we report 18/18 direct-attack, 12/12 sneaky-obfuscated, and 14/14
semantic paraphrase blocks with 0% false positives on a 15-query
legitimate set.  Critical-block retrieval achieves 91% recall@3 on
an 11-query operational benchmark and a 100% completeness rate
when retrieved.  The entire stack is implemented in pure Python
standard library plus FastAPI; total code is approximately 8,000
lines.

**Keywords:** retrieval-augmented generation, hyperdimensional
computing, prompt injection, jailbreak defence, compliance,
financial systems, auditability.
