# Cover letter

To the Program Committee,

We are submitting *VEDA-X: A Vectorless Retrieval Stack with
Hyperdimensional Guardrails and Atomic Compliance Chunking for
Regulated Financial Knowledge Management* for consideration at
your venue.  This is an applied-systems paper rather than a
theoretical contribution; the work is in production at a major
Indian financial-clearing institution and the entire stack is
implemented in the Python standard library (no PyTorch, no
sentence-transformers, no external embedding API at any stage).

The paper makes three contributions that we believe are individually
modest but collectively novel for the regulated-RAG setting:

1. A **vectorless retrieval pipeline** — BM25 combined with
   hyperdimensional query expansion and adaptive top-k cutoff —
   that achieves the determinism, auditability, and
   zero-dependency profile that internal audit and external
   regulators require.

2. A **five-layer defence-in-depth guardrail** whose semantic
   component (L1.5) blocks paraphrased attacks via deterministic
   `blake2b` hash-hypervector centroid matching with a contrastive
   legitimate-seed anchor.  We report 100% catch rates against
   three attack families and 0% false positives on a 15-query
   legitimate set drawn from production query logs.

3. An **atomic critical-block chunking** scheme that guarantees
   compliance-critical standard operating procedures cannot be
   split across retrievals.  We propose two complementary author
   interfaces — inline markers and a folder convention — and
   measure 100% block-completeness when retrieved.

All numbers in the paper are reproducible from the open-sourced
repository; the relevant scripts complete in under five seconds.
The repository ships the LaTeX source, six SVG figures, the
benchmark scripts, and 165 unit tests.

We believe the work fits the venue because:

- it directly addresses the OWASP LLM Top-10 attack classes;
- it provides a deployment-tested defence-in-depth pattern;
- it identifies a concrete failure mode (partial retrieval of
  critical SOPs) that the literature has not addressed and offers
  a constructive remedy.

The paper is single-authored.  No part of it has been previously
published or is under simultaneous consideration elsewhere.

Sincerely,
Abhishek Kumar
Tata Consultancy Services, Mumbai
ipsabhi423@gmail.com

Corresponding author:
Prof. Atul Kumar Pandey
Birla Institute of Technology, Mesra (Patna Campus)
atulkrpandey@gmail.com
