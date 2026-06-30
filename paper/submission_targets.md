# Submission targets — shortlist with fit notes

Pick by reviewer fit and your timeline.  Workshops have the highest
acceptance rate; journals have the longest review time but the most
weight on a CV.

## Workshops / industry tracks (fastest, 2–4 month turnaround)

### EMNLP Industry Track
- **Fit:** Strong.  Industry deployment of NLP system, applied
  emphasis, room for honest negative-result discussion (e.g.
  parallel-multi-agent slower than serial due to GIL).
- **Deadline:** typically June for November conference.
- **Acceptance rate:** ~30–35%.

### ACL Industry Track
- **Fit:** Strong.  Same as EMNLP industry but earlier in the
  cycle.
- **Deadline:** typically February for August conference.

### USENIX SecAI Workshop / AISec at CCS
- **Fit:** Strong for the guardrail half.  The atomic-chunking
  contribution is novel for the security community.
- **Deadline:** USENIX SecAI typically May; AISec typically July.

### KDD Applied Data Science Track
- **Fit:** Moderate.  Production-deployed retrieval pipeline with
  measurable accuracy/latency numbers.
- **Deadline:** typically February for August conference.

### NeurIPS / ICML Safety Workshops
- **Fit:** Moderate.  Guardrail story fits the safety angle.
- **Deadline:** August–September.

## Conferences (full paper, 4–6 month turnaround)

### ACSAC Industrial Track
- **Fit:** Strong.  Industrial Track explicitly accepts
  deployment-grade work with measurable security properties.
- **Deadline:** typically June for December conference.
- **Acceptance rate:** ~25%.

### CIKM Applied Track
- **Fit:** Strong for the retrieval contribution.
- **Deadline:** typically May for November conference.

### CODS-COMAD (Indian)
- **Fit:** Strong — Indian deployment context, applied research.
- **Deadline:** typically September for January conference.
- **Acceptance rate:** ~40%.

## Journals (longest review, highest weight)

### Information Processing & Management (Elsevier)
- **Fit:** Strong.  Has published applied RAG work.  Section on
  vectorless retrieval would land well.
- **Review:** 3–6 months first round.
- **Impact factor:** ~7.

### Journal of Web Semantics (Elsevier)
- **Fit:** Moderate.  Semantic-search angle aligns.
- **Review:** 3–6 months.

### Sadhana (Indian Academy of Sciences)
- **Fit:** Strong for the Indian-deployment story.
- **Review:** 4–8 months.

## My honest recommendation

For a one-shot publish target, **EMNLP Industry Track or ACSAC
Industrial Track** are the best fit:
- The contribution is an applied-systems story, not a theory
  result.
- Honest reporting of negative results (parallel multi-agent
  slower than serial) is welcomed at industry tracks and tends to
  irritate theory venues.
- Both venues have non-trivial visibility on practitioner CVs.

If turnaround time matters more than venue weight, **CODS-COMAD**
or a **NeurIPS safety workshop** will accept faster.

## What to NOT target

- **NeurIPS / ICML / ACL main track.**  These expect a primary
  theoretical or benchmark-breaking empirical contribution.  Our
  work would be borderline at best and the review process is
  brutal for applied work without a SOTA-beating number.

- **arXiv-only.**  arXiv preprint is fine and recommended in
  parallel with venue submission, but as the only target it
  carries less weight than a peer-reviewed venue.
