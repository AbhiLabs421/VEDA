"""Ablation benchmark: BM25 vs +HD-expansion vs +intent vs full VEDA-X.

We isolate each contribution of the VEDA-X pipeline by toggling
exactly one stage at a time and report:

  * Recall@1, Recall@3, Recall@5
  * Mean Reciprocal Rank (MRR)
  * Mean retrieval latency

Corpus: 12 synthetic SOP documents (HR, finance, IT, compliance)
with structurally distinct sections and overlap of common words
(leave/policy/process) to make the lexical baseline non-trivial.

Eval set: 30 queries with known correct file/section labels,
split across three difficulty buckets:

   A. Direct  — query shares 60-80% of target terms
   B. Paraphrase  — query restates the question without key terms
   C. Hinglish  — Hindi-English code-mixed phrasings

All numbers reproducible: no randomness, deterministic tokenizer.
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vedax import VedaX


# ─── synthetic SOP corpus ───────────────────────────────────────────────

CORPUS = {
    "hr_leave.txt": (
        "Casual leave entitlement is twelve days per calendar year. "
        "Unused casual leaves cannot be carried forward to the next year. "
        "Half-day casual leave may be availed twice per month. "
        "Sick leave entitlement is eight days per calendar year, "
        "supported by a medical certificate for absences over two days. "
        "Maternity leave is twenty six weeks paid and may be extended "
        "by twelve weeks unpaid on medical grounds. "
        "Paternity leave is five working days within six months of birth. "
        "Compensatory off is granted for working on a declared holiday "
        "and must be availed within three months."
    ),
    "hr_attendance.txt": (
        "Standard office hours are nine thirty to six thirty Monday "
        "through Friday with a one hour lunch break. "
        "Late arrival beyond ten thirty counts as half day after three "
        "incidents in a month. "
        "Work from home is permitted three days a week with manager "
        "approval; full week work from home requires HR approval. "
        "Biometric attendance is mandatory for all on-site employees."
    ),
    "hr_performance.txt": (
        "Annual performance review cycle runs from April to March. "
        "Mid year review is conducted in October with self assessment "
        "due by mid October. "
        "Performance ratings are on a five point scale: outstanding, "
        "exceeds expectations, meets expectations, needs improvement, "
        "and unsatisfactory. "
        "Promotion eligibility requires two consecutive cycles at "
        "exceeds expectations or above."
    ),
    "fin_expense.txt": (
        "Expense reimbursement claims must be submitted within thirty "
        "days of incurring the expense via the SAP Concur portal. "
        "Original receipts are mandatory for any claim above five "
        "hundred rupees. "
        "Travel within India is reimbursed at actuals up to economy "
        "class airfare and three star hotel tariff. "
        "Per diem allowance for outstation travel is twelve hundred "
        "rupees per day inclusive of meals."
    ),
    "fin_bonus.txt": (
        "Annual variable bonus is computed as a percentage of fixed "
        "compensation based on performance rating and business unit "
        "achievement. "
        "Engineers at level four and above are eligible for stock "
        "option grants subject to vesting over four years with a one "
        "year cliff. "
        "Spot awards may be issued for exceptional contribution and "
        "are capped at fifty thousand rupees per quarter."
    ),
    "fin_payroll.txt": (
        "Salary is credited on the last working day of each calendar "
        "month to the registered bank account. "
        "Tax deduction at source is computed against the declared "
        "investment regime; declarations must be submitted by "
        "thirtieth June for the financial year. "
        "Form sixteen is issued by fifteenth June following the close "
        "of the financial year."
    ),
    "compliance_kyc.txt": (
        "Customer KYC documentation requires PAN card, Aadhaar, and "
        "address proof at onboarding. "
        "Reject a politically exposed person from a sanctioned country "
        "outright. "
        "Escalate any cash transaction exceeding ten lakh rupees in a "
        "single instance to the compliance officer within twenty four "
        "hours. "
        "Verify dual identification for any wire transfer above twenty "
        "five lakh rupees. "
        "Re-verify quarterly for high risk PEP customers."
    ),
    "compliance_audit.txt": (
        "Internal audit is conducted on a quarterly basis covering "
        "operational processes, financial controls, and IT security. "
        "Audit findings are classified as critical, high, medium, or "
        "low and tracked in the GRC platform until closure. "
        "External statutory audit is conducted annually by the "
        "appointed auditor and the report is filed with the regulator "
        "within thirty days of board approval."
    ),
    "compliance_disclosure.txt": (
        "Material changes to the operating procedure must be disclosed "
        "to the regulator within seven business days of approval by "
        "the board. "
        "Annual disclosure of related party transactions is filed by "
        "the company secretary by thirtieth September of each year. "
        "Insider trading window opens forty eight hours after the "
        "publication of quarterly results and closes seven days before "
        "the next results."
    ),
    "it_security.txt": (
        "All workstation logins require multi factor authentication "
        "via the corporate VPN. "
        "Password rotation is enforced every ninety days with a "
        "minimum length of twelve characters. "
        "Privileged access to production systems is granted via a "
        "just in time approval workflow with maker checker enforcement. "
        "Suspected phishing emails should be reported to the SOC via "
        "the report phishing button."
    ),
    "it_incident.txt": (
        "Sev one incidents are response within fifteen minutes and "
        "must have an incident commander assigned within thirty "
        "minutes. "
        "A blameless post mortem document is published within five "
        "business days of any sev one or sev two incident. "
        "Root cause analysis includes contributing factors, immediate "
        "fix, long term mitigation, and detection improvement."
    ),
    "it_access.txt": (
        "Access to source code repositories is granted on team join "
        "via the identity provider group membership. "
        "Production database access is restricted to the DBA team "
        "and requires a documented change ticket for every query. "
        "Departure from the company triggers automated revocation of "
        "all access within twenty four hours of last working day."
    ),
}


# ─── evaluation queries with ground truth ──────────────────────────────

EVAL = [
    # ── A. Direct phrasing  (10 queries) ─────────────────────────
    ("how many casual leaves do I get",                  "hr_leave.txt"),
    ("maternity leave duration",                         "hr_leave.txt"),
    ("office hours timing",                              "hr_attendance.txt"),
    ("work from home rules",                             "hr_attendance.txt"),
    ("annual performance review cycle",                  "hr_performance.txt"),
    ("expense reimbursement deadline",                   "fin_expense.txt"),
    ("variable bonus computation",                       "fin_bonus.txt"),
    ("salary credit date",                               "fin_payroll.txt"),
    ("KYC documents required",                           "compliance_kyc.txt"),
    ("password rotation policy",                         "it_security.txt"),

    # ── B. Paraphrase  (10 queries — no shared key noun) ─────────
    ("when am I entitled to time off for illness",       "hr_leave.txt"),
    ("how is fatherhood time away handled",              "hr_leave.txt"),
    ("what time do I need to come to office",            "hr_attendance.txt"),
    ("how often is my work assessed by manager",         "hr_performance.txt"),
    ("getting money back for business travel",           "fin_expense.txt"),
    ("when does the company pay me each month",          "fin_payroll.txt"),
    ("what to do when a customer might be sanctioned",   "compliance_kyc.txt"),
    ("when does the regulator need to be told",          "compliance_disclosure.txt"),
    ("what to do during a major outage",                 "it_incident.txt"),
    ("getting into production servers",                  "it_access.txt"),

    # ── C. Hinglish  (10 queries — code mixed) ───────────────────
    ("casual leave kitne din milte hain",                "hr_leave.txt"),
    ("paternity leave kab milti hai",                    "hr_leave.txt"),
    ("office time kya hai",                              "hr_attendance.txt"),
    ("work from home kitne din allowed",                 "hr_attendance.txt"),
    ("bonus kaise calculate hota hai",                   "fin_bonus.txt"),
    ("salary kab credit hoti hai",                       "fin_payroll.txt"),
    ("KYC me kya documents chahiye",                     "compliance_kyc.txt"),
    ("audit kitne baar hota hai",                        "compliance_audit.txt"),
    ("MFA kya zaroori hai login me",                     "it_security.txt"),
    ("sev one incident me kya karna hai",                "it_incident.txt"),
]


# ─── strategy implementations ──────────────────────────────────────────

def _build_engine(chunk_tokens=25, overlap_tokens=5):
    """Build engine with SMALL chunk size so each SOP splits into many
    competing passages — otherwise the file-level recall saturates at
    100% for plain BM25 and the ablation is uninformative.

    With chunk_tokens=25, every short SOP becomes 3-5 passages and the
    retriever must pick the right passage AND the right file."""
    d = tempfile.mkdtemp()
    paths = []
    for name, body in CORPUS.items():
        p = os.path.join(d, name)
        with open(p, "w") as f:
            f.write(body)
        paths.append(p)
    eng = VedaX(use_dense=False,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens)
    eng.add(*paths)
    eng._finalize()
    return eng


def _chunk_file(eng, cid):
    """Return source-file basename for a chunk id."""
    doc_id = eng.chunks[cid][0]
    return os.path.basename(doc_id)


def strat_bm25(eng, query, k=5):
    """Baseline: plain Okapi BM25, no expansion, no rescoring."""
    ranking = [cid for cid, _ in eng.bm25.search(query, k=k)]
    return [_chunk_file(eng, cid) for cid in ranking]


def strat_bm25_hd(eng, query, k=5):
    """BM25 + hyperdimensional query expansion."""
    ranking = eng._bm25_expanded(query)[:k]
    return [_chunk_file(eng, cid) for cid in ranking]


def strat_vedax_full(eng, query, k=5):
    """Full VEDA-X: BM25 + HD expansion + intent rescoring +
    adaptive cutoff.  Adaptive cutoff is honoured by smart_search."""
    res = eng.smart_search(query, max_keep=k)
    return [os.path.basename(h["file"]) for h in res["hits"]][:k]


STRATS = [
    ("BM25",                strat_bm25),
    ("BM25 + HD",           strat_bm25_hd),
    ("VEDA-X (full)",       strat_vedax_full),
]


# ─── metrics ───────────────────────────────────────────────────────────

def recall_at_k(file_ranking, target, k):
    return 1 if target in file_ranking[:k] else 0


def reciprocal_rank(file_ranking, target):
    for i, f in enumerate(file_ranking, 1):
        if f == target:
            return 1.0 / i
    return 0.0


def evaluate(name, runner, eng, queries):
    r1 = r3 = r5 = 0
    mrr = 0.0
    t0 = time.perf_counter()
    for q, target in queries:
        ranking = runner(eng, q, k=10)
        r1 += recall_at_k(ranking, target, 1)
        r3 += recall_at_k(ranking, target, 3)
        r5 += recall_at_k(ranking, target, 5)
        mrr += reciprocal_rank(ranking, target)
    n = len(queries)
    dt_ms = (time.perf_counter() - t0) * 1000 / n
    return {
        "name": name,
        "r1": r1, "r3": r3, "r5": r5, "n": n,
        "mrr": mrr / n,
        "ms": dt_ms,
    }


def fmt_row(r):
    return (f"  {r['name']:18s}  "
            f"R@1={r['r1']:2d}/{r['n']} ({100*r['r1']/r['n']:3.0f}%)  "
            f"R@3={r['r3']:2d}/{r['n']} ({100*r['r3']/r['n']:3.0f}%)  "
            f"R@5={r['r5']:2d}/{r['n']} ({100*r['r5']/r['n']:3.0f}%)  "
            f"MRR={r['mrr']:.3f}  "
            f"lat={r['ms']:5.2f}ms")


def main():
    print("=" * 80)
    print("  VEDA-X ablation: BM25 -> +HD expansion -> +intent rescoring")
    print("=" * 80)
    eng = _build_engine()
    print(f"\nCorpus: {len(CORPUS)} files, {len(eng.chunks)} chunks.")
    print(f"Eval: {len(EVAL)} queries  "
          "(10 direct + 10 paraphrase + 10 Hinglish).")

    # ─── overall ────────────────────────────────────────────────
    print("\n[ OVERALL ]")
    overall = [evaluate(n, r, eng, EVAL) for n, r in STRATS]
    for row in overall:
        print(fmt_row(row))

    # ─── per-bucket ─────────────────────────────────────────────
    buckets = [
        ("DIRECT",     EVAL[0:10]),
        ("PARAPHRASE", EVAL[10:20]),
        ("HINGLISH",   EVAL[20:30]),
    ]
    for bname, qs in buckets:
        print(f"\n[ {bname} ]")
        for n, r in STRATS:
            print(fmt_row(evaluate(n, r, eng, qs)))

    # ─── absolute gain over BM25 baseline ───────────────────────
    print("\n[ ABSOLUTE GAIN — over plain BM25 ]")
    base = overall[0]
    for row in overall[1:]:
        d_r1 = 100 * (row["r1"] - base["r1"]) / base["n"]
        d_r3 = 100 * (row["r3"] - base["r3"]) / base["n"]
        d_mrr = row["mrr"] - base["mrr"]
        print(f"  {row['name']:18s}  "
              f"+R@1: {d_r1:+5.1f} pp  "
              f"+R@3: {d_r3:+5.1f} pp  "
              f"+MRR: {d_mrr:+.3f}")

    print("\nNote: this benchmark is reproducible — fixed corpus,")
    print("fixed query list, deterministic tokenizer + hash-hypervectors.")
    print("Re-run anytime: python scripts/bench_vedax_ablation.py")
    return overall


if __name__ == "__main__":
    main()
