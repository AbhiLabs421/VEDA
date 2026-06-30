# Research paper: VEDA-X

Submission-ready research paper for the VEDA-X stack.

```
paper/
├── paper.tex                — IEEE conference template, single file
├── generate_figures.py      — pure-stdlib SVG figure generator
├── figures/                 — six SVG figures (see below)
├── abstract.md              — 200-word plain-text abstract
├── cover_letter.md          — venue-agnostic cover letter
├── submission_targets.md    — short list of venues with fit notes
└── README.md                — this file
```

## Compiling the PDF

The repository ships LaTeX source and SVG figures.  The build needs
three commands:

```bash
# 1. Generate SVG figures (reproducible, deterministic, no deps).
python paper/generate_figures.py

# 2. Convert SVG → PDF for pdflatex inclusion.
#    (rsvg-convert from librsvg2-bin, or Inkscape's CLI)
cd paper
for f in figures/*.svg; do
  rsvg-convert -f pdf -o "${f%.svg}.pdf" "$f"
done

# 3. Standard LaTeX build.
pdflatex paper.tex
bibtex   paper
pdflatex paper.tex
pdflatex paper.tex
```

If neither `rsvg-convert` nor Inkscape is installed, every figure
SVG can be opened directly in a browser and exported to PDF; or
replace `\includegraphics{figures/figN.pdf}` with the `svg` package
(`\includesvg{...}`) which calls Inkscape transparently.

## Figures

| File | Purpose |
|---|---|
| `fig1_architecture.svg`        | 5-layer system architecture |
| `fig2_defense_in_depth.svg`    | guardrail pyramid |
| `fig3_atomic_chunking.svg`     | critical-block boundary expansion (before/after) |
| `fig4_attack_catch_rate.svg`   | adversarial bar chart (18/18, 12/12, 14/14, 6/6, 15/15) |
| `fig5_latency.svg`             | single vs sequential multi vs parallel multi-agent |
| `fig6_centroid_scatter.svg`    | attack vs legit cosine separation |

All figures are deterministic — re-running `generate_figures.py`
produces byte-identical SVGs.

## Reproducing the numbers in the paper

| Claim | Script | Expected |
|---|---|---|
| 100% adversarial catch, 0% FP | `scripts/test_guardrail_adversarial.py` | 18/18 direct, 12/12 sneaky, 14/14 semantic, 6/6 PII, 0/15 FP |
| 91% recall@3, 100% completeness | `scripts/bench_critical_retrieval.py` | recall 10/11, complete 10/10 |
| Latency 47 / 63 / 103 ms | `scripts/bench_single_vs_multi_agent.py` | single fastest |
| 165 unit tests pass | `python -m unittest discover -s tests` | OK |

## Submission targets

See `submission_targets.md` for the shortlist.  Quick view:

- **Workshop tracks (fast, high acceptance):** EMNLP industry track,
  ACL applied NLP, KDD applied data science, ICML safety workshops.
- **Security venues:** USENIX SecAI workshop, AISec at CCS,
  ACSAC industrial track.
- **Information-systems journals:** *Information Processing &
  Management*, *Journal of Web Semantics*.
- **Indian / regional:** *Sadhana* (Indian Academy of Sciences),
  COMSNETS short paper, CODS-COMAD.

Pick by reviewer fit and your timeline.
