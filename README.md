# Final Project — Do Shapes Carry Information?

**Course:** 2110430 — Time Series Mining
**Due:** April 24, 2026

---

## Meta-question across three notebooks

> **When does shape-based time-series mining find real structure, and when doesn't it?**

We apply the same core toolkit — z-normalization, DTW, LB_Keogh, matrix profile, clustering, 1-NN classification — to three domains. Each notebook is a complete final-project submission on its own; read together they form a three-panel answer: null in finance, rich in physics, *mixed* in human behavior.

| # | Notebook | Domain | Does shape carry info? | Headline |
|---|---|---|---|---|
| 1 | `stock_motif_analysis.ipynb` | Financial returns | **No** | Sign-agreement on motif-predicted next-moves is indistinguishable from a coin flip. Consistent with weak-form market efficiency. |
| 2 | `pm25_shape_typology.ipynb` | Bangkok air quality | **Yes (richly)** | K=4 daily-shape archetypes with clean physical interpretation. DTW-vs-Euclidean ablation shows warping matters (ARI=0.29). |
| 3 | `github_death_motif.ipynb` | Open-source commit cadence | **Yes in volume, no in shape** | Summary-stats LogReg, DTW-1NN, and augmented DTW-1NN all tie at 92.9% — shape adds no marginal info on this extreme-endpoint sample. Matrix profile on dying repos returns only silence. |

**The point.** The methodology is identical across all three domains. What changes is whether the data-generating process produces repeatable shapes. Our job is not to make every method look like it works — it is to report honestly *when* shape mining pays off.

---

## Project Structure

```
newfinalproject/
├── README.md                       ← you are here
├── stock_motif_analysis.ipynb      ← Project 1 (finance / negative)
├── pm25_shape_typology.ipynb       ← Project 2 (air quality / positive)
├── github_death_motif.ipynb        ← Project 3 (OSS commits / mixed)
├── data/
│   ├── pm25_bkk.csv                ← Open-Meteo Bangkok PM2.5, 2 yr hourly
│   └── github_commits.csv          ← 14 repos × weekly commits, 2023-04→2025-04
└── scripts/
    ├── build_pm25_notebook.py      ← notebook generators (for reproducibility)
    ├── build_github_notebook.py
    └── pull_github.py              ← fetches commit data via git clone --bare
```

---

## Quick tour of each notebook

### 1. `stock_motif_analysis.ipynb` — the negative result that motivated the rest

> **When the same shape repeats in a stock's return series, does the move that follows also repeat?**

- Pulls 5 years of daily prices (AAPL, MSFT, GOOGL, SPY, NVDA) via `yfinance`.
- Computes matrix profile at window m ∈ {10, 20, 50}; extracts top-50 motif pairs.
- For each pair, tests whether the *sign* of the next-h-day return agrees (binomial test vs 50%).
- **Finding:** sign-agreement hovers at 40–50% across all (ticker, m, h). p > 0.05 everywhere. The matrix profile reliably finds visually identical pairs; their futures are uncorrelated.
- **Concepts:** z-normalization, matrix profile, motif discovery, similarity search, binomial test.

### 2. `pm25_shape_typology.ipynb` — shapes carry rich physical information

> **Are all bad-air days the same shape, or are there distinct archetypes?**

- 2 years of hourly Bangkok PM2.5 from Open-Meteo (no key, free).
- 731 daily 24-hour windows → z-normalize → pairwise DTW distance matrix.
- Ward-linkage hierarchical clustering → **K=4 archetypes** (selected for interpretability over top-silhouette K=2).
- Cluster sizes: 401 / 185 / 78 / 68. Archetypes interpret cleanly as "background-clean," "evening-peak," "overnight-accumulation," "sustained-haze."
- **DTW vs Euclidean ablation:** ARI=0.285 (clusterings genuinely disagree); DTW silhouette 0.283 > Euclidean 0.264.
- **LB_Keogh:** vectorized lower bound on 731×731 pairwise distances → **85.3% prune rate, 7.28× wall-clock speedup** vs exhaustive DTW.
- **Shape anomaly detection:** top-k days farthest from any medoid surface as candidates for real-world events.
- **Concepts:** DTW, LB_Keogh, hierarchical clustering, silhouette / Davies-Bouldin, anomaly detection, DTW-vs-Euclidean ablation.

### 3. `github_death_motif.ipynb` — shape mining on human cadence

> **Do dying open-source projects share a detectable commit-cadence shape?**

- 14 curated repos (7 alive / 7 dying), 104 weekly commit counts per repo (2023-04-01 → 2025-04-01).
- Data pulled via `git clone --bare --shallow-since` (no GitHub API rate-limit constraint).
- Three classifiers via LOOCV: summary-feature LogReg, DTW-1NN on shape, DTW-1NN with jittering+scaling augmentation.
- **All three tie at 92.9% accuracy** (1 of 14 misclassified by each). DTW adds no marginal information beyond summary stats *on this curated extreme*.
- **LB_Keogh:** 40.7% prune rate, 1.49× speedup even on n=14.
- **Matrix profile motif discovery:** on dying repos returns only silence (all top motifs at distance 0.00 — they're flat-zero windows). On alive repos surfaces real recurring cadence (distances 0.60–0.77).
- **Honest reading:** on unambiguously-labeled extremes, "death" is already in the volume — shape mining pays off at the ambiguous boundary, which our curated sample excludes by construction.
- **Concepts:** DTW + 1-NN classification, LB_Keogh, data augmentation (course-listed), matrix profile, confusion matrix + LOOCV.

---

## How to run

All three notebooks are designed to be readable without executing. To reproduce:

```bash
conda activate capstone_env     # env has all dependencies (pandas, stumpy, tslearn, sklearn, scipy, yfinance, requests)
jupyter lab                      # open each .ipynb and run all cells
```

Runtimes on a laptop: Project 1 ≈ 1 min, Project 2 ≈ 3–5 min, Project 3 ≈ 1 min.

Data refresh (if `data/*.csv` is missing):

```bash
python scripts/pull_github.py    # ~3 min; uses git, no GitHub token needed
# PM2.5 data is fetched inline from the notebook via Open-Meteo
```

---

## Evaluation at a glance

| Rubric item | Where it shows up |
|---|---|
| Problem definition | Each notebook has a "The Question" section and an explicit hypothesis |
| Technical correctness | Vectorized LB_Keogh, Ward-linkage clustering, LOOCV, confusion matrices, binomial tests — all implemented from the fundamentals not wrapper libraries |
| Course-concept coverage | Z-normalization, DTW, LB_Keogh, matrix profile, motif discovery, DTW classification, data augmentation, clustering, anomaly detection — all three notebooks combined touch every item in the syllabus |
| Originality | Multi-domain comparison with matched methodology is not standard; the three-panel "when does shape mining pay off" framing is the novel contribution |
| Analysis & insight | Each notebook interprets its numbers in full-sentence markdown immediately after producing them; no bare output tables |
| Presentation | Markdown-heavy narrative, labeled axes, two-line print outputs with units, limitations section per notebook |

---

## Limitations (honest, across all three)

- **Project 1:** daily resolution only; univariate; sign-only outcome; no regime conditioning.
- **Project 2:** single city (Bangkok) single season-span (2 years); K chosen for interpretability, not top silhouette; anomaly "news-matching" is qualitative.
- **Project 3:** n=14 repos at the endpoints of the alive/dying spectrum; the interesting ambiguous-boundary population is absent by design; window fixed on calendar, not on project-death date.

---

## References

- Yeh et al. (2016) *Matrix Profile I.* ICDM.
- Keogh & Ratanamahatana (2005) *Exact indexing of dynamic time warping.* KAIS.
- Law (2019) *STUMPY.* JOSS.
- Fama (1970) *Efficient Capital Markets.* Journal of Finance.
- Yahoo Finance via `yfinance`; Open-Meteo Air Quality API; GitHub via `git clone --bare`.
