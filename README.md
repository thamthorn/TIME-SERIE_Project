# Stock Motif Discovery: Do Recurring Chart Patterns Predict Anything?

**Course:** 2110430 — Time Series Mining
**Due:** April 24, 2026

---

## The Question

Technical analysts believe that chart patterns repeat — when a familiar shape shows up in a stock chart, they claim the move that follows will also repeat.

This project tests that belief:

> **When the same shape (motif) repeats in a stock's return series, does the move that follows also repeat?**

---

## TL;DR Answer

**No.** Visually identical patterns in daily stock returns are followed by moves that are statistically indistinguishable from coin flips. Evidence supports weak-form market efficiency.

---

## Method (One Diagram)

```
Stock prices (5 years, 5 tickers)
        ↓
Convert to log returns
        ↓
Matrix Profile (window m = 20 days)
        ↓
Top-50 motifs (most-similar pairs)
        ↓
For each motif pair:
  occurrence A → next-h-day cumulative return → sign
  occurrence B → next-h-day cumulative return → sign
  Do they AGREE?
        ↓
Binomial test vs 50% coin flip
        ↓
Repeat for h ∈ {1, 5, 10} and m ∈ {10, 20, 50}
```

---

## Key Concepts Used

| Concept | Where it's used |
|---|---|
| **Z-normalization** | Compare pure *shape*, not magnitude |
| **Matrix Profile** | Find every window's closest twin, exact and fast |
| **Motif discovery** | Low MP values = most-repeated shapes |
| **Similarity search** | Euclidean distance on z-normalized windows |
| **Binomial test** | Is sign-agreement rate different from 50%? |

---

## Project Structure

```
newfinalproject/
├── README.md                      ← you are here
└── stock_motif_analysis.ipynb     ← full pipeline + analysis
```

---

## How to Run

### 1. Install dependencies

```bash
pip install stumpy yfinance scipy matplotlib pandas numpy
```

### 2. Open the notebook

```bash
jupyter notebook stock_motif_analysis.ipynb
```

### 3. Run all cells

The notebook is self-contained — data is downloaded automatically via `yfinance`. Runs top-to-bottom in ~1–2 minutes on a laptop (no GPU needed).

---

## Notebook Sections

| Section | What it does |
|---|---|
| **1. Introduction** | The question and hypotheses (H0 vs H1) |
| **2. Setup** | Imports, random seed |
| **3. Data** | Download AAPL / MSFT / GOOGL / SPY / NVDA, convert to log returns |
| **4. Matrix Profile** | Compute MP for AAPL and plot the valleys |
| **5. Motif Discovery** | Extract top-3 motif pairs and visualize overlays |
| **6. Predictive Test** | Sign-agreement test + binomial p-values at h=1, 5, 10 |
| **7. Robustness** | Repeat across all 5 tickers × 3 window sizes |
| **8. Discussion** | Findings, limitations, what would change the answer |
| **9. Conclusion** | Direct answer to the research question |

---

## Key Results

- **Sign-agreement rate** hovers around **40–50%** across all tickers and window sizes.
- **Binomial p-values > 0.05** in all cases — we cannot reject the null (coin flip).
- The Matrix Profile *does* find visually near-identical motif pairs — the issue is that visual similarity does not translate into forward-return repeatability.

---

## Why This Finding Matters

A **negative result** with strong evidence is a contribution:

1. Supports **weak-form market efficiency** — patterns simple enough to be detected with an off-the-shelf tool are already arbitraged away.
2. Cautions practitioners: **"recurrence ≠ predictiveness"**. The Matrix Profile optimizes for input similarity, not for future outcomes.
3. Points future work toward **richer similarities** (DTW, learned embeddings) and **regime-conditioned analysis**.

---

## Limitations (Honest List)

- **Daily resolution** — intraday predictability not tested.
- **Univariate** — no cross-asset motifs (e.g., oil → airlines).
- **Z-normalized Euclidean only** — a warping-aware or learned similarity might find predictive structure.
- **No regime conditioning** — trending vs mean-reverting periods mixed together.
- **Sign-only outcome** — magnitude of forward return ignored.

---

## References

- Yeh, C.-C. M. et al. (2016). *Matrix Profile I: All Pairs Similarity Joins for Time Series.* ICDM.
- Law, S. M. (2019). *STUMPY: A Powerful and Scalable Python Library for Time Series Data Mining.* JOSS.
- Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance.
- Yahoo Finance data via the `yfinance` Python package.
