# Eve Alpha Lane — the qlib recipe (A–E), results

_As-of 2026-06-27. The breadth + factor-library + GBDT + Top-k + validation
pipeline the survey called for, run on real data and reported honestly._

## What was built (A–E)

| | Lever | Module |
|---|---|---|
| A | **Breadth** — all 503 S&P 500 daily histories (2017–2026; 472 usable after a long-history filter) | `eve_yahoo`, `eve_ingest` |
| B | **Alpha158-style factor library** — 45 leakage-safe factors (K-line shape + roc/ma/std/rsv/rsi/vma/corr/beta × 5 windows), cross-sectionally rank-normalized | `eve_features.py` |
| C | **GBDT** (LightGBM, native API) predicting cross-sectional return-rank | `eve_gbdt.py` |
| D | **Top-k long-short** + GBDT→RL allocator; **embargo** in walk-forward | `eve_portfolio.py`, `eve_rl.py` |
| E | **PBO/CSCV** (Bailey et al.) + Deflated Sharpe | `stats.py` |

## Results (472 names, daily, 2017–2026, 5-fold walk-forward, embargo=2)

| | strategy | Ann.Ret | Ann.Vol | Sharpe | Deflated SR | turnover |
|---|---|---|---|---|---|---|
| **GROSS (0 bps)** | gbdt_daily | 7.6% | 9.1% | **+0.83** | 0.46 | 1.43 |
| | gbdt_hold5 | 2.7% | 8.9% | +0.30 | 0.09 | 0.34 |
| | gbdt_hold20 | −1.3% | 7.5% | −0.18 | 0.01 | 0.10 |
| | equal_weight (beta) | 15.9% | 21.2% | 0.75 | 0.42 | 0.02 |
| **NET (5 bps)** | gbdt_daily | −10.5% | 9.1% | **−1.15** | 0.00 | 1.43 |
| | gbdt_hold5 | −1.6% | 8.9% | −0.18 | 0.01 | 0.34 |
| | gbdt_hold20 | −2.6% | 7.5% | −0.35 | 0.00 | 0.10 |
| | equal_weight (beta) | 15.7% | 21.2% | 0.74 | 0.41 | 0.02 |

**PBO = 0.00–0.01** in every run — the result is **robust, not a selection artifact.**

## The finding

1. **The factor model works — gross.** `gbdt_daily` has a real cross-sectional
   edge: gross Sharpe **0.83**, t=2.1, PBO≈0. Breadth + a factor library + GBDT is
   the right recipe; it produces genuine predictive signal (this is roughly qlib
   territory *before* costs).
2. **The alpha is short-horizon and small.** Holding 5 days halves it (0.30),
   holding 20 days kills it (−0.18). It decays faster than turnover control can
   save — so crude turnover control (hold-N) trades alpha away for cost savings at
   a losing exchange rate.
3. **Retail cost is the wall.** Gross alpha ≈ 7.6%/yr at 1.43×/day turnover. At
   5 bps that's ≈ 18%/yr in costs → net −10.5%. **Break-even ≈ 2 bps per side —
   institutional execution.** No turnover scheme rescues it at retail costs.
4. **So Sharpe ≫ 1 is not real here.** Even a robust, non-overfit factor model
   with real gross alpha nets **negative** at retail costs. A deflated, post-cost
   OOS Sharpe of 4 on a daily equity strategy is not attainable honestly — the
   binding constraint is execution cost, not model quality.

**Caveat (inflates everything above):** the universe is the *current* S&P 500, so
results carry **survivorship bias** — true gross alpha is lower than shown. A real
backtest needs point-in-time constituents.

## Where the remaining edge could live (honest, not Sharpe-4)

- **Adaptive cost-aware sizing** (`GBDTAllocator`: GBDT→RL with the DSR +
  turnover-penalty objective) instead of fixed hold-N — optimizes the alpha/cost
  frontier continuously. Must run in a **separate process** from LightGBM (torch's
  `libomp` and LightGBM's `libiomp5` abort when sharing a process on macOS). Likely
  still below beta, but the right tool to quantify the best achievable net.
- **Slower-decay signals** (fundamentals, longer horizons) — lower turnover per
  unit alpha; the only structural fix for the cost wall.
- **Lower costs** — institutional execution (<2 bps). Not available to us.
- **Point-in-time universe** — to remove survivorship bias and get an honest gross.
