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

## Cost-aware optimization — Gârleanu–Pedersen partial adjustment

Research-led decision (after surveying the top RL/ML trading repos): the only
idea that targets *our* wall — the turnover/alpha-decay tradeoff — is **cost-aware
optimal allocation** (CFMTech's deep-RL-for-portfolio, whose DDPG approximates the
**Gârleanu–Pedersen 2013** closed form: trade only a fraction `rate` of the way
toward an "aim" portfolio each step). We built the deterministic version
(`GBDTPartialAdjust`) — robust, interpretable, torch-free — and swept `rate` on the
real GBDT alpha:

| trade rate | turnover | Net Sharpe | Deflated SR |
|---|---|---|---|
| 1.00 (full rebalance) | 1.43 | **−1.15** | 0.00 |
| 0.50 | 0.51 | −0.34 | 0.00 |
| 0.25 | 0.25 | −0.03 | 0.00 |
| 0.10 | 0.10 | +0.23 | 0.01 |
| 0.05 | 0.05 | +0.36 | 0.02 |
| **0.02** | 0.02 | **+0.48** | 0.04 |
| equal-weight (beta) | 0.02 | +0.74 | 0.15 |

PBO = 0.04 (robust). **The method works**: slowing the trade rate lifts net Sharpe
monotonically from −1.15 to **+0.48**, recovering a positive net from the signal
that lost money at full turnover — the turnover wall was the killer, and GP is the
right tool (no need to reinvent it with RL).

**But the honest ceiling is +0.48** — still **below passive beta (0.74)** and not
deflated-significant. Net Sharpe is still rising as `rate→0`, i.e. the optimum is a
near-static factor tilt earning a tiny sliver (~0.9%/yr): the alpha decays so fast
that capturing any of it net forces such slow trading that almost nothing is left,
and that sliver is smaller than simply owning the market.

**Decision implication for the RL lane:** the GP sweep already maps the *optimal*
cost-aware frontier this signal can reach (≈ +0.48). A learned DSR-RL allocator
chases the same frontier with added overfit risk and the LightGBM/torch OpenMP
hazard; it will not materially exceed a sub-beta, sub-significant ceiling. So the
robust call is **not** to chase it with RL — the binding constraint is the signal's
fast decay vs retail cost, which no allocator can fix. **Sharpe ≫ 1 is not
attainable here; ≫ 4 is not real.**

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
