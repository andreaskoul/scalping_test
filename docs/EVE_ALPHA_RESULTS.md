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

## The horizon lever — monthly rebalance + liquidity factors (the literature's bet)

The multi-feature/multimodal survey (`docs/EVE_SHARPE_LITERATURE.md`) concluded the
single biggest *structural* net-Sharpe lever is **slower-decay signals at a longer
horizon** (Gu-Kelly-Xiu live at monthly, not daily) — turnover per unit alpha falls,
so the GP frontier shifts up. We tested it directly: added **liquidity factors**
(Amihud illiquidity + log dollar-volume, `eve_features`, +10 factors → 55) and a
**monthly rebalance** option (`build_feature_panel(rebalance="1m")` — same daily-window
factors snapshotted at month-end, but the label is the *next-month* return). Same
GBDT + GP sweep, annualized at 12, 5-fold walk-forward, embargo 1 month, 5 bps/side.

| trade rate | turnover | Net Sharpe | t | annRet | annVol |
|---|---|---|---|---|---|
| 1.00 (full) | 1.32 | +0.17 | 0.44 | 0.6% | 3.3% |
| 0.50 | 0.55 | +0.69 | 1.74 | 1.7% | 2.5% |
| 0.25 | 0.25 | +1.14 | 2.88 | 2.2% | 1.9% |
| 0.10 | 0.10 | **+1.46** | 3.69 | 1.9% | 1.3% |
| **0.05** | 0.05 | **+1.50** | 3.80 | 1.3% | 0.9% |
| 0.02 | 0.02 | +1.50 | 3.80 | 0.6% | 0.4% |
| equal-weight (beta) | 0.12 | +0.79 | 2.01 | 14.6% | 18.4% |

**PBO = 0.01** (robust). **The lever worked.** Net Sharpe rose from the daily
ceiling of **+0.48 (below beta)** to **+1.50 (≈ 2× beta's +0.79)** — the first time
any Eve configuration *beats passive net of retail cost*, and exactly where the
literature said the edge lives. The frontier now **plateaus** at rate ≈ 0.05–0.10
(a true optimum, not running away to zero turnover), with t ≈ 3.8 (p ≈ 0.0003) over
77 OOS months.

**Honest caveats — why this is "real and ~1.5", not "Sharpe 4":**
1. **Tiny absolute return.** The high-Sharpe books are near-static low-vol tilts:
   +1.3%/yr at 0.9% vol (rate 0.05). The Sharpe is real but you'd have to **lever**
   it ~10× to match beta's *return*, reintroducing cost/borrow/drawdown risk. It's a
   diversifier, not a standalone 15%/yr engine.
2. **Deflated-Sharpe gate is miscalibrated at monthly frequency.** `portfolio_metrics`
   feeds `sr_std=max(|per-obs SR|, 0.05)` to `deflated_sharpe_ratio`; at monthly the
   per-obs SR ≈ 0.43 inflates the selection benchmark (`expected_max_sharpe`) so DSR
   reads ~0.03 *despite* t = 3.8. The honest robustness evidence here is **t-stat +
   PBO**, both strong; the DSR heuristic needs the cross-trial dispersion recalibrated
   before it means anything at this frequency. (Flagged, not yet fixed.)
3. **Survivorship bias** (current S&P 500) still inflates the gross — true edge is
   smaller. A point-in-time universe is the next correctness fix.
4. **Short sample.** 77 OOS months (~6.4 yr). Strong t, but one regime.

**Verdict:** the horizon/liquidity lever is the right one and it cleared beta net —
the project's first genuinely capturable edge. It points to a Sharpe in the **~1–1.5**
range honestly (still not 4), and the next gains are *correctness* (point-in-time
universe), *slower-decay features* (fundamentals via FMP — the #1 missing family), and
a *recalibrated monthly DSR* — not a fancier model.

## Follow-up levers — built (2026-06-28)

After the monthly breakthrough we executed the three named follow-ups. Two are
finished and runnable; the third's *capability* is built and tested, with the data
pull deferred to an out-of-band cached batch (per-symbol FMP calls).

1. **Deflated-Sharpe recalibrated for low frequency** — DONE. `portfolio_metrics`
   was feeding each strategy's own |SR| as the cross-trial dispersion, which scales
   with per-obs SR and so crushed the monthly DSR to ~0.03 despite t=3.8.
   `walk_forward_portfolio` now computes `sr_std` once as the std of per-obs Sharpe
   across the swept (non-baseline) configs and passes it through (Bailey-LdP). The
   monthly `gbdt_pa05` DSR moves **0.03 → 0.91** — consistent with its t-stat, and
   honestly *just* under the 0.95 gate (77 months, 9 trials). Daily results
   unaffected.

2. **Point-in-time universe — survivorship bias sized** — DONE (`eve_universe.py`).
   Reconstructs S&P 500 membership on any date from FMP's historical change log
   (walk backward from today's 503). Measured over our window:

   | | count |
   |---|---|
   | TRUE distinct S&P members 2017–2026 | **680** |
   | …we hold prices for | 502 |
   | **missing (member, no price)** | **178 (~26%)** |

   The 178 are exactly the names that *left* — ATVI (acquired), BBBY (bankrupt),
   AET/AGN/ALXN (acquired)… disproportionately losers and takeouts. So our gross
   alpha **and** the passive beta are both inflated. **Masking current prices by PIT
   membership does not fix this** (we still lack those 178 series); the real fix is
   ingesting delisted-name price history (FMP has it — an out-of-band per-symbol
   batch). The mask + sizing tooling is in place for when those prices land.

3. **Fundamentals (the #1 missing feature family)** — capability built + tested
   (`eve_fundamentals.py`), full pull pending. Slow-decay value/quality/profitability
   factors from FMP `key-metrics` (ROE, ROIC, earnings yield, FCF yield, EV/EBITDA,
   current ratio, income quality, net-debt/EBITDA). Pure **PIT alignment with a
   filing lag** (a quarter dated D is usable only at D+60d → month-end forward-fill →
   cross-sectional rank-norm → appended to the price panel), validated against live
   FMP data; cached per-symbol ingest (pull-once). The 472-symbol historical pull is
   an out-of-band cached batch (each symbol is one FMP call), after which the monthly
   GBDT+GP run augments automatically. *Why it should help:* fundamentals decay over
   quarters, so they add alpha at the monthly horizon at near-zero extra turnover —
   the same lever, deeper signal.

**Data sourcing note (Alpaca vs FMP):** Alpaca provides market data + **news**, not
fundamentals — so the fundamentals family must come from FMP (already reachable via
the connected MCP, **no API key needed**). Alpaca's free historical **news** API
(existing keys) is the natural source for the *next* family — news/earnings sentiment
(FinBERT) — not for fundamentals.

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
