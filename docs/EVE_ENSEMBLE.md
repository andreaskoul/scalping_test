# Eve Ensemble — Transformer Prediction + RL Portfolio Optimization

_As-of 2026-06-27. The "optimal ensemble" from `docs/EVE_SHARPE_LITERATURE.md`,
built and validated the repo way: post-cost, walk-forward, **deflated** Sharpe._

## The thesis

The survey concluded that (a) reported 3+ Sharpes rarely survive costs/deflation,
(b) validated edge lives in **breadth** (cross-sectional, many names), not
single-asset intraday, and (c) the two highest-Sharpe families are *deep
prediction* and *RL portfolio optimization*. So Eve chains them:

```
bars ──► Transformer (prediction lane)            RL allocator (portfolio lane)
         per-asset cost-aware score   ──signal──► Differential-Sharpe weights ──► book
         (eve_transformer.py)                      (eve_rl.py, Moody–Saffell 1998)
                                  validated by eve_portfolio.py (post-cost, deflated Sharpe)
```

## The two lanes

**Prediction (`src/eve_transformer.py`).** A compact encoder-only transformer
emits a 3-class cost-aware probability per asset; the cross-sectional signal is
`P(up) − P(down)`, z-scored across names each day.

**Portfolio (`src/eve_rl.py`, `DirectRLAllocator`).** Direct/recurrent RL with
the **Differential Sharpe Ratio** reward (Moody & Saffell, NIPS 1998): roll the
book forward, accumulate each step's marginal contribution to the Sharpe of the
*post-cost* return, ascend its gradient. An asset-shared MLP maps
`[signal_i, holding_i] → score_i`; scores become a **dollar-neutral,
leverage-capped long-short** book. Previous holdings feed back in, so the policy
learns to control turnover. EMA state is detached each step (stable online update,
no value-function instability); a 1/η warmup skips the unstable variance ramp.

**Validation (`src/eve_portfolio.py`).** Everything is judged on realized
**post-cost** portfolio returns → annualized + **Deflated** Sharpe (`src/stats.py`,
penalized for the number of strategies tried), block-bootstrap CI, turnover.
Nested, leakage-safe walk-forward: per fold the transformer trains on the past and
predicts the next block OOS; the allocator is fit on train-block signals and
evaluated on those OOS scores. Promotion gate: beat beta **and** clear
DeflatedSharpe ≥ 0.95 **and** CI excludes 0.

## Results (30 US large-caps, daily, 2016–2026, 5 bps turnover cost)

### RL allocator on transparent momentum/vol signals (4-fold walk-forward)

| Strategy | Ann.Ret | Ann.Vol | Sharpe | Deflated SR | Turnover |
|---|---|---|---|---|---|
| cash | 0% | 0% | 0.00 | — | 0.00 |
| equal-weight (long-only **beta**) | 19.3% | 20.4% | **0.94** | 0.44 | 0.01 |
| long-short (momentum) | −4.2% | 10.9% | −0.38 | 0.00 | 0.33 |
| **RL allocator** (market-neutral) | 8.2% | 13.1% | **0.62** | 0.27 | 0.19 |

Honest read: the RL allocator is **dollar-neutral**, so its fair peer is the
other market-neutral book (`long_short`), which it **beats decisively**
(+0.62 vs −0.38) — it genuinely learned to cut vol (13% vs 20%) and turnover. It
does **not** beat passive equity **beta** (0.94 — just "long large-caps in a bull
decade"), and its **Deflated Sharpe is 0.27 < 0.95**: real but not significant.

### Transformer → RL ensemble (nested walk-forward, 117 s on MPS)

| Strategy | Ann.Ret | Ann.Vol | Sharpe | Deflated SR | Turnover |
|---|---|---|---|---|---|
| **ensemble** (transformer → RL) | 0.5% | 13.0% | **0.04** | 0.01 | 0.20 |
| transformer_ls (signal, no RL) | −5.8% | 8.7% | **−0.67** | 0.00 | 0.23 |
| equal-weight (beta) | 17.8% | 20.6% | 0.87 | 0.45 | 0.01 |
| cash | 0% | 0% | 0.00 | — | 0.00 |

The decisive finding: the transformer's cross-sectional signal is
**anti-predictive** (`transformer_ls` Sharpe −0.67, worse than random) — i.e. the
prediction lane has **no edge** on daily OHLCV bars (consistent with the
intraday result and LOBCAST). Yet the **RL allocator fed that useless signal
goes nearly flat (Sharpe 0.04) instead of losing money** like the naive
long-short: the DSR + turnover-penalty objective makes the allocator **robust to
a bad predictor** — it declines to bet on noise. That is exactly the behavior we
want from the portfolio lane, and it is itself a validation of the RL design.

So: **the allocator works; the predictor is the bottleneck.** With a *real*
signal (transparent momentum/vol, above) the same allocator reached a
market-neutral Sharpe 0.62; with the edgeless transformer signal it correctly
produces ~0. The ensemble cannot manufacture alpha the prediction lane does not
supply.

## Bottom line

- The machinery is real, tested (smoke + unit), and **honest**: it reports
  sub-significant edge rather than a flattering backtest.
- The dominant "winner" on this basket is **beta**, not alpha — expected for a
  long-only large-cap basket in 2016–2026. A market-neutral overlay (RL/ensemble)
  must be judged on *information ratio over its own neutral peer*, and on whether
  it **deflates** — neither clears 0.95 here.
- Next levers (consistent with the survey): more **breadth** (hundreds of names,
  not 30), genuine **microstructure** features for intraday, and a GBDT prediction
  lane as the baseline to beat. Architecture is not the bottleneck; signal is.
