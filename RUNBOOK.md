# Max-EV assembly — runbook

This bot started as a taker-only latency/model arb. It is now a multi-edge
engine assembled from the research pass (two academic papers + a survey of the
Polymarket bot ecosystem). **The max-EV profile is the default** — running
`python -m src.main` turns every edge on; env vars only *detune* it.

> Paper-trade is still the default. `--live` sends real orders and needs
> `POLY_PRIVATE_KEY`. Read the caveats at the bottom before going live.

---

## The edges, where they live, and why they pay

| # | Edge | Module | Evidence | Expected-PnL mechanism |
|---|------|--------|----------|------------------------|
| 1 | **Favourite-longshot tilt** | `pricing.wedge_estimate`, `signal._check_leg` | Portnaya 2026, Tbl 5 (β=−0.398; +6.3pp) | Polymarket YES is systematically rich, most at low p* / long TTE. We **haircut BUY edges** by the expected wedge and **re-admit the low tail on SELL** — stops buying into the longshot premium that was bleeding us. |
| 2 | **Maker / post-only** | `execute.py`, `pricing.maker_rebate_per_share` | CLOB v2 rebates (~20% crypto), liquidity rewards | Flips fee from −1.75% taker cost to a **+rebate**. On a wide spread the engine posts inside instead of crossing; same signal, fee sign reversed. |
| 3 | **Resolution-aware strike** | `resolution.py`, `main` refresher | Chainlink settles 5m/15m/threshold; Binance candle settles hourly | Anchors Up/Down strike to the real **Price-to-Beat** (window-open ref) instead of the biased first-observation hack — re-enables the highest-volume segment that `SKIP_UPDOWN` had killed. |
| 4 | **Perp-funding carry + basis** | `funding.py` | DRADIS "Basis"; Portnaya uses exchange r | Funding → risk-neutral carry `r` in the pricer, plus a capped smart-money **fade** of extreme funding. |
| 5 | **OFI/ML directional nudge** | `microstructure.py`, `pricing.physical_to_risk_neutral` | Deep et al. 2025 (OFI 43% importance, 88% AUC) | A transparent OFI-dominant logistic gives a physical P(up); converted to risk-neutral (MMM) and blended into p*, scaled by p(1−p). |
| 6 | **OBI toxicity veto** | `microstructure.MicrostructureEngine` | DRADIS OBI veto −0.60 | Blocks takers crossing into stacked opposing size — the adverse-selection trap behind most of the bleed. |
| 7 | **Crowd-momentum + impulse-fade** | `microstructure.py` | reference repo; `polyrec` fade backtest | Continuation in the last ~3 min before expiry; mean-reversion fade of over-extensions when there's time to revert. Regime-split by TTE so they never fight. |
| 8 | **Mean-reversion overlay** | `meanrev.py` | Portnaya AR(1) 4h half-life | Fades `D_t = P_poly − P_fair` when it stretches past its own EWMA. Higher-capacity, latency-insensitive. |
| 9 | **Kelly sizing** | `sizing.py` | standard | Sizes by edge/variance, shrinks near coin-flip. Caps (never exceeds notional/risk limits). |
| 10 | **Combinatorial arb** | `arbitrage.py` | Saguillo 2025 ($40M) | Adds **rebalance** (YES+NO<1) and **bucket/box** (YES(K_low)+NO(K_high)<1) to the existing strike-ladder, unified as `ComboArb`. |
| 11 | **Cross-venue arb** | `kalshi.py` | Gebele 2026 (2–4% LOOP gaps) | Detects Polymarket↔Kalshi lock-$1 opportunities on identical BTC contracts. **Scan-only by default.** |

Validate-first tooling: `python -m src.pnl_attribution` groups realised PnL by
side × price × p* to confirm the wedge (#1) on your own fills before trusting it.

---

## Config (max-EV defaults — all in `src/config.py`)

Every knob is env-overridable. Highlights:

```
LONGSHOT_TILT_MULT=1.0      # 0 disables the wedge haircut
MAKER_ENABLED=1             # post-only path (earns rebate)
MAKER_REBATE_RATE=0.0125    # verify vs docs.polymarket.com/changelog
SELL_PRICE_MIN=0.03         # re-admit longshot tail on SELL only
SKEW_COEF=0.15              # OTM smile bump
CARRY_FROM_FUNDING=1        # perp funding → pricing carry r
ML_OVERLAY=1   ML_WEIGHT=0.25
OBI_VETO=1     OBI_VETO_THRESHOLD=-0.60
MOMENTUM_ENABLED=1   IMPULSE_FADE_ENABLED=1
MEANREV_ENABLED=1    MEANREV_BAND=0.05   MEANREV_HALF_LIFE_SECS=14400
KELLY_ENABLED=1      KELLY_FRACTION=0.30
USE_PRICE_TO_BEAT=1  CHAINLINK_BASIS_ADJ=1
REBALANCE_ARB_ENABLED=1   BUCKET_ARB_ENABLED=1
XVENUE_ENABLED=0          # turn on to scan Kalshi (detect-and-log)
MIN_TTE_SECS=60           # trade closer to expiry than the old 180s
```

Conservative profile (to A/B against the old behavior): set
`LONGSHOT_TILT_MULT=0 MAKER_ENABLED=0 ML_OVERLAY=0 OBI_VETO=0 MEANREV_ENABLED=0
KELLY_ENABLED=0 USE_PRICE_TO_BEAT=0` → reproduces the pre-assembly taker bot.

---

## Running

```bash
python -m src.main                       # paper, max-EV, runs forever
python -m src.main --duration 60         # paper, stop after 60 min + PnL
python -m src.pnl_attribution            # confirm the wedge on fills.db
XVENUE_ENABLED=1 python -m src.main      # also scan Kalshi (logs only)
python -m src.main --live                # REAL orders (needs POLY_PRIVATE_KEY)
```

A trained directional model can be dropped in as `model_weights.json`
(`{"ofi": w, "ret_fast": w, "ret_slow": w, "bias": w}`); absent that, the
OFI-dominant defaults are used.

---

## Validate → calibrate → train (the improvement loop)

The bot now learns its priors from your own data instead of hardcoded constants.

```bash
# 1. Validate the edge on real history (drives the live overlay stack):
python -m src.backtest --days 7                 # per-market PnL ± 95% CI
python -m src.backtest --days 7 --ofi-replay --maker-fill-prob 0.5
LONGSHOT_TILT_MULT=0 python -m src.backtest --days 7   # A/B the wedge

# 2. Attribute live paper fills + audit calibration (needs resolved markets):
python -m src.pnl_attribution                   # side×price×TTE, maker, Brier

# 3. Fit the priors from your resolved fills (run nightly):
python -m src.calibrate                          # → wedge_coeffs.json, calib_coeffs.json

# 4. Train the directional model from a Binance trade tape:
python -m src.ml_train --symbol btcusdt --hours 6 --min-auc 0.55   # → model_weights.json
```

`main.py` hot-loads `wedge_coeffs.json`, `calib_coeffs.json`, and `model_weights.json`
at startup; each falls back to the paper/cold-start prior when absent. New honest
knobs: `MAKER_FILL_PROB` (paper-mode resting-order fill rate; 0.5–0.6 is realistic)
and `--ofi-replay` (activates the OFI/ML overlay in the backtest from aggTrades).

## Caveats (read before live)

- **Maker fills are now modelled, not assumed.** Set `MAKER_FILL_PROB` (e.g. 0.5)
  so paper/backtest only fill a resting order some of the time. It defaults to 1.0
  for backward-compatibility — lower it before trusting maker PnL.
- **Combo legs are flattened on orphan.** `execute_atomic` unwinds filled legs if
  a later leg fails; still validate live atomicity with small size first.
- **ML transfer is unproven on crypto seconds.** The 88% AUC is SPY/minutes.
  arXiv:2511.15960 shows ML fails on raw binary direction — hence `ML_WEIGHT`
  is small and always passes through the risk-neutral conversion.
- **Cross-venue is scan-only.** Two-venue execution needs Kalshi creds + a
  Kalshi executor; today it detects and logs.
- **Combo/arb leg risk in live mode.** Multi-leg combos fill atomically in
  paper; live needs the batched-order endpoint or a flatten-on-orphan step
  (logged as a warning today).
- **Region blocks.** Binance/Polymarket/Gamma may return HTTP 403/451 from some
  hosts (incl. CI). Run from an allowed region or a self-hosted runner.
