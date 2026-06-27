# Adam & Eve — Goals, Literature, and Status

_Analytical summary. Last updated 2026-06-27 (Eve **ensemble** built — transformer
prediction → direct-RL portfolio allocation — and validated on real equities:
the allocator works, the predictor has no edge, so the ensemble ≈ beta. Adam at
280 captured / 23 resolved. 248 tests passing.)_

_See also `docs/EVE_SHARPE_LITERATURE.md` (highest-Sharpe survey),
`docs/EVE_ENSEMBLE.md` (transformer+RL ensemble), and **`docs/EVE_ALPHA_RESULTS.md`**
(the qlib recipe A–E on 472 S&P names: GBDT has real **gross** alpha — Sharpe 0.83,
PBO≈0 — but it's short-horizon and **nets negative at retail costs**; break-even
≈2 bps/side. The wall is execution cost, not model quality; a deflated post-cost
Sharpe of 4 is not attainable honestly)._

This repo is becoming **two sibling trading engines behind one validation spine**:

- **Adam** — a constrained specialist: a latency/execution-arbitrage scalper on
  Polymarket BTC/ETH window & threshold markets, priced off Binance.
- **Eve** — a broader opportunity engine that learns from historical multi-venue
  data (Alpaca first) and will evaluate wider markets later.
- **Verifier** — the shared promotion layer that decides, machine-readably,
  whether either engine may paper / canary / live trade.

**Hard rule.** Adam may reach live before Eve, but *only through the Verifier*.
Eve places no live orders until it has cleared historical → replay → advisory →
paper → canary, in that order.

---

## 1. Why the architecture looks like this (the validation thesis)

For a latency/execution scalper, **a historical backtest is a weak validator**:
public Polymarket history has no L2 depth, queue position, or event timing, so a
backtest can prove a *pricing* hypothesis but never an *execution* edge. The
design therefore separates three truth tiers by what each can actually prove
(see `docs/VALIDATION_PLAN.md`):

```
Backtest        → hypothesis filter   (cheap, biased; pricing edges only)
Paper + replay  → DECISION truth        (state, eligibility, edge persistence,
                                         counterfactual fills)
Live canary     → EXECUTION truth       (fill rate, queue, adverse selection)
```

Everything below is built to move a strategy *up* those tiers under explicit,
statistical gates — never on raw backtest means.

---

## 2. Literature & tooling (researched, with how it's applied)

| Area | Key sources | How it's used here |
|---|---|---|
| **Execution realism** | `nkaz001/hftbacktest` (queue models: `PowerProbQueueFunc`, `RiskAverseQueueModel`); `mileswangs/pm-hftbacktest` (Rust, design ref); `txbabaxyz/polyrec` (BTC UP/DOWN domain ref); "Market Maker's Dilemma" (arXiv:2502.18625) | `src/makerfill.py` queue-aware `P(fill before GTD)` with conservatism exponent n; fill is a **label, not an optimism switch**. |
| **Validation statistics** | Bailey & López de Prado — **Deflated Sharpe Ratio**, PBO/CSCV; Politis–Romano stationary/block bootstrap; Newey–West HAC; purged/embargoed CV | `src/stats.py`: block-bootstrap CIs, HAC SE, t-stat, Deflated SR. Sweep edges are discounted by **#configs tried**; significance is per-event, not raw mean. |
| **Polymarket microstructure & fees** | Polymarket docs ("Prices & Orderbook", unified book + split/merge); 2026 fee changelog | Cost model: **taker fees live since Jan 2026**; maker rebate is **pro-rata daily (~20% crypto), not a guaranteed per-share credit**; fee curve peaks at p=0.5. The verdict uses fee-inclusive **taker** edge as the conservative floor. |
| **Combinatorial arbitrage** | Saguillo et al. 2025 (arXiv:2508.03474): market-rebalancing vs combinatorial arb | `src/arbitrage.py` strike/bucket scanners; rebalance is **structurally impossible** on Polymarket's unified book and is disabled (see §5). |
| **Cost-aware labels & baselines (Eve, built)** | López de Prado triple-barrier / meta-labeling (*Advances in Financial ML*, 2018); FI-2010 ternary labels (Ntakaris et al., arXiv:1705.03233) + LOBCAST critique; bitcoin walk-forward under costs (arXiv:2606.00060) | `src/eve_labels.py`: the no-trade boundary **is** the round-trip cost (cost-aware execution filter); `src/eve_baselines.py`: required baseline set + anchored walk-forward scored on **post-cost expectancy** (not accuracy), reusing `stats.py` for CI/t/DSR. |
| **Transformer perception (Eve, pass 2)** | TLOB (Berti & Kasneci 2025, arXiv:2502.15757, dual attention); DeepLOB (1808.03668); PatchTST (2211.14730, channel-independent patching); `thuml/Time-Series-Library`; LOBCAST benchmark (2308.01915) | Eve's planned model lane. **Caveat from the literature itself:** reported LOB F1 (70–90%) collapses under spread/cost-aware labels and generalizes poorly out-of-distribution → transformer must beat the baselines above *after costs*. |
| **Research workflow** | `microsoft/qlib` patterns | Manifests, walk-forward, experiment versioning for Eve's lake. |
| **Eve historical data** | Alpaca Market Data API (`data.alpaca.markets`): `/v2/stocks/bars` (feed=iex free) + `/v1beta3/crypto/{loc}/bars`; ≤10k/page, `next_page_token` pagination, RFC-3339 bars `t/o/h/l/c/v/n/vw` | `src/eve_ingest.py`: cached, paginating, rate-limited downloader → Eve lake. **Two idempotency layers** (lake partition skip + persistent per-request response cache) so the same API call is never re-run; injectable transport (tests need no creds). Historical-only; Alpaca is never a live feed or execution venue. |
| **RL/policy (later)** | `TradeMaster`, `FinRL` | Only after supervised, calibrated Eve is stable. |

---

## 3. Adam — status

**Mature / proven (built and tested):**

- **Pricing & signal engine** (`src/signal.py`, `pricing.py`): option-implied
  terminal probability with σ blend, self-calibration, favourite–longshot wedge,
  microstructure/ML nudge, mean-reversion overlay, fractional Kelly, maker/taker
  edge legs.
- **Decision telemetry & capture** (`telemetry.py`, `poly_ws.py` event sink):
  per-candidate `DecisionTrace`, reject taxonomy, raw L2 event shards.
- **Replay v1 — counterfactual evaluator** (`src/replay.py`): refetches Binance,
  merges with recorded poly books, drives the **real** `SignalGenerator`, scores
  taker/maker counterfactual fills, time-split OOS sweep + walk-forward.
- **Maker-fill model** (`src/makerfill.py`): queue-ahead, `power_prob_fill`,
  expected (uncertain) rebate.
- **Shared stats** (`src/stats.py`) and **outcome/edge labels** (`backfill.py`:
  edge@5s/30s, executable) and **calibration dashboards** (`decision_analysis.py`).
- **Promotion verdict** (`src/adam_report.py`): machine-readable
  `adam: paper_ok=… canary_ok=… live_ok=…` over an Adam-scoped gate subset
  (schema, leakage/time-split, replay-parity, cost/slippage, calibration,
  edge-persistence, statistical-significance).
- **Execution-health hardening** (validated **live**): no-book counted once per
  empty token per heartbeat (32k→~15/heartbeat), jittered WS backoff +
  reconnect/subscription counters, Gamma `Retry-After`/exponential backoff,
  per-run telemetry cap, fills-DB reconcile fix.
- **Arbitrage robustness** (§5): phantom rebalance disabled, per-`arb_id`
  cooldown, freshness/pair-skew gates, and a replay validator (`arb_replay.py`).
- **Canary scaffold** (`canary.py`): shadow probe planner; **no live orders**.

**Honest empirical findings (this is the point of the machinery):**

- **Replay OOS sweep** (captured 24h): taker edge **negative** after costs; maker
  edge positive in-sample (val t≈2.1) but **Deflated Sharpe ≈ 0.18** — *not*
  significant once corrected for trying 5 configs.
- **Live verdict** (n=19 resolved model signals): edge **+0.114/event but
  t=1.23 (<2)**, CI95 spans 0, **edge@5s ≈ 0** (decays within reaction lag),
  Brier **0.13 vs 0.25** no-skill (well-calibrated). → `paper_ok=False`.
- **Net read:** Adam's **pricing is calibrated**, but there is **no
  statistically significant post-cost directional edge yet**, and the edge
  largely vanishes within 5 s.

**Verdict refresh (2026-06-27, paper capture restarted):**

- Capture holds **280 model signals**; **23 are on resolved markets** (climbing
  as markets settle: 20→23 within the session). `adam_report` on a snapshot:
  taker fee-incl edge **+0.090/event but t=1.08 (<2)**, CI95 spans 0
  [−0.124, +0.273], Brier **0.133 vs 0.246** no-skill (well-calibrated).
  `paper_ok=False`, blocked by n<30, edge t<2, edge@5s. **Same character as the
  n=19 read at session start: calibrated pricing, no significant post-cost edge
  yet.** Resolution is wall-clock-bound — the capture keeps running.

**Pending for Adam:**

- Accumulate **≥30 resolved model signals** (20 resolved of 280 captured) for a
  full-power verdict; backfill resolution as markets settle.
- **Live canary**: tiny real post-only probes → the only source of true fill
  rate / adverse selection. Gated behind a separate approval.
- Wire Adam behind the shared `Engine`/`Verifier` interface (pass 2).

---

## 4. Eve — status

**Built (substrate + evaluation bar):**

- **Research contract** (`docs/EVE_TRANSFORMER_RESEARCH.md`): scope, Alpaca =
  historical-only, label design, MPS device policy, training gates, promotion
  path (Stage 0 offline → Stage 5 production).
- **Lake substrate** (`src/eve_data.py`): `EveLake` partitioning, `AlpacaBar`
  normalization, leakage-safe `build_bar_sequences`, `chronological_split`,
  manifests, `select_torch_device` (mps→cuda→cpu). **Pure helpers run without
  alpaca-py or torch** (tests don't need credentials or a GPU).
- **Cost-aware labels** (`src/eve_labels.py`): `CostModel` (per-side fee /
  half-spread / slippage → round-trip), and the production-adjacent label where
  **the no-trade boundary IS the round-trip cost** — a directional action is
  labeled only when its *post-cost* edge is positive (de Prado triple-barrier /
  cost-aware execution filter). `post_cost_pnl` charges the same cost, so a
  correctly-labeled trade is positive-EV by construction.
- **Baselines + post-cost walk-forward** (`src/eve_baselines.py`): the required
  baseline set (no-trade, persistence/last-value, logistic, momentum-rule
  stand-in for the rule stack), an **anchored walk-forward** (train past → test
  next block → roll), and scoring on **post-cost expectancy** — never raw
  accuracy — reusing `stats.py` (block-bootstrap CI, HAC t, Deflated Sharpe).
  Emits a machine line `eve_baseline: best=… post_cost_edge=… t=…
  beats_notrade=…`. A baseline only "wins" if its post-cost edge beats 0 with
  CI excluding 0 and t≥2 (same discipline as Adam's verdict).
  Validated on synthetic data: an AR(1) momentum signal lets `persistence`
  clear cost (t≈20); on pure noise **every** directional baseline goes negative
  after costs and the gate blocks — the FI-2010 failure mode, reproduced.
- **Cached Alpaca ingest** (`src/eve_ingest.py`): paginating, rate-limited
  downloader for stock (`/v2/stocks/bars`) and crypto (`/v1beta3/crypto/.../
  bars`) historical bars → normalized lake partitions. **Two idempotency
  layers** so the same API call is never re-run: a lake partition-exists skip
  (no request built) and a persistent per-request response cache (`force`
  re-ingest rebuilds from disk, zero network). Transport + sleep are injectable
  so tests need no credentials. **Verified live** against the user's keys:
  BTC/USD daily bars fetched, then a re-run made 0 network calls and a forced
  re-run served 4/4 from cache.

- **Lake read-back + real-data verdict** (`eve_data.read_symbol_bars`,
  `eve_baselines.build_report_from_lake`): partitions are **timeframe-scoped**
  (`bars-1min` vs `bars-1day`, so timeframes never collide on a shared date),
  and the baseline verdict runs straight off real lake bars.
- **Yahoo Finance ingest** (`src/eve_yahoo.py`): free source for **forex**
  (Alpaca FX returns **403 "not authorized for FX data"** on the current plan —
  it's a paid add-on) and other asset classes. Same chart endpoint shape, same
  lake schema (`provider="yahoo"`), and the **same two idempotency layers**
  (partition skip + response cache) as the Alpaca path; no auth (browser
  User-Agent only). Verified live.

**Honest real-data finding — consistent across providers & asset classes:**

- **Crypto 1-min (Alpaca):** BTC/USD (164k) + ETH/USD (114k), Mar–Jun 2026.
- **Forex 1h (Yahoo):** EUR/USD, GBP/USD, USD/JPY (~12k each), 2024–2026.
- In **all five**, with a 3 bps round-trip cost, **every directional baseline is
  negative after costs** (edge ≈ −0.0002/event, t ≈ −22 to −159); **no-trade
  (flat) wins**, gate blocks (`beats_notrade=false`). This is the FI-2010/
  cost-aware result reproduced on real data, twice over: naive intraday
  directional signals don't survive costs. The transformer's bar is now concrete
  and measured: **produce positive post-cost expectancy on these exact OOS
  series.**

- **Compact transformer** (`src/eve_transformer.py`): encoder-only, regularized
  (small d_model, 2 layers, dropout + weight decay, learned positional emb),
  trained with cross-entropy on cost-aware labels. Implements the baseline
  `fit/predict/predict_proba` protocol so it drops into the **same** post-cost
  walk-forward harness (`extra_models=`) and is judged on the identical OOS
  series. torch is decoupled from numpy (features are native lists), so it runs
  on torch 2.2 + numpy 2; trains on **MPS**.

**First transformer result (honest, literature-predicted):**

- Trained on MPS, 3-fold anchored walk-forward, 3 bps cost. On **BTC/USD 1-min**
  it scores **−0.000185/event** (t=−117, cover 0.62) and on **EUR/USD 1h**
  **−0.000107/event** (t=−16): **negative after costs, does not beat no-trade.**
  Notably its **Brier is far better** (0.65 vs logistic's 1.04) — it is *better
  calibrated* but calibration ≠ profit. This is exactly the TLOB/FI-2010/LOBCAST
  caveat reproduced with our own model: OHLCV bars at this horizon/cost carry no
  post-cost edge. The lever is breadth / microstructure features, not a fancier
  sequence model (see `docs/EVE_SHARPE_LITERATURE.md`).

- **Ensemble — transformer prediction + RL portfolio optimization**
  (`src/eve_portfolio.py`, `src/eve_rl.py`, `src/eve_ensemble.py`; see
  `docs/EVE_ENSEMBLE.md`). Direct/recurrent RL allocator on the **Differential
  Sharpe Ratio** net of costs (Moody & Saffell 1998) → dollar-neutral
  leverage-capped book; chained behind the transformer's cross-sectional score;
  nested leakage-safe walk-forward; judged on **Deflated** Sharpe.
  **Real-equity results (30 large-caps, daily, 2016–2026, 5 bps):** the RL
  allocator on transparent momentum/vol signals is market-neutral Sharpe **0.62**
  (beats the momentum long-short −0.38, doesn't beat beta 0.94, DSR 0.27). The
  **transformer→RL ensemble is Sharpe ≈ 0.04** because the transformer's
  cross-sectional signal is edgeless (transformer-LS −0.67); the allocator
  correctly goes flat on a useless signal rather than losing money — **the
  allocator works; the predictor is the bottleneck.**

**Not built yet (pass 2):**

- **A prediction lane with real edge** (the actual bottleneck): more **breadth**
  (hundreds of names), a **GBDT** lane (qlib's winner; the baseline to beat), and
  genuine **microstructure** features for intraday. Architecture is ready; signal
  is missing.
- **Options** (later, by user request): do an **extended options-model
  literature review first**, then expand features, then ingest.
- **Advisory-only** predictions → paper → canary.

Eve now has a **labeling scheme, the evaluation bar a model must clear, a real
(cached) ingest path, and a first real-data verdict**, but **no model and no
live adapters yet** — by design.

---

## 5. Arbitrage sub-finding (resolved)

Live "arb" activity (104 leg-fills) was **~entirely phantom**:

- **Rebalance arbs (YES+NO<\$1) are structurally impossible** on Polymarket's
  unified order book (`ask_YES + ask_NO = 1 + spread_YES ≥ 1`); observed sub-\$1
  sums are stale cross-book snapshots. With no cooldown, **one** phantom fired
  **52× in 2 s**. → disabled by default + freshness/pair-skew gates + cooldown.
- The **genuine model-free arb (strike monotonicity)** is rare and trivial: the
  `arb_replay.py` validator found **0 in 24h** (41 evaluable pairs) and **1 in
  the live ~10h window** — worth **0.73¢/share, \$0.07 total**. Not a PnL engine
  at retail latency; the validator is the gate before building atomic execution.

---

## 6. Shared spine / Verifier — status

- **Implemented:** the statistical core (`stats.py`) and an **Adam-scoped,
  machine-readable verdict** (`adam_report.py`) covering a subset of gates.
- **Pending (pass 2):** sibling `Engine` interfaces (`MarketState`,
  `Opportunity`, `Prediction`, `TradeIntent`, `EngineDecision`, `Engine`,
  `Verifier`); the full 9-gate **dual-engine** Verifier (adds drawdown,
  category/regime stability, canary, kill-switch); and the live gate.

Target output shape:

```
adam: paper_ok=true  canary_ok=false live_ok=false
eve:  paper_ok=false canary_ok=false live_ok=false
```

---

## 7. Bottom line & next steps

- **Adam** is close to canary-readiness on *plumbing* (replay, fills, labels,
  verdict, exec-health all real and tested) but **has not yet demonstrated a
  significant post-cost edge**. The next decisive evidence is (a) ≥30 resolved
  signals and (b) a live maker canary — not more pricing work.
- **Arbitrage** is settled: phantom-dominated, real edge trivial; parked behind
  the validator.
- **Eve** now has cost-aware labels, a post-cost baseline bar, a cached ingest,
  and a **first real-data verdict** (crypto 1-min: no naive post-cost edge). The
  next step is a transformer that must beat the no-trade baseline on that exact
  series; data breadth (forex via an alt source, options later) feeds it.
- **Verifier** exists in Adam-scoped form; the dual-engine version is the spine
  to build once Eve has a model worth gating.

**Build order from here:** Adam live canary (execution truth) → full Verifier +
Engine interfaces → ~~Eve baselines~~ ✓ → ~~Eve Alpaca ingest~~ ✓ → ~~baselines
on real bars~~ ✓ (crypto + Yahoo FX) → ~~Eve transformer~~ ✓ → ~~transformer+RL
ensemble~~ ✓ (allocator works, predictor has no edge → ≈ beta) → **a prediction
lane with real edge** (breadth + GBDT + microstructure — the actual bottleneck) →
options (literature-first) → Eve advisory. Live trading only after replay +
canary + Verifier all pass.
