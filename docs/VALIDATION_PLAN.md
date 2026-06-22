# Validation & Telemetry Architecture — Implementation Plan

How this bot proves an edge before risking real money. The thesis: for a
latency/execution arbitrage scalper, **historical backtest is a weak validator**
(public Polymarket history has no L2 depth, queue, or timing), so it is
downgraded to a research filter. The primary loop becomes **paper-as-recorder +
replay**, and the *only* source of execution truth is a **tiny live canary**.

## 0. Principles (design invariants)

1. **Three truth tiers, by what each can actually prove:**

   ```
   Backtest        → hypothesis filter   (cheap, biased; pricing edges only)
   Paper + replay  → DECISION truth       (state + "what would I do", eligibility, edge persistence)
   Live canary     → EXECUTION truth      (fill rate, queue, adverse selection, latency)
   ```

   Paper is decision truth, **not** fill truth: it "fills" against a counterparty
   that doesn't exist, so it cannot tell you fill probability or adverse
   selection. Only the canary can.

2. **Record only what you cannot refetch.** Binance klines/aggTrades are public
   history → refetch on demand. Polymarket L2/timing is not historically
   available → capture it live. This is the core insight (~10× less storage).

3. **Protect the hot path.** The eval loop is event-driven and O(1)-tuned. All
   telemetry is async, bounded-queue, drop-on-full with a counter — never blocks
   or slows evaluation.

4. **Partition validation by edge type** (matrix in §9). Different edges need
   different tiers.

5. **Reuse, don't rebuild:** the F1 `FILL_COLUMNS` + PRAGMA-migration pattern,
   `pnl._fetch_resolution`, `MicroReplay`, `Settings`, the `poly_ws` "moved"
   dirty-signal, `Executor` / `OrderLifecycle`.

## 1. Architecture overview

```
                          ┌────────────── live/paper run ──────────────┐
 Binance WS ─┐            │  eval loop → SignalGenerator.evaluate(trace) │
 Poly WS ────┼─ feeds ───►│      │                │                      │
             │            │   DecisionTrace    Signal/Fill               │
             │      ┌─────▼──────▼────┐     ┌─────▼─────┐                │
 (tap poly   └─────►│  Recorder (async │     │ Executor  │                │
  events only)      │  queue + writer) │     │ (+canary) │                │
                    └───┬────────┬─────┘     └─────┬─────┘                │
                        │        │                 │                      │
                  decisions.db  poly_events/*   fills.db / canary.db      │
            ┌───────────▼────────▼─────────────────▼───────────┐
            │  Outcome backfill (resolution + book Δ@+5s/+30s)  │
            └───────────┬───────────────────────────────────────┘
        ┌───────────────▼───────────────┐     ┌──────────────────────┐
        │  src.replay (refetch Binance,  │     │  src.backtest         │
        │  rebuild poly timeline, drive  │     │  (downgraded: pricing │
        │  evaluate, sweep, OOS split)   │     │   research + guards)  │
        └────────────────────────────────┘     └──────────────────────┘
```

## 2. Data model

### 2a. `decisions` table (SQLite WAL) — one row per evaluated candidate

```
-- identity / timing
decision_id PK   ts_wall   ts_mono   run_id   stage(universe|eval|risk|exec)
-- market
market_id question token_id symbol strike expiry_ts tte is_updown is_threshold
-- inputs
spot_bid spot_ask spot_mid sigma iv carry drift ofi
poly_bid poly_ask poly_bid_sz poly_ask_sz book_age
-- model
p_fair p_star wedge edge_buy edge_sell chosen_side is_maker maker_price size
-- outcome of this decision
signal(int) reject_reason sampled(int)
-- backfilled (NULL until job runs)
resolution edge_at_5s edge_at_30s executable(int) realized_pnl
```

Rejects dominate volume → `sampled` flags Bernoulli/reservoir-sampled reject
rows; signals & fills are always logged.

### 2b. Raw Polymarket event log — `poly_events/<run_id>/<hour>.jsonl.gz`

Append-only gzip JSONL shards + `manifest.json`. **Gated on the existing
`_refresh_top → moved` signal** so we only record when top-of-book changes for a
tracked token. Top-of-book on every moved event; **full L2 snapshot every N s**
so `walk_book` sizing is replayable.

```
{ts_mono, ts_wall, token_id, etype:"book|price_change",
 best_bid, best_ask, bid_sz, ask_sz, levels?:[...]}   # levels only on snapshots
```

### 2c. `canary` table — live shadow-maker probes

```
order_id ts_post ts_event token_id side price size
filled(int) ts_fill queue_ahead_est
mark_5s mark_30s resolution adverse_bps rebate_earned
```

### 2d. Storage sizing (≈200-token universe)

- Poly events (moved-gated + 10s snapshots): ~50–150 MB/day gzip → 7d ≈ 0.5–1 GB.
- Decisions (signals+fills + ~2% reject sample): ~5–20 MB/day.
- Binance: not stored (refetched). **Total < 1.5 GB/week.** Retention + nightly compaction.

## 3. Component 1 — Decision telemetry

### 3a. Reject-reason taxonomy (exact `evaluate()` return points)

`TTE_TOO_SHORT, TTE_TOO_LONG, UPDOWN_SKIPPED, STRIKE_UNSET, SPOT_MISSING,
BOOK_STALE, PRICE_BAND_BUY, PRICE_BAND_SELL, OBI_VETO, NO_EDGE, COOLDOWN,
KELLY_ZERO, SIZE_TOO_SMALL` — plus main-loop stages `BLACKLISTED`,
`RISK_BLOCKED`, `NO_SPOT`, `NO_BOOK`, `WARMUP`.

### 3b. `src/telemetry.py`

- `@dataclass DecisionTrace` — mutable struct `evaluate` fills with the snapshot
  + per-leg edges + `reject_reason`.
- `class Recorder` — bounded `asyncio.Queue`; background writer; non-blocking
  `record()` (drop+count if full); batched commits; SQLite WAL.
- Reject sampler (`TELEMETRY_REJECT_SAMPLE=0.02`).

### 3c. `signal.py` change (opt-in, zero behavior change)

`evaluate(..., trace: DecisionTrace | None = None)` — when provided, populate at
every return point with inputs already in scope + reason; **return value
unchanged** (Signal|None), so existing tests are untouched.

### 3d. `main.py` integration

Instantiate `Recorder(run_id)`; build a `DecisionTrace` per candidate; pass to
`evaluate`; `recorder.record(trace)` (sampled for rejects); also emit
`stage=risk`/`stage=exec` rows when risk/blacklist block; register the feed tap.

**Acceptance:** funnel reconstructable; < 1% hot-path overhead (heartbeat
timing); queue-drop counter ≈ 0 under normal load.

## 4. Component 2 — Raw Polymarket capture

- Optional `event_sink` on `PolyWS._handle`; push top-of-book when
  `_refresh_top` returns `moved`; timer pushes full-L2 snapshots every N s.
- `Recorder` owns the gzip-shard writer + manifest + rotation/retention.
- No Binance capture (refetchable).

**Acceptance:** a recorded shard rebuilds the exact `BookSnapshot` timeline the
live `poly_ws.snapshot()` produced (replay-parity test).

## 5. Component 3 — Outcome backfill (`src/backfill.py`)

Nightly job over unlabeled `decisions` (reuses `pnl._fetch_resolution`):

- `resolution` ← Gamma close.
- `edge_at_5s/30s` ← recompute edge from the recorded poly timeline at t+Δ +
  refetched Binance spot at t+Δ.
- `executable` ← did the book trade through/consume the level within GTD
  (from price_change size deltas)? (estimate)
- `realized_pnl` ← per-fill vs resolution.

Produces labels: `edge_persisted`, `executable`, `won`.

## 6. Component 4 — Replay engine (`src/replay.py`)

1. Load recorded poly timeline for the window.
2. Refetch Binance klines+aggTrades (`MicroReplay`).
3. Merge into a wall-clock-ordered stream.
4. On each poly top-of-book change (+periodic): rebuild `BookSnapshot`,
   synthesize `BinanceTick` + `MicroFeatures`, call the **real**
   `SignalGenerator.evaluate(trace=...)`.
5. **Counterfactual fills:** taker = cross recorded book; maker = filled iff the
   recorded book subsequently trades through/consumes the posted level before
   GTD/window-end (estimate, flagged).
6. **Parameter sweep** over a config grid (fast — feeds cached).
7. **Train/validation split by time** (e.g. 60/40); report OOS only;
   walk-forward option.

**Outputs:** OOS edge per config, eligibility funnel, edge-persistence curve,
counterfactual fill rate.

**Acceptance:** replaying a recorded paper window reproduces that run's signals/
fills within tolerance (replay-parity); sweeps deterministic given a seed.

## 7. Component 5 — Shadow-maker canary (`src/canary.py`)

The only execution-truth source. Live, post-only, **tiny**.

- `Executor` shadow mode / `CanaryTrader`: post real $1–2 post-only orders at the
  maker price the signal would use, to **measure**, not for PnL.
- Safety: `CANARY_ENABLED` flag + `CANARY_MAX_TOTAL_EXPOSURE`, liquid-only, GTD
  auto-cancel via `OrderLifecycle.expire_maker_orders`, HALT kill-switch.
- Measure: fill rate, time-to-fill, **adverse selection** (mark @+5s/+30s and at
  resolution vs entry), rebate actually credited.

**Go-live gate:** over ≥ N (~200) probes, measured maker fill-rate ≥ threshold
**and** adverse-selection drift ≤ rebate + edge.

## 8. Component 6 — Backtest downgrade (`src/backtest.py`)

Keep for pricing-edge research at scale; stop it lying:

- `--dump-fills`: per-fill `spot, strike, tte, σ, p_raw, poly_price, resolution`.
- Fill-realism guards: (1) reject illiquid tail prints (price must persist across
  consecutive prints / treat <0.15,>0.85 as non-fillable); (2) skip the early
  window; (3) higher short-TTE σ floor so `p_raw` isn't a step function of
  `sign(spot−strike)`.
- Report **per-event** (independent) samples.

**Acceptance:** the "gross edge on 93.5% of minutes" artifact collapses to a sane
fraction; headline becomes believable (low-single-digit pp).

## 9. Validation matrix (edge → tier → metric → gate)

| Edge | Primary tier | Metric | Go gate |
|---|---|---|---|
| Favourite-longshot wedge | Backtest + `calibrate` | OOS PnL/event, t-stat | t ≥ 2 (per-event CI) |
| p\* calibration | Decisions backfill + `calibrate` | Brier, reliability gap | Brier_after < before |
| Skew / carry | Backtest | OOS PnL vs off | positive, significant |
| OFI/ML direction | Backtest `--ofi-replay` + decisions | walk-forward AUC | AUC ≥ 0.55 OOS |
| Stale-quote pickoff | Replay + canary | edge@5s; fill rate | edge@5s>0; fill≥X% |
| Maker rebate capture | **Canary only** | realized rebate, fill rate | net maker PnL > 0 |
| Adverse selection | **Canary only** | drift@30s vs entry | ≤ rebate+edge |
| Combinatorial / x-venue | Decisions + canary | observed credit, leg-fill | both legs fill, credit>0 |

## 10. Metrics & dashboards (`src/decision_analysis.py`, extends `pnl_attribution`)

Eligibility funnel (reason × stage); edge-persistence curve (edge@0/5s/30s — the
"did it vanish?" answer); calibration (p_fair vs realized — reuse Brier code);
executable rate; canary report (fill rate, adverse-selection bps, rebate).

## 11. Phasing, dependencies, effort

| Ph | Deliverable | Depends | Effort | Acceptance |
|---|---|---|---|---|
| **A** | `telemetry.py` + `DecisionTrace` + `evaluate(trace=)` + taxonomy + `main` wiring | F1 | M | funnel + <1% overhead + tests |
| **B** | Raw poly capture (`poly_ws` sink + shard writer + retention) | A | M | replay-parity on sample |
| **C** | `backfill.py` (resolution + edge@Δt + executable) | A,B | M | labels populated |
| **D** | `decision_analysis.py` (funnel, persistence, calibration) | A,C | S | dashboards render |
| **E** | `replay.py` (rebuild + refetch + evaluate + sweep + OOS) | B,C | L | replay-parity + deterministic sweep |
| **F** | Backtest downgrade (`--dump-fills` + guards + per-event) | — | S–M | 93.5% artifact gone |
| **G** | `canary.py` (shadow maker + caps + report) | A; live creds | L | go-live gate metrics |

**Run order:** A → B → (start recording) → C → D, F in parallel → E → G.
Paper recording 24–72h between D and E.

**Critical path to a trustworthy "go live?" answer:** A → B → C → E → G.

## 12. Risks & mitigations

- Telemetry slows hot path → bounded queue, drop+count, sample rejects, batched
  commit; measure overhead.
- Storage blowup → moved-gated events + snapshot cadence + retention; no Binance.
- Replay overfitting → time-split, OOS-only, walk-forward.
- Counterfactual fills are estimates → flag; canary is truth; never gate go-live
  on replay fills alone.
- Canary capital risk → tiny size, exposure cap, liquid-only, GTD auto-cancel,
  HALT.
- Clock skew → monotonic for intra-source order, wall for cross-source merge;
  record both.
- Schema drift → reuse the F1 `*_COLUMNS` + PRAGMA-migration pattern.

## 13. Go-live decision rule

Authorize tiny live **only when**: (1) replay OOS net edge > 0 under the
realistic cost model on the validation split; (2) edge-persistence median @5s > 0
(opportunity survives reaction lag); (3) canary fill-rate and adverse-selection
clear §7. Scale size only as canary stats hold across regimes.
