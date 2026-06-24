# EVE Transformer Research

## Purpose

This document defines the first implementation lane for adding an EVE transformer stack to this repo.

EVE means event value estimation: short-horizon prediction, execution filtering, calibration, and risk-aware sizing across broader markets than the current Adam bot.

The goal is not to replace the current rule, pricing, risk, telemetry, and execution modules. The goal is to add a research lane that can be trained, evaluated, and gated before any production use.

Transformer outputs begin as advisory features. Promotion to sizing or execution control requires explicit gates.

## Alpaca Scope

Alpaca is historical batch data only for the initial EVE path.

Alpaca should be used to download historical equities, crypto, options, quote, trade, bar, and snapshot data where the account plan permits it. Alpaca should not be used as Eve's live market-data feed or live execution venue in this phase.

Live inference and live trading must use venue-native realtime adapters. The Alpaca historical lake can still help pretrain and validate market-context models, but it must remain separated from live capture and execution code.

## Architecture References

### TLOB

TLOB is the strongest near-term reference for transformer modeling of limit order books. It supports LOB tensors, event-time windows, BTC/FI-2010 style datasets, and short-horizon up/stable/down labels. Its reported high F1 scores are useful research signals, but its own paper notes that performance deteriorates when labels account for spread and transaction-cost proxies.

Use TLOB for LOB tensor design, label experiments, attention diagnostics, and model baselines. Do not treat its benchmark scores as live trading evidence.

### TransLOB

TransLOB is an older but important LOB transformer reference. It helps frame temporal attention over book states and benchmark expectations. It should be treated as architecture literature, not a production dependency.

### Time-Series-Library

Time-Series-Library is the broad model zoo reference. PatchTST, iTransformer, TimeXer, FEDformer, Autoformer, Informer, and related models can guide Eve's medium-horizon forecasting experiments. Many examples assume CUDA, so MPS support needs local testing.

### Qlib

Qlib is a research-workflow reference: dataset handlers, factor-style evaluation, experiment tracking, and financial ML baselines. Use its patterns before adopting its full stack.

### TradeMaster and FinRL

TradeMaster and FinRL are reinforcement-learning and environment-design references. They are useful later for action-policy research, but supervised calibrated prediction should come first.

## Accuracy Claims Audit

Transformer papers can report 70-90 percent accuracy or F1 on limit-order-book benchmarks. For Eve, those claims are not enough.

Every cited performance claim must record:

- dataset and date range,
- horizon,
- label definition,
- metric,
- split method,
- baseline,
- transaction-cost treatment,
- code/data availability,
- known leakage risks.

An 80 percent benchmark result is not an 80 percent trade win rate. Directional accuracy ignores spread, fees, slippage, queue position, maker fill probability, adverse selection, and latency. A model only matters if it improves post-cost expectancy and risk-adjusted returns under strict walk-forward evaluation.

## Apple MPS Path

Apple Silicon MPS is useful for local transformer prototyping. It is not a substitute for validation.

Device selection should prefer MPS when available, then CUDA, then CPU:

```python
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

Training and evaluation manifests must record the device. CPU smoke tests must remain available. Any MPS-only success should be checked against CPU on a small deterministic batch before trust.

## Data Lake

Eve needs a small reproducible lake before transformer training grows.

Recommended local layout:

```text
data/lake/
  raw/
    alpaca/
    polymarket/
    kalshi/
    binance/
  normalized/
  features/eve_transformer/
  labels/eve_transformer/
  predictions/eve_transformer/
  models/eve_transformer/
  reports/eve_transformer/
```

Raw data is append-only. Normalized data is schema-versioned. Feature sets, labels, predictions, model artifacts, and reports all carry manifests with source paths, row counts, schema version, feature version, label version, split policy, and git commit where available.

SQLite-first is acceptable for the first implementation. Parquet can be added later when dependency and size pressure justify it.

## First Labels

The first labels should be simple and auditable:

- future spread-adjusted mid movement,
- future fair-value movement,
- fill-adjusted expected edge,
- adverse-selection markout,
- no-trade/buy/sell action class after fees.

Raw direction is a research baseline, not the production target. The preferred first production-adjacent label is future tradable edge after fees and slippage, with an explicit no-trade class.

Resolution labels must be leakage-safe. Related markets from the same event should not be randomly split across train and validation when event context could leak outcome information.

## Training Gates

No model advances unless these gates pass:

- schema validation,
- row-count and missingness checks,
- monotonic timestamp checks,
- no future-field leakage,
- grouped chronological splits,
- baseline comparison,
- calibration report,
- post-cost evaluation,
- category/regime breakdown,
- reproducible artifact manifest,
- CPU smoke test,
- MPS smoke test when available.

Baselines must include no-trade, last-value/no-change, logistic regression, and the current rule stack. Gradient boosted trees and small MLPs are useful if dependencies are added.

The transformer must beat simple baselines after costs, not only on raw prediction metrics.

## Promotion Path

Stage 0: offline research only.

Stage 1: write predictions to the lake, with no live consumption.

Stage 2: expose frozen calibrated predictions to `src.signal` as advisory features.

Stage 3: allow `src.sizing` to use confidence under strict caps.

Stage 4: allow strategy selection only after replay and canary evidence.

Stage 5: production control requires a separate approval document.

Rollback is a config switch. Missing, stale, or failed artifacts disable Eve and leave Adam/rule-only behavior intact.

## Repo Integration

Initial implementation should touch narrow boundaries:

- `src/storage.py`: lake path helpers and manifest helpers.
- `src/eve_data.py`: historical-only ingestion normalization and lake writing.
- `src/ml_train.py`: later transformer training entrypoint.
- `src/microstructure.py`: shared order-book/spread/imbalance features.
- `src/replay.py`: deterministic event stream reconstruction.
- `src/backtest.py`: walk-forward model evaluation.
- `src/calibrate.py`: confidence calibration.
- `src/signal.py`: advisory prediction consumption.
- `src/risk.py`: hard caps and disable rules.
- `src/sizing.py`: confidence-aware sizing only after gates.
- `src/telemetry.py`: model id, feature version, confidence, decision, and outcome logging.
- `src/decision_analysis.py`: model recommendation versus rule decision analysis.
- `src/pnl_attribution.py`: PnL by model, rule, market category, and regime.

## Minimal First Implementation

The first code slice should not train a transformer yet. It should create the substrate:

1. Add an Eve lake helper.
2. Add Alpaca historical-only normalization and manifest writing.
3. Add tests for deterministic path building, record normalization, and manifest generation.
4. Add optional dependency notes without requiring Alpaca credentials for tests.
5. Keep runtime trading behavior unchanged.

Once this is stable, the next slice can add a small PyTorch/MPS model wrapper and a dataset builder.
