# Highest-Sharpe Models — Literature & GitHub Survey (critically assessed)

_As-of 2026-06-27. Compiled for Eve. Read this with the repo's validation ethos
(see `docs/VALIDATION_PLAN.md`, `src/stats.py` deflated Sharpe): **a reported
Sharpe is a hypothesis, not a fact.** Almost every headline Sharpe below is
in-sample, single-regime, pre-cost, or un-deflated. We rank by how much the
number survives scrutiny, not by how big it is._

## 0. The one-paragraph truth

The biggest *reported* Sharpes (2.5–3.7) come from RL / portfolio-optimization
papers on **daily, cross-sectional** data and rarely survive transaction costs,
multiple-testing correction, or a regime change. The best *independently
benchmarked* models (Microsoft qlib, 20-seed mean±std on CSI300) top out around
**Information Ratio ≈ 1.3** — and there **gradient-boosted trees match or beat
transformers/LSTMs.** In the **intraday LOB** regime closest to Adam/Eve, the
deep-learning literature reports high classification F1 that **collapses after
costs and out-of-sample** (LOBCAST) — which is exactly what our own transformer
just reproduced (negative post-cost edge on BTC 1-min and EUR/USD 1h).

## 1. Highest *reported* Sharpe ratios (with the asterisks)

| Reported SR | Source | Setting | Why to distrust it |
|---|---|---|---|
| **3.66** (ann., "after costs") | ML portfolio-weight optimization (AFA working paper) | Daily long–short equity | Cross-sectional, selection over models; no deflated-SR / PBO reported |
| **3.21 / 2.47** | "Innovative Reward Functions in RL" (MDPI Mathematics 14/5/794) | Crypto, 2022 test window | Single bearish regime; RL reward can game variance; one split |
| **~2** | "Investment sizing with DL prediction uncertainties" (arXiv:2007.15982), Eurodollar futures | Intraday futures, costs included | Lim/Zohren/Roberts lineage — among the more honest; still one market, one era |
| **0.86 ± 0.28** | Memory-augmented SAC (ScienceDirect 2024) | Multi-asset 2014–2024, path-dependent costs | The *honest* number: wide CI, modest SR once costs are path-dependent |

Takeaway: the credible, cost-aware, multi-period studies land **below ~1**, not
at 3. The 3+ numbers are the ones that most need a deflated-Sharpe / PBO check.

## 2. The independently-benchmarked reality — Microsoft qlib

qlib publishes a 20-seed mean±std leaderboard on CSI300 (China A-shares, daily).
Information Ratio ≈ Sharpe for these long–short factor strategies:

| Model (dataset) | Information Ratio | Ann. Return |
|---|---|---|
| HIST (Alpha360) | **1.37 ± 0.27** | 9.9% |
| IGMTF (Alpha360) | 1.35 ± 0.25 | 9.5% |
| DoubleEnsemble (Alpha158) | **1.34 ± 0.11** | 11.6% |
| TRA (Alpha360) | 1.28 ± 0.42 | 9.2% |
| MLP (Alpha158) | 1.14 ± 0.23 | 9.0% |
| **LightGBM** (Alpha158) | 1.02 | 9.0% |
| Transformer / LSTM / ALSTM / GRU | _included, **not** at the top_ | — |

Two uncomfortable facts for a "transformer will win" thesis:
1. **The ceiling is ~1.3 IR**, on broad daily cross-sections, not 3+.
2. **Gradient-boosted trees (LightGBM, DoubleEnsemble) match or beat the
   transformer/LSTM** on tabular factor data. This is the single most replicated
   result in the applied literature and the reason qlib's own defaults are GBDT.

## 3. The intraday / LOB regime (closest to Adam & Eve)

- **DeepLOB** (arXiv:1808.03668), **TLOB** (arXiv:2502.15757): strong mid-price
  trend-prediction **F1** on FI-2010. TLOB's *own* paper flags degradation once
  spread / transaction-cost proxies enter the labels.
- **LOBCAST benchmark** (arXiv:2308.01915): the decisive caveat — reported LOB-DL
  accuracy **drops significantly on new data**; profit analysis is far weaker
  than the F1 suggests. Robustness/generalization, not architecture, is the wall.
- **Our corroboration (this repo, today):** a compact encoder transformer trained
  on cost-aware labels gives **negative post-cost expectancy** on real BTC/USD
  1-min (−0.000185/event) and EUR/USD 1h (−0.000107/event) — it does **not** beat
  the no-trade baseline, despite *better calibration* (Brier 0.65 vs logistic
  1.04). Calibration ≠ profit; OHLCV bars at these costs carry no edge.

## 4. Most-starred GitHub repos — LSTM / Transformer in stocks & finance

Authoritative star counts (GitHub API, 2026-06-27). **Stars measure popularity,
not profitability** — the most-starred *pure* stock-prediction repos are
tutorial-grade and predict price *levels*, not cost-aware tradable edge.

| Stars | Repo | What it is | Validation honesty |
|---|---|---|---|
| **45.2k** | [microsoft/qlib](https://github.com/microsoft/qlib) | Quant platform + model zoo (LSTM/GRU/ALSTM/Transformer/TFT/TRA) + benchmark | **Best.** 20-seed leaderboard, walk-forward; IR~1.3 ceiling |
| 20.7k | [AI4Finance/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | Financial LLMs | Not price-prediction; sentiment/NLP |
| 15.5k | [AI4Finance/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Deep-RL trading framework | Sharpe varies wildly by env; backtest-overfit prone |
| 13.7k | [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | LLM-driven factor/model R&D (qlib companion) | Rigorous when paired with qlib |
| 12.5k | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | Transformer zoo: PatchTST, iTransformer, TimesNet, FEDformer | General TS; clean reference impls |
| **9.4k** | [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) | Most-starred *pure* LSTM/DL stock repo + trading-bot sims | Tutorial-grade; **no rigorous OOS/cost validation** |
| 6.5k | [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020) | Informer long-sequence transformer (AAAI'21 best paper) | Forecasting accuracy, not trading PnL |
| 4.3k | [AI4Finance/ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) | Massively-parallel DRL (trading-applicable) | Library, not a validated strategy |
| 3.4k | [AI4Finance/FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading) | RL trading infra | Env-dependent |
| 2.6k | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) | Patching transformer (Eve's design ref) | SOTA forecasting MSE; not a trading claim |
| 2.5k | [thuml/Autoformer](https://github.com/thuml/Autoformer) | Decomposition transformer | Forecasting benchmark |
| 0.16k | [LeonardoBerti00/TLOB](https://github.com/LeonardoBerti00/TLOB) | The LOB dual-attention transformer (our exact domain) | New; F1-focused; honest about costs |

## 5. What this means for Eve (recommendations)

1. **Stop expecting an OHLCV-bar transformer to clear costs.** Our result +
   LOBCAST + TLOB's own caveat all agree. The bar (`eve_baselines`) is doing its
   job by saying "no edge."
2. **Where edge actually shows up in the validated literature:** *breadth*
   (qlib-style daily cross-section over many names, where IR~1.3 is real) — not
   single-asset intraday. The lever is more *symbols/features*, not a fancier
   sequence model.
3. **For intraday, the missing ingredient is microstructure** (LOB depth, queue,
   order flow — Sirignano–Cont 1803.06917, DeepLOB), which Alpaca/Yahoo **bars do
   not contain.** This is also Adam's domain; it's why Adam keeps L2 event shards.
4. **Default to GBDT as the model baseline to beat**, not a transformer — it wins
   on tabular factors in the only rigorously-benchmarked setting.
5. **Always deflate.** Whatever we adopt, run it through `src/stats.py`
   deflated-Sharpe with the true #configs tried before believing any Sharpe.

**Concrete next experiments for Eve**, in order of expected payoff:
(a) cross-sectional daily equities from Alpaca (many symbols) → qlib-style
ranking labels → GBDT vs transformer under our post-cost harness;
(b) only then richer features; (c) microstructure features require an L2 source
(Adam already has Polymarket L2; equities L2 would need a venue feed).
