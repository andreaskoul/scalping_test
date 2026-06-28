"""News sentiment lane — leak-safe supervised scoring (SESTM-style).

The literature verdict (`docs/EVE_SHARPE_LITERATURE.md`, news section): a *learned*
sentiment score beats lexicons and matches/beats embeddings out-of-sample, and the
only credible post-cost monthly number (Ke-Kelly-Xiu SESTM) is ~1.5 — while
pretrained embeddings on a historical backtest invite **look-ahead bias** (the model
encodes the future). So this lane learns sentiment the leak-safe way:

* **SESTM-style supervised scoring** (`SestmSentiment`): screen words whose presence
  skews toward high- vs low-return documents, weight them by that skew, score an
  article as the mean weight of its sentiment words. No pretrained model -> no
  future knowledge baked in.
* **Expanding-window refit** (`monthly_sentiment_factor`): the score for month *t*
  uses word-weights estimated **only from articles dated before t**, so the factor
  at every date is causal — leak-safe by construction, exactly what our anchored
  walk-forward needs.

Source = Alpaca news (headlines + timestamps + symbols, free with our keys).
numpy + stdlib only.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .eve_features import _csrank_norm
from .eve_portfolio import Panel

_ALPACA_NEWS = "https://data.alpaca.markets/v1beta1/news"
_TOKEN = re.compile(r"[a-z][a-z']+")
# Minimal stopword set — function words carry no sentiment and only add noise.
_STOP = frozenset(
    "the a an and or but of to in on for with at by from as is are was were be been "
    "this that these those it its it's their there here you your we our they them he "
    "she his her not no than then so if into out up down over under after before new "
    "says said will would could should can may might has have had do does did about".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN.findall((text or "").lower()) if w not in _STOP and len(w) > 2]


# --------------------------------------------------------------------------- #
# Alpaca news ingest (cached, pull-once)
# --------------------------------------------------------------------------- #
def fetch_news(
    symbols: Sequence[str],
    start: str,
    end: str,
    *,
    key: str,
    secret: str,
    cache_dir: str | Path,
    per_symbol: bool = True,
    sleep: float = 0.0,
    refresh: bool = False,
) -> dict[str, list[dict]]:
    """Fetch Alpaca news for ``symbols`` over [start, end), caching each symbol once.

    Returns symbol -> list of ``{created_at, headline, summary, symbols}`` records.
    A symbol whose cache file exists is not re-fetched (resumable full-universe pull).
    Dates are ISO (``YYYY-MM-DD``). numpy-free; only the network edge here.
    """
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {}
    for sym in symbols:
        fp = cdir / f"{sym.upper()}.json"
        if fp.exists() and not refresh:
            out[sym.upper()] = json.loads(fp.read_text())
            continue
        arts = _fetch_symbol_news(sym, start, end, key=key, secret=secret, sleep=sleep)
        fp.write_text(json.dumps(arts))
        out[sym.upper()] = arts
    return out


def _fetch_symbol_news(sym, start, end, *, key, secret, sleep=0.0, max_pages=200):
    arts: list[dict] = []
    page_token = None
    for _ in range(max_pages):
        q = {"symbols": sym, "start": f"{start}T00:00:00Z", "end": f"{end}T00:00:00Z",
             "limit": 50, "sort": "asc"}
        if page_token:
            q["page_token"] = page_token
        req = urllib.request.Request(_ALPACA_NEWS + "?" + urllib.parse.urlencode(q))
        req.add_header("APCA-API-KEY-ID", key)
        req.add_header("APCA-API-SECRET-KEY", secret)
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.load(r)
        for a in payload.get("news", []):
            arts.append({"created_at": a.get("created_at"), "headline": a.get("headline"),
                         "summary": a.get("summary"), "symbols": a.get("symbols")})
        page_token = payload.get("next_page_token")
        if not page_token:
            break
        if sleep:
            time.sleep(sleep)
    return arts


# --------------------------------------------------------------------------- #
# SESTM-style supervised sentiment
# --------------------------------------------------------------------------- #
class SestmSentiment:
    """Supervised word-sentiment screener (a faithful, simplified Ke-Kelly-Xiu SESTM).

    Fit on tokenized documents each labeled +1 (high forward return) or -1 (low).
    For each word, ``f_j`` = fraction of its *labeled* occurrences in +1 documents;
    words with ``|f_j-0.5| >= alpha`` and at least ``min_count`` occurrences are kept
    as the sentiment set, with weight ``2*f_j-1`` in [-1, 1]. An article's score is
    the mean weight of its sentiment words (0 if it contains none).
    """

    def __init__(self, *, alpha: float = 0.06, min_count: int = 10):
        self.alpha = alpha
        self.min_count = min_count
        self.weights: dict[str, float] = {}

    def fit(self, docs: Sequence[Sequence[str]], labels: Sequence[int]) -> "SestmSentiment":
        pos: dict[str, int] = {}
        tot: dict[str, int] = {}
        for toks, y in zip(docs, labels):
            if y == 0:
                continue
            for w in set(toks):           # presence, not count (per-doc)
                tot[w] = tot.get(w, 0) + 1
                if y > 0:
                    pos[w] = pos.get(w, 0) + 1
        self.weights = {}
        for w, c in tot.items():
            if c < self.min_count:
                continue
            f = pos.get(w, 0) / c
            if abs(f - 0.5) >= self.alpha:
                self.weights[w] = 2.0 * f - 1.0
        return self

    def score(self, toks: Sequence[str]) -> float:
        vals = [self.weights[w] for w in toks if w in self.weights]
        return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
# Leak-safe monthly per-stock sentiment factor
# --------------------------------------------------------------------------- #
def _month(ts: str) -> str:
    return ts[:7]


def _collect_articles(records_by_symbol: dict[str, list[dict]], symbols: Sequence[str]):
    """-> list of (month, symbol_index, tokens) for symbols in the panel."""
    sidx = {s.upper(): j for j, s in enumerate(symbols)}
    arts = []
    for sym, recs in records_by_symbol.items():
        j = sidx.get(sym.upper())
        if j is None:
            continue
        for r in recs:
            ts = r.get("created_at")
            if not ts:
                continue
            toks = tokenize((r.get("headline") or "") + " " + (r.get("summary") or ""))
            if toks:
                arts.append((_month(ts), j, toks))
    return arts


def monthly_sentiment_factor(
    panel: Panel,
    records_by_symbol: dict[str, list[dict]],
    *,
    min_train_months: int = 12,
    alpha: float = 0.06,
    min_count: int = 10,
) -> np.ndarray:
    """Leak-safe ``(T, N)`` sentiment factor: each month scored by a dictionary fit
    only on *earlier* months' articles, labeled by that month's realized return sign.

    For month t (>= ``min_train_months``): label every article in months < t by the
    sign of its stock's panel ``fwd_returns`` that month, fit :class:`SestmSentiment`,
    then score month t's articles and average per symbol. Earlier months and
    symbol-months with no article are NaN (downstream rank-norm -> neutral mid-rank).
    """
    T, N = panel.fwd_returns.shape
    month_of_date = [d[:7] for d in panel.dates]
    pos = {m: t for t, m in enumerate(month_of_date)}  # month -> panel row
    arts = _collect_articles(records_by_symbol, panel.symbols)

    # Bucket article token-lists by month for fast slicing.
    by_month: dict[str, list[tuple[int, list[str]]]] = {}
    for m, j, toks in arts:
        by_month.setdefault(m, []).append((j, toks))

    months_sorted = sorted(set(m for m, _, _ in arts) | set(month_of_date))
    factor = np.full((T, N), np.nan)

    for t in range(min_train_months, T):
        cur_month = month_of_date[t]
        # Training docs: all articles strictly before this panel month, labeled by
        # the realized sign of their stock's fwd return that month (leak-safe: those
        # returns are known by month t).
        docs, labels = [], []
        for m, bucket in by_month.items():
            if m >= cur_month or m not in pos:
                continue
            row = pos[m]
            for j, toks in bucket:
                y = int(np.sign(panel.fwd_returns[row, j]))
                if y != 0:
                    docs.append(toks)
                    labels.append(y)
        if len(docs) < 50:
            continue
        model = SestmSentiment(alpha=alpha, min_count=min_count).fit(docs, labels)
        # Score current month's articles, average per symbol.
        sums = np.zeros(N)
        cnts = np.zeros(N)
        for j, toks in by_month.get(cur_month, []):
            sums[j] += model.score(toks)
            cnts[j] += 1.0
        with np.errstate(invalid="ignore"):
            row_vals = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), np.nan)
        factor[t] = row_vals
    return factor


def augment_panel_with_sentiment(
    panel: Panel,
    records_by_symbol: dict[str, list[dict]],
    *,
    rank_normalize: bool = True,
    **factor_kwargs,
) -> Panel:
    """Append the leak-safe monthly sentiment factor as one extra feature."""
    fac = monthly_sentiment_factor(panel, records_by_symbol, **factor_kwargs)
    fac = _csrank_norm(fac) if rank_normalize else np.nan_to_num(fac, nan=0.0)
    signals = np.concatenate([panel.signals, fac[:, :, None]], axis=2)
    return Panel(dates=panel.dates, symbols=panel.symbols,
                 signals=signals, fwd_returns=panel.fwd_returns)
