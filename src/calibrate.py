"""
Self-calibration — fit the priors from your own resolved fills.

Turns the hardcoded 2023-paper priors into a learning loop. Reads resolved
model fills and fits:

  1. **p* recalibration** (Platt-style linear): outcome ≈ a + b·p*. Corrects a
     systematically biased pricer; written to calib_coeffs.json.
  2. **Favourite-longshot wedge**: market richness (price − p*) regressed on
     (p*, tte_hours) → intercept, β_pfair, β_tte. Written to wedge_coeffs.json.

The bot hot-loads both at startup (pricing.load_wedge_coeffs / load_calibration).
Run it as a nightly job:

    python -m src.calibrate            # reads fills.db, writes the two JSONs

Caveat: fills are a *selected* sample (we only trade where an edge appeared), so
the wedge fit is biased toward what we traded. For an unbiased wedge, fit on the
backtest's full per-minute dataset; this fill-based fit is the cheap online
version that adapts to live behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp
import numpy as np

from .pnl import _fetch_resolution
from .pnl_attribution import _load_fills
from .storage import default_db_path


async def _resolved_samples(db_path: str) -> list[dict]:
    """Per-fill (p_star, price, tte_hours, outcome) for resolved model fills."""
    fills = _load_fills(db_path)
    if not fills:
        return []
    async with aiohttp.ClientSession() as s:
        markets = list({f["market_id"] for f in fills})
        resols = await asyncio.gather(*[_fetch_resolution(s, m) for m in markets])
    resol = dict(zip(markets, resols))
    out = []
    for f in fills:
        if f.get("source", "model") != "model":
            continue
        ps = f["p_star"]
        if not (0.0 < ps < 1.0):
            continue
        ref = resol.get(f["market_id"])
        if ref is None:
            continue
        out.append({
            "p_star": ps,
            "price": f["price"],
            "tte_hours": max(0.0, f.get("tte_at_fill", 0.0)) / 3600.0,
            "outcome": 1.0 if ref >= 0.5 else 0.0,
        })
    return out


def fit_calibration(samples: list[dict]) -> dict | None:
    """OLS recalibration outcome ≈ a + b·p*. Returns coeffs + Brier scores."""
    if len(samples) < 20:
        return None
    ps = np.array([s["p_star"] for s in samples])
    y = np.array([s["outcome"] for s in samples])
    X = np.column_stack([np.ones_like(ps), ps])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    cal = np.clip(a + b * ps, 0.0, 1.0)
    return {
        "a": float(a), "b": float(b), "n": len(samples),
        "brier_before": float(np.mean((ps - y) ** 2)),
        "brier_after": float(np.mean((cal - y) ** 2)),
        "version": time.time(),
    }


def fit_wedge(samples: list[dict]) -> dict | None:
    """OLS of market richness (price − p*) on (p*, tte_hours)."""
    if len(samples) < 30:
        return None
    ps = np.array([s["p_star"] for s in samples])
    tte = np.array([s["tte_hours"] for s in samples])
    rich = np.array([s["price"] - s["p_star"] for s in samples])
    X = np.column_stack([np.ones_like(ps), ps, tte])
    (i, bp, bt), *_ = np.linalg.lstsq(X, rich, rcond=None)
    return {
        "intercept": float(i), "beta_pfair": float(bp), "beta_tte_hr": float(bt),
        "n": len(samples), "version": time.time(),
    }


def _write(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    import os
    os.replace(tmp, path)


async def calibrate(db_path: str = default_db_path("fills.db")) -> None:
    samples = await _resolved_samples(db_path)
    print(f"Resolved model fills: {len(samples)}")
    if not samples:
        print("Nothing to calibrate yet.")
        return

    cal = fit_calibration(samples)
    if cal:
        _write(cal, "calib_coeffs.json")
        print(f"calib_coeffs.json: a={cal['a']:+.4f} b={cal['b']:+.4f}  "
              f"Brier {cal['brier_before']:.4f} → {cal['brier_after']:.4f} (n={cal['n']})")
    else:
        print("Too few samples for a calibration fit (need ≥20).")

    wedge = fit_wedge(samples)
    if wedge:
        _write(wedge, "wedge_coeffs.json")
        print(f"wedge_coeffs.json: intercept={wedge['intercept']:+.4f} "
              f"β_pfair={wedge['beta_pfair']:+.4f} β_tte={wedge['beta_tte_hr']:+.5f} "
              f"(n={wedge['n']})")
    else:
        print("Too few samples for a wedge fit (need ≥30).")


def cli() -> None:
    ap = argparse.ArgumentParser(description="Fit wedge + calibration from resolved fills")
    ap.add_argument("--db", default=default_db_path("fills.db"))
    args = ap.parse_args()
    asyncio.run(calibrate(args.db))


if __name__ == "__main__":
    cli()
