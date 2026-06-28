"""Point-in-time S&P 500 membership — the survivorship-bias correctness lane.

Our price lake holds only the *current* index members, so any backtest over it
trades a forward-looking universe: names that were later dropped (acquisitions,
bankruptcies, relegations — disproportionately losers) are silently excluded,
inflating both gross alpha and the passive benchmark. This module reconstructs
*who was actually in the index on each date* from FMP's historical change log, so
we can (a) size the survivorship hole and (b) mask a panel to its PIT members.

Reconstruction walks **backward** from today's constituents: a change event at
date D (``symbol`` added, ``removedTicker`` removed) is undone for every date
strictly before D — the added name was *not* a member before D, the removed name
*was*. numpy-free, pure stdlib; FMP JSON is read from the cached reference dir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SP500History:
    """Current members + dated add/remove change events (newest-undoable first)."""

    current: set[str]
    changes: list[dict]  # each: {date, symbol(added), removedTicker, ...}

    def members_at(self, date: str) -> set[str]:
        """Index membership as of ``date`` (YYYY-MM-DD), reconstructed from today."""
        m = set(self.current)
        for c in sorted(self.changes, key=lambda c: c["date"], reverse=True):
            if c["date"] <= date:
                break
            added, removed = c.get("symbol"), c.get("removedTicker")
            if added:
                m.discard(added)   # added after `date` -> not yet a member
            if removed:
                m.add(removed)     # removed after `date` -> still a member then
        return m

    def universe_over(self, dates: Sequence[str]) -> set[str]:
        """All names that were members on *any* of the given dates (the true set)."""
        ever: set[str] = set()
        for d in dates:
            ever |= self.members_at(d)
        return ever


def load_sp500_history(reference_dir: str | Path) -> SP500History:
    """Load cached FMP ``sp500_current.json`` + ``sp500_changes.json``."""
    ref = Path(reference_dir)
    current = json.loads((ref / "sp500_current.json").read_text())
    changes = json.loads((ref / "sp500_changes.json").read_text())
    cur = {r["symbol"].upper() for r in current if r.get("symbol")}
    chg = [c for c in changes if c.get("date")]
    for c in chg:  # normalize tickers
        if c.get("symbol"):
            c["symbol"] = c["symbol"].upper()
        if c.get("removedTicker"):
            c["removedTicker"] = c["removedTicker"].upper()
    return SP500History(current=cur, changes=chg)


@dataclass(frozen=True)
class SurvivorshipReport:
    true_universe: int      # distinct names ever in the index over the window
    have_prices: int        # of those, how many we hold price history for
    missing: int            # members we have NO price for -> the survivorship hole
    missing_symbols: list[str]

    @property
    def hole_frac(self) -> float:
        return self.missing / self.true_universe if self.true_universe else 0.0


def survivorship_gap(
    history: SP500History, owned_symbols: Iterable[str], dates: Sequence[str]
) -> SurvivorshipReport:
    """Quantify the survivorship hole: true PIT universe vs the symbols we hold."""
    owned = {s.upper() for s in owned_symbols}
    ever = history.universe_over(dates)
    missing = sorted(ever - owned)
    return SurvivorshipReport(
        true_universe=len(ever), have_prices=len(ever & owned),
        missing=len(missing), missing_symbols=missing,
    )


def membership_mask(
    history: SP500History, symbols: Sequence[str], dates: Sequence[str]
):
    """Boolean ``(T, N)`` mask: True where ``symbols[j]`` was a PIT member at ``dates[t]``.

    Lets a panel restrict each date's tradable set to genuine members (so a name is
    not traded before it joined or after it left). Does *not* recover dropped names
    we lack prices for — that needs delisted-price ingest; this only removes
    entry/exit-timing look-ahead within the names we do hold.
    """
    import numpy as np

    syms = [s.upper() for s in symbols]
    mask = np.zeros((len(dates), len(syms)), dtype=bool)
    for t, d in enumerate(dates):
        members = history.members_at(d)
        for j, s in enumerate(syms):
            mask[t, j] = s in members
    return mask
