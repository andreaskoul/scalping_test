"""
Risk manager — enforces hard limits and provides a kill switch.

Limits checked before every order:
  1. Per-trade notional cap (redundant with signal sizing; final gate)
  2. Rolling per-minute notional cap
  3. Daily drawdown stop
  4. Binance WS staleness
  5. HALT file kill switch (touch ./HALT to stop the bot remotely)
  6. Error budget (consecutive order errors)
"""

import logging
import os
import time
from collections import deque
from pathlib import Path

from .signal import Signal

log = logging.getLogger(__name__)

HALT_FILE = Path("HALT")


class RiskManager:
    def __init__(
        self,
        max_notional_per_trade: float = 25.0,
        max_notional_per_minute: float = 200.0,
        daily_drawdown_stop: float = 500.0,
        binance_stale_secs: float = 2.0,
        poly_stale_secs: float = 5.0,
        max_errors_per_minute: int = 5,
    ):
        self.max_notional_per_trade = max_notional_per_trade
        self.max_notional_per_minute = max_notional_per_minute
        self.daily_drawdown_stop = daily_drawdown_stop
        self.binance_stale = binance_stale_secs
        self.poly_stale = poly_stale_secs
        self.max_errors = max_errors_per_minute

        self._notional_window: deque[tuple[float, float]] = deque()  # (ts, notional)
        self._daily_loss: float = 0.0
        self._day_start: float = time.time()
        self._error_window: deque[float] = deque()  # timestamps of errors
        self._halted: bool = False

    # ------------------------------------------------------------------ #

    def check(
        self,
        signal: Signal,
        binance_ts: float,
        poly_ts: float,
    ) -> tuple[bool, str]:
        """Return (allowed, reason). reason is empty string if allowed."""
        now = time.monotonic()
        wall = time.time()

        if HALT_FILE.exists():
            self._halted = True
            return False, "HALT file present"

        if self._halted:
            return False, "bot halted"

        # Staleness
        if (now - binance_ts) > self.binance_stale:
            return False, f"Binance data stale ({now - binance_ts:.2f}s)"
        if (now - poly_ts) > self.poly_stale:
            return False, f"Polymarket book stale ({now - poly_ts:.2f}s)"

        # Reset daily loss counter at midnight UTC
        if wall - self._day_start > 86400:
            self._daily_loss = 0.0
            self._day_start = wall

        if self._daily_loss >= self.daily_drawdown_stop:
            return False, f"Daily drawdown stop hit (loss={self._daily_loss:.2f})"

        # Per-trade notional
        notional = signal.price * signal.size
        if notional > self.max_notional_per_trade:
            return False, f"Per-trade notional {notional:.2f} > {self.max_notional_per_trade}"

        # Rolling per-minute notional
        cutoff = now - 60.0
        self._notional_window = deque(
            (t, n) for t, n in self._notional_window if t >= cutoff
        )
        window_total = sum(n for _, n in self._notional_window)
        if window_total + notional > self.max_notional_per_minute:
            return False, f"Per-minute notional {window_total + notional:.2f} > {self.max_notional_per_minute}"

        # Error budget
        err_cutoff = now - 60.0
        self._error_window = deque(t for t in self._error_window if t >= err_cutoff)
        if len(self._error_window) >= self.max_errors:
            return False, f"Error budget exceeded ({len(self._error_window)} errors/min)"

        return True, ""

    def record_fill(self, notional: float) -> None:
        self._notional_window.append((time.monotonic(), notional))

    def record_error(self) -> None:
        self._error_window.append(time.monotonic())
        if len(self._error_window) >= self.max_errors:
            log.warning("Error budget exceeded — check connectivity")

    def record_loss(self, amount: float) -> None:
        """amount is positive for a loss (entry price − current value)."""
        self._daily_loss += amount

    def halt(self, reason: str = "manual") -> None:
        log.warning("Risk halt: %s", reason)
        self._halted = True

    def is_halted(self) -> bool:
        return self._halted or HALT_FILE.exists()
