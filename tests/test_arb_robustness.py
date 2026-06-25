import time

from src.arbitrage import find_strike_arbs
from src.arb_replay import ArbEpisode, detect_strike_episodes, pair_coverage, summarize
from src.poly_universe import PolyMarket
from src.poly_ws import BookSnapshot
from src.replay import ReplayEvent

_NOW = time.time()


def _market(strike, token_id, symbol="btcusdt", expiry_offset=1800.0):
    return PolyMarket(
        condition_id=f"c{strike}", question=f"BTC above {strike}",
        yes_token_id=token_id, no_token_id=f"no-{token_id}",
        yes_price=0.5, no_price=0.5, strike=strike,
        expiry_ts=_NOW + expiry_offset, tick_size=0.01,
        symbol=symbol, is_threshold=True, fee_rate=0.07, fee_exponent=1.0,
    )


class _WS:
    """Books keyed by token -> (bid, ask, bid_sz, ask_sz, ts)."""
    def __init__(self, books):
        self._b = books

    def snapshot(self, tid):
        b = self._b.get(tid)
        if b is None:
            return None
        return BookSnapshot(tid, b[0], b[1], b[2], b[3], ts=b[4])


def _violation_books(ts_low, ts_high):
    # ask_low=0.50, bid_high=0.55 → a real monotonicity violation
    return _WS({
        "tl": (0.48, 0.50, 100, 100, ts_low),
        "th": (0.55, 0.57, 100, 100, ts_high),
    })


def _markets():
    m_low, m_high = _market(78000, "tl"), _market(78500, "th")
    m_high.expiry_ts = m_low.expiry_ts
    return [m_low, m_high]


# ---------- freshness gate ----------

def test_fresh_books_fire():
    ws = _violation_books(1000.0, 1000.0)
    arbs = find_strike_arbs(_markets(), ws, max_book_age_secs=1.5, clock=lambda: 1000.5)
    assert len(arbs) == 1


def test_stale_books_suppressed():
    ws = _violation_books(1000.0, 1000.0)
    arbs = find_strike_arbs(_markets(), ws, max_book_age_secs=1.5, clock=lambda: 1010.0)
    assert arbs == []


def test_pair_skew_suppresses_crossbook_phantom():
    # Both fresh vs the clock, but the two legs are 0.5s apart (> 0.30 skew):
    # exactly the stale-cross pattern that manufactures phantom arbs.
    ws = _violation_books(1000.0, 1000.5)
    arbs = find_strike_arbs(_markets(), ws, max_book_age_secs=2.0, clock=lambda: 1000.6)
    assert arbs == []


def test_near_simultaneous_legs_fire():
    ws = _violation_books(1000.0, 1000.1)  # 0.1s apart < 0.30 skew
    arbs = find_strike_arbs(_markets(), ws, max_book_age_secs=2.0, clock=lambda: 1000.2)
    assert len(arbs) == 1


def test_replay_disables_freshness_with_inf():
    # Ancient timestamps but inf max age → detection still works (offline replay).
    ws = _violation_books(1.0, 1.0)
    arbs = find_strike_arbs(_markets(), ws, max_book_age_secs=float("inf"), clock=lambda: 1e9)
    assert len(arbs) == 1


# ---------- arb_replay validator ----------

def _ev(ts, token, bid, ask):
    return ReplayEvent(
        ts_wall=ts, ts_mono=ts, token_id=token, etype="book",
        snapshot=BookSnapshot(token, bid, ask, 100.0, 100.0, ts),
    )


def test_detect_strike_episode_persistence():
    events = [
        _ev(1.0, "tl", 0.48, 0.50),
        _ev(1.0, "th", 0.43, 0.45),   # consistent: bid_high < ask_low
        _ev(2.0, "th", 0.60, 0.62),   # violation opens
        _ev(5.0, "th", 0.43, 0.45),   # violation closes
    ]
    eps = detect_strike_episodes(events, _markets(), min_credit=0.0)
    assert len(eps) == 1
    e = eps[0]
    assert (e.k_low, e.k_high) == (78000, 78500)
    assert e.start_ts == 2.0 and e.end_ts == 5.0
    assert e.persistence_secs == 3.0
    assert e.realized_floor > 0


def test_no_episode_when_monotonicity_holds():
    events = [
        _ev(1.0, "tl", 0.80, 0.82),
        _ev(2.0, "th", 0.40, 0.42),
        _ev(3.0, "th", 0.45, 0.47),
    ]
    assert detect_strike_episodes(events, _markets(), min_credit=0.0) == []


def test_pair_coverage_counts_legs_with_data():
    # Only the low leg has a book event → the pair is not evaluable.
    events = [_ev(1.0, "tl", 0.48, 0.50)]
    total, with_data = pair_coverage(events, _markets())
    assert total == 1 and with_data == 0
    events += [_ev(1.0, "th", 0.55, 0.57)]
    assert pair_coverage(events, _markets()) == (1, 1)


def test_summarize_latency_buckets():
    eps = [
        ArbEpisode("a", "btcusdt", 1.0, 1, 2, 0.0, 0.2, 1, 0.05, 10.0),   # 0.2s
        ArbEpisode("a", "btcusdt", 1.0, 1, 2, 10.0, 13.0, 3, 0.04, 10.0),  # 3.0s
    ]
    s = summarize(eps, latencies=(0.5, 1.0, 2.0))
    assert s["episodes"] == 2
    assert s["capturable"][0.5]["count"] == 1   # only the 3s one survives
    assert s["capturable"][2.0]["count"] == 1
