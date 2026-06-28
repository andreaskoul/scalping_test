import numpy as np

from src.eve_universe import (
    SP500History,
    membership_mask,
    survivorship_gap,
)


def _history():
    # Today: {AAA, BBB, CCC}. Changes:
    #  - 2020-06: DDD added, BBB removed  -> before this, members were {AAA, BBB, DDD-? no}
    #  - 2018-03: AAA added, EEE removed
    current = {"AAA", "BBB", "CCC"}
    changes = [
        {"date": "2020-06-15", "symbol": "DDD", "removedTicker": "BBB"},
        {"date": "2018-03-10", "symbol": "AAA", "removedTicker": "EEE"},
    ]
    # NB current has BBB but the 2020 event removed BBB and added DDD; the change log
    # here is a toy — we only test the backward-undo mechanics, not real consistency.
    return SP500History(current=current, changes=changes)


def test_members_at_walks_backward():
    h = _history()
    # After all changes (today)
    assert h.members_at("2026-01-01") == {"AAA", "BBB", "CCC"}
    # Just before the 2020-06 change: undo it -> remove DDD(not present), add BBB(already)
    before_2020 = h.members_at("2020-01-01")
    assert "DDD" not in before_2020  # DDD only added in 2020-06
    # Just before 2018-03: undo both -> AAA not yet a member, EEE was a member
    before_2018 = h.members_at("2017-01-01")
    assert "AAA" not in before_2018
    assert "EEE" in before_2018


def test_universe_over_is_union_including_dropped():
    h = _history()
    dates = ["2017-01-01", "2019-01-01", "2026-01-01"]
    ever = h.universe_over(dates)
    # EEE was a member back in 2017 even though it's gone today -> must appear.
    assert "EEE" in ever
    assert {"AAA", "BBB", "CCC"} <= ever


def test_survivorship_gap_counts_missing():
    h = _history()
    dates = ["2017-01-01", "2026-01-01"]
    # We only hold prices for today's names -> EEE is the survivorship hole.
    rep = survivorship_gap(h, owned_symbols={"AAA", "BBB", "CCC"}, dates=dates)
    assert "EEE" in rep.missing_symbols
    assert rep.missing >= 1
    assert 0.0 < rep.hole_frac < 1.0


def test_membership_mask_shape_and_values():
    h = _history()
    dates = ["2017-01-01", "2026-01-01"]
    symbols = ["AAA", "EEE", "CCC"]
    mask = membership_mask(h, symbols, dates)
    assert mask.shape == (2, 3)
    # 2017: AAA not yet member, EEE member, CCC member
    assert not mask[0, 0] and mask[0, 1] and mask[0, 2]
    # 2026: AAA member, EEE gone, CCC member
    assert mask[1, 0] and not mask[1, 1] and mask[1, 2]
