import numpy as np

from src.eve_fundamentals import (
    DEFAULT_FUND_FIELDS,
    augment_panel_with_fundamentals,
    fetch_fundamentals,
    _pit_series,
)
from src.eve_portfolio import Panel


def _panel(T=6, N=3, F=4):
    # Monthly dates 2020-01..2020-06
    dates = [f"2020-{m:02d}-28" for m in range(1, T + 1)]
    rng = np.random.default_rng(0)
    return Panel(dates=dates, symbols=["AAA", "BBB", "CCC"][:N],
                 signals=rng.normal(0, 1, (T, N, F)),
                 fwd_returns=rng.normal(0, 0.01, (T, N)))


def test_pit_series_respects_filing_lag():
    # One quarterly record dated 2020-01-31; with a 60-day lag it is NOT usable
    # until ~2020-04-01, so Jan/Feb/Mar panel dates must be NaN, Apr+ filled.
    recs = [{"date": "2020-01-31", "returnOnEquity": 0.25}]
    dates = [f"2020-{m:02d}-28" for m in range(1, 7)]
    s = _pit_series(recs, dates, ["returnOnEquity"], lag_days=60)
    assert np.isnan(s[0, 0]) and np.isnan(s[1, 0]) and np.isnan(s[2, 0])  # Jan-Mar
    assert s[4, 0] == 0.25 and s[5, 0] == 0.25                            # May-Jun filled


def test_pit_series_forward_fills_latest_available():
    recs = [
        {"date": "2020-01-31", "returnOnEquity": 0.10},
        {"date": "2020-03-31", "returnOnEquity": 0.20},
    ]
    dates = [f"2020-{m:02d}-28" for m in range(1, 9)]  # Jan..Aug
    s = _pit_series(recs, dates, ["returnOnEquity"], lag_days=30)
    # Q1 (Jan31)+30d usable ~Mar -> Mar..May show 0.10; Q1' (Mar31)+30d ~Apr30 -> Jun+ show 0.20
    assert s[2, 0] == 0.10           # Mar
    assert s[5, 0] == 0.20           # Jun (latest available forward-filled)


def test_augment_appends_factors_and_is_neutral_for_missing():
    panel = _panel(F=4)
    recs = {"AAA": [{"date": "2019-09-30", "returnOnEquity": 0.3, "earningsYield": 0.05}]}
    aug = augment_panel_with_fundamentals(
        panel, recs, fields=["returnOnEquity", "earningsYield"], lag_days=60
    )
    # Two fundamental factors appended.
    assert aug.signals.shape == (panel.signals.shape[0], panel.signals.shape[1], 4 + 2)
    # Original price factors untouched.
    assert np.allclose(aug.signals[:, :, :4], panel.signals)
    # Rank-normed fundamentals live in [-0.5, 0.5]; symbols w/o data -> mid-rank ~0.
    fund = aug.signals[:, :, 4:]
    assert fund.min() >= -0.5 - 1e-9 and fund.max() <= 0.5 + 1e-9
    # BBB/CCC have no records -> mid-rank (0) at every date.
    assert np.allclose(fund[:, 1, :], 0.0) and np.allclose(fund[:, 2, :], 0.0)


def test_fetch_fundamentals_caches_and_does_not_refetch(tmp_path):
    calls = {"n": 0}

    def fake_fetcher(sym):
        calls["n"] += 1
        return [{"date": "2020-03-31", "returnOnEquity": 0.1}]

    a = fetch_fundamentals(["AAA", "BBB"], fake_fetcher, tmp_path)
    assert calls["n"] == 2 and set(a) == {"AAA", "BBB"}
    # Second call hits cache -> no new fetches.
    b = fetch_fundamentals(["AAA", "BBB"], fake_fetcher, tmp_path)
    assert calls["n"] == 2 and b["AAA"] == a["AAA"]


def test_default_fields_are_known_keymetrics():
    # Guard against typos drifting from the FMP key-metrics schema.
    assert "returnOnEquity" in DEFAULT_FUND_FIELDS
    assert "earningsYield" in DEFAULT_FUND_FIELDS
    assert len(DEFAULT_FUND_FIELDS) == len(set(DEFAULT_FUND_FIELDS))
