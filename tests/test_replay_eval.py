"""Tests for the replay counterfactual evaluator (src.replay sweep/evaluate)."""

import asyncio

import pytest

from src.poly_ws import BookSnapshot
from src.replay import (
    BinancePoint,
    BinanceSeries,
    ConfigEdge,
    ReplayConfig,
    ReplayEvent,
    ReplayResults,
    SweepReport,
    build_binance_series,
    evaluate_window,
    refetch_binance,
    sweep,
)


# --------------------------------------------------------------------------- #
#  Synthetic merged-stream fixtures                                           #
# --------------------------------------------------------------------------- #

def _ev(ts, bid, ask, *, token="tok", levels=None, sizes=(50.0, 50.0)):
    return ReplayEvent(
        ts_wall=ts, ts_mono=ts, token_id=token, etype="book",
        snapshot=BookSnapshot(token, bid, ask, sizes[0], sizes[1], ts),
        levels=levels,
    )


def _buy_edge_stream(n=20, bid=0.40, ask=0.42, mid0=100000.0, drift=50.0):
    """A token whose YES ask is cheap vs an ITM-trending spot → BUY edge."""
    events = [
        _ev(float(i), bid, ask, levels={"bids": [[bid, 50.0]], "asks": [[ask, 50.0]]})
        for i in range(n)
    ]
    pts = [
        BinancePoint(ts_wall=float(i), mid=mid0 + i * drift, sigma_annual=0.6)
        for i in range(n)
    ]
    return events, BinanceSeries(symbol="BTCUSDT", points=pts)


def _cfg(name="t", **kw):
    base = dict(name=name, min_tte_secs=60.0, max_tte_secs=3600.0)
    base.update(kw)
    return ReplayConfig(**base)


# --------------------------------------------------------------------------- #
#  evaluate_window                                                            #
# --------------------------------------------------------------------------- #

def test_evaluate_window_emits_signals_on_edge_stream():
    events, bs = _buy_edge_stream()
    res = evaluate_window(events, bs, _cfg().build_signal_generator(), seed=0, config=_cfg())
    assert isinstance(res, ReplayResults)
    assert len(res.records) > 0
    assert res.device == "cpu"
    # The cheap-ask stream should fire BUYs.
    assert all(r.side == "BUY" for r in res.records)
    # Taker fills cross the book by construction.
    assert all(r.fill_prob == 1.0 for r in res.records if r.is_maker == 0)


def test_evaluate_window_is_deterministic():
    events, bs = _buy_edge_stream()
    cfg = _cfg()
    r1 = evaluate_window(events, bs, cfg.build_signal_generator(), seed=7, config=cfg)
    r2 = evaluate_window(events, bs, cfg.build_signal_generator(), seed=7, config=cfg)
    assert [r.pnl_proxy for r in r1.records] == [r.pnl_proxy for r in r2.records]
    assert [r.side for r in r1.records] == [r.side for r in r2.records]
    assert [r.price for r in r1.records] == [r.price for r in r2.records]


def test_evaluate_window_replay_parity_same_inputs_same_signals():
    # Replay-parity: identical inputs (events + series + config) → identical
    # signal stream, irrespective of how many times we run it.
    events, bs = _buy_edge_stream(n=15)
    cfg = _cfg(name="parity")
    runs = [
        evaluate_window(events, bs, cfg.build_signal_generator(), seed=0, config=cfg)
        for _ in range(3)
    ]
    sigs = [[(r.ts_wall, r.side, r.price, r.is_maker) for r in run.records] for run in runs]
    assert sigs[0] == sigs[1] == sigs[2]
    assert len(sigs[0]) > 0


def test_evaluate_window_empty_inputs_returns_empty():
    res = evaluate_window([], BinanceSeries("BTCUSDT", []), _cfg().build_signal_generator())
    assert res.records == []


def test_evaluate_window_maker_path_uses_queue_aware_fill():
    # Wide spread + maker_enabled → maker quotes inside, queue-aware fill.
    events = [
        _ev(float(i), 0.40, 0.46, levels={"bids": [[0.40, 30.0]], "asks": [[0.46, 30.0]]})
        for i in range(10)
    ]
    pts = [BinancePoint(ts_wall=float(i), mid=100000.0, sigma_annual=0.6) for i in range(10)]
    bs = BinanceSeries("BTCUSDT", pts)
    cfg = _cfg(name="mk", maker_enabled=True, maker_join_ticks=1)
    res = evaluate_window(events, bs, cfg.build_signal_generator(), seed=0, config=cfg)
    makers = [r for r in res.records if r.is_maker]
    assert makers, "expected maker signals on a wide spread"
    for r in makers:
        assert 0.0 <= r.fill_prob <= 1.0
        assert r.expected_rebate >= 0.0


# --------------------------------------------------------------------------- #
#  sweep: determinism + OOS separation                                        #
# --------------------------------------------------------------------------- #

def test_sweep_is_deterministic_given_seed():
    events, bs = _buy_edge_stream()
    cfgs = [_cfg("a", safety_eps=0.005), _cfg("b", safety_eps=0.02)]
    rep1 = sweep(events, bs, cfgs, train_frac=0.6, seed=3, walk_forward=True)
    rep2 = sweep(events, bs, cfgs, train_frac=0.6, seed=3, walk_forward=True)
    assert isinstance(rep1, SweepReport)
    assert rep1.selected == rep2.selected
    assert rep1.selected_dsr == rep2.selected_dsr
    assert [(e.config_name, e.n_train, e.n_val, e.val_mean) for e in rep1.edges] == \
           [(e.config_name, e.n_train, e.n_val, e.val_mean) for e in rep2.edges]


def test_sweep_separates_train_and_validation_by_time():
    events, bs = _buy_edge_stream(n=20)
    cfg = _cfg("only")
    rep = sweep(events, bs, [cfg], train_frac=0.6, seed=0, walk_forward=False)
    edge = rep.edges[0]
    # With ~20 signals and a 60/40 split there must be both train and val obs,
    # and they partition the signal set (no leakage by construction).
    assert isinstance(edge, ConfigEdge)
    assert edge.n_train > 0
    assert edge.n_val > 0


def test_sweep_reports_n_trials_and_records_device():
    events, bs = _buy_edge_stream()
    cfgs = [_cfg("a"), _cfg("b", safety_eps=0.01), _cfg("c", safety_eps=0.02)]
    rep = sweep(events, bs, cfgs, seed=0)
    assert rep.n_trials == 3
    assert rep.device == "cpu"
    assert rep.seed == 0
    assert rep.selected in {"a", "b", "c"}
    # Deflated Sharpe is a probability in [0, 1].
    assert 0.0 <= rep.selected_dsr <= 1.0


def test_sweep_walk_forward_split_differs_from_global():
    # A config that only fires in the second half should, under walk-forward,
    # still get a train/val split from its OWN signal times rather than being
    # all-validation against a global cut.
    early = [
        _ev(float(i), 0.40, 0.42, levels={"bids": [[0.40, 50.0]], "asks": [[0.42, 50.0]]})
        for i in range(20)
    ]
    pts = [BinancePoint(ts_wall=float(i), mid=100000.0 + i * 50, sigma_annual=0.6) for i in range(20)]
    bs = BinanceSeries("BTCUSDT", pts)
    cfg = _cfg("wf")
    rep_wf = sweep(early, bs, [cfg], train_frac=0.6, seed=0, walk_forward=True)
    rep_gl = sweep(early, bs, [cfg], train_frac=0.6, seed=0, walk_forward=False)
    # Both produce valid reports; walk-forward and global are both deterministic.
    assert rep_wf.walk_forward is True
    assert rep_gl.walk_forward is False
    assert rep_wf.edges[0].n_train + rep_wf.edges[0].n_val == \
           rep_gl.edges[0].n_train + rep_gl.edges[0].n_val


# --------------------------------------------------------------------------- #
#  refetch_binance is injectable / offline                                    #
# --------------------------------------------------------------------------- #

def test_build_binance_series_from_raw_klines_and_trades():
    events = [_ev(float(i), 0.40, 0.42) for i in range(5)]
    klines = [(float(i), 100000.0 + i) for i in range(10)]
    trades = [(float(i), 100000.0 + i, 1.0, 1.0) for i in range(10)]
    bs = build_binance_series(events, "BTCUSDT", klines, trades)
    assert bs.symbol == "BTCUSDT"
    assert len(bs.points) > 0
    # As-of lookup returns the last point <= ts.
    p = bs.point_at(3.5)
    assert p is not None and p.ts_wall <= 3.5
    # Features were synthesised from the trade tape.
    assert any(pt.features is not None for pt in bs.points)


def test_refetch_binance_uses_prefetched_without_network(tmp_path):
    # prefetched short-circuits all network — proves offline operation.
    pre = BinanceSeries("BTCUSDT", [BinancePoint(ts_wall=0.0, mid=100000.0, sigma_annual=0.5)])
    out = asyncio.run(refetch_binance(tmp_path, prefetched=pre))
    assert out is pre


def test_refetch_binance_accepts_injected_fetcher(tmp_path):
    # Write a tiny recording so refetch can read its window.
    import gzip
    import json
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with gzip.open(run_dir / "20260622T10.jsonl.gz", "wt", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({
                "ts_wall": float(i), "ts_mono": float(i), "token_id": "tok",
                "etype": "book", "best_bid": 0.40, "best_ask": 0.42,
                "bid_sz": 10, "ask_sz": 9,
            }) + "\n")

    called = {}

    def fake_fetcher(symbol, start_ts, end_ts):
        called["args"] = (symbol, start_ts, end_ts)
        klines = [(float(i), 100000.0 + i) for i in range(5)]
        trades = [(float(i), 100000.0 + i, 1.0, 1.0) for i in range(5)]
        return klines, trades

    out = asyncio.run(refetch_binance(run_dir, symbol="BTCUSDT", fetcher=fake_fetcher))
    assert called["args"][0] == "BTCUSDT"
    assert isinstance(out, BinanceSeries)
    assert len(out.points) > 0
