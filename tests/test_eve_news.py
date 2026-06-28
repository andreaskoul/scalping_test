import numpy as np

from src.eve_news import (
    SestmSentiment,
    augment_panel_with_sentiment,
    monthly_sentiment_factor,
    tokenize,
)
from src.eve_portfolio import Panel


def test_tokenize_drops_stopwords_and_short():
    toks = tokenize("The company BEATS earnings, a big surge!")
    assert "beats" in toks and "earnings" in toks and "surge" in toks
    assert "the" not in toks and "a" not in toks  # stopwords gone


def test_sestm_learns_directional_words():
    # "surge" only in positive docs, "plunge" only in negative -> weights +/-.
    docs, labels = [], []
    for _ in range(40):
        docs.append(["surge", "beats"]); labels.append(1)
        docs.append(["plunge", "miss"]); labels.append(-1)
    m = SestmSentiment(alpha=0.1, min_count=5).fit(docs, labels)
    assert m.weights.get("surge", 0) > 0.5
    assert m.weights.get("plunge", 0) < -0.5
    assert m.score(["surge", "beats"]) > 0
    assert m.score(["plunge"]) < 0
    assert m.score(["unseen", "words"]) == 0.0  # no sentiment words -> neutral


def _panel_with_signal(T=30, N=4, seed=0):
    # fwd return sign for symbol 0 is driven by a 'good'/'bad' headline word.
    rng = np.random.default_rng(seed)
    dates = [f"20{17 + t // 12:02d}-{1 + t % 12:02d}-28" for t in range(T)]
    fwd = rng.normal(0, 0.01, (T, N))
    return Panel(dates=dates, symbols=[f"S{j}" for j in range(N)],
                 signals=rng.normal(0, 1, (T, N, 3)), fwd_returns=fwd), dates


def test_monthly_factor_is_causal_and_shaped():
    panel, dates = _panel_with_signal(T=24, N=3)
    # Every month, symbol 0 gets a headline; word "up" when its fwd return >0 else "down".
    recs = {"S0": []}
    for t, d in enumerate(dates):
        word = "rallies" if panel.fwd_returns[t, 0] > 0 else "tumbles"
        recs["S0"].append({"created_at": f"{d}T12:00:00Z",
                           "headline": f"S0 stock {word} today", "summary": "",
                           "symbols": ["S0"]})
    fac = monthly_sentiment_factor(panel, recs, min_train_months=6, min_count=3, alpha=0.05)
    assert fac.shape == (24, 3)
    # Pre-train months are NaN (leak-safe: no dictionary yet).
    assert np.isnan(fac[:6]).all()
    # Symbols with no news are NaN throughout.
    assert np.isnan(fac[:, 1]).all() and np.isnan(fac[:, 2]).all()


def test_augment_appends_one_factor_in_range():
    panel, dates = _panel_with_signal(T=20, N=3)
    recs = {"S0": [{"created_at": f"{d}T12:00:00Z", "headline": "beats and surges",
                    "summary": "", "symbols": ["S0"]} for d in dates]}
    aug = augment_panel_with_sentiment(panel, recs, min_train_months=6, min_count=2, alpha=0.02)
    assert aug.signals.shape == (20, 3, 3 + 1)
    assert np.allclose(aug.signals[:, :, :3], panel.signals)  # originals untouched
    fund = aug.signals[:, :, 3]
    assert fund.min() >= -0.5 - 1e-9 and fund.max() <= 0.5 + 1e-9  # rank-normed
