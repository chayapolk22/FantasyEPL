import numpy as np
import pandas as pd

import baseline_model
from baseline_model import add_period, compare, walk_forward


def make_panel(n_seasons_rounds=((("2025-26"), 38), ("2026-27", 3)), players=30):
    rng = np.random.default_rng(0)
    rows = []
    for season, n_rounds in n_seasons_rounds:
        for rnd in range(1, n_rounds + 1):
            for p in range(players):
                rows.append({"code": p, "season": season, "round": rnd,
                             "total_points": float(rng.integers(0, 10)),
                             "gws_observed": rnd, "n_fixtures": 1})
    return add_period(pd.DataFrame(rows))


def test_periods_are_in_time_order_across_seasons():
    panel = make_panel()
    last = panel[(panel["season"] == "2025-26") & (panel["round"] == 38)]["period"].iloc[0]
    first = panel[(panel["season"] == "2026-27") & (panel["round"] == 1)]["period"].iloc[0]
    assert first == last + 1


def test_walk_forward_never_trains_on_the_future(monkeypatch):
    panel = make_panel()
    seen = []

    def spy(train, test):
        seen.append((train["period"].max(), test["period"].min()))
        return np.zeros(len(test))

    monkeypatch.setattr(baseline_model, "score",
                        lambda gw, col: {"mae": 0, "rmse": 0, "ic": 0, "ic_starters": 0, "top20_pts": 0})
    monkeypatch.setattr(baseline_model, "naive_forecast", lambda gw: pd.Series(0.0, index=gw.index))
    walk_forward(panel, fit_predict=spy)

    assert seen, "no gameweeks were forecast"
    assert all(train_max < test_t for train_max, test_t in seen)
    assert all(train_max == test_t - 1 for train_max, test_t in seen)


def test_compare_counts_lower_error_as_a_win():
    results = pd.DataFrame([
        {"period": t, "method": m, "mae": e, "rmse": e, "ic": ic, "ic_starters": ic, "top20_pts": ic}
        for t in range(4)
        for m, e, ic in (("model", 1.0, 0.3), ("naive", 2.0, 0.1))
    ])
    out = compare(results)
    assert out.loc["mae", "model_wins"] == 1.0
    assert out.loc["ic", "model_wins"] == 1.0
    assert np.isclose(out.loc["ic", "diff"], 0.2)
