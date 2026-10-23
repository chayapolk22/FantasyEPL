"""
Rotation-risk model: how many minutes will a player get, and what are they
worth?

Credit-risk framing. A bond's expected loss is P(default) x loss given
default; here expected points are

    xpts = P(60+) * E[pts | 60+]  +  P(1-59) * E[pts | 1-59]  +  P(0) * 0

so "will he play?" and "what does he score when he plays?" are separate
models (a hurdle model, in statistics). The first is a multinomial logistic
regression over three minutes buckets per fixture: 0, 1-59 and 60+ (the line
for full appearance points). The second is a ridge regression fitted only on
gameweeks in that bucket.

The split matters for everything downstream: the Monte Carlo simulator in
Layer 3 needs to draw "did he play?" before drawing points, because a
benched player is a 0 no matter how good his fixture is.

Backtests (walk-forward, same setup as baseline_model.py):
  minutes:  multiclass log loss and Brier score vs. naive frequencies from
            the last five gameweeks
  points:   hurdle xpts vs. the baseline ridge vs. the naive 5-GW mean

At the deadline FPL publishes each flagged player's chance of playing
(injury, suspension, illness). The forecast caps P(plays) with it. There is no
history of those flags, so the backtest cannot use them.

Writes data/minutes_backtest.csv, data/backtest_results_hurdle.csv
and data/forecast_next_gw_minutes.csv. Run build_dataset.py first.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline_model import (
    MIN_TRAIN_GWS, RIDGE_ALPHA, add_period, compare, design_matrix,
    walk_forward,
)
from baseline_model import fit_predict as ridge_fit_predict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL_PATH = DATA_DIR / "player_gw_panel.csv"
MINUTES_BACKTEST_PATH = DATA_DIR / "minutes_backtest.csv"
BACKTEST_PATH = DATA_DIR / "backtest_results_hurdle.csv"
FORECAST_PATH = DATA_DIR / "forecast_next_gw_minutes.csv"

# Minutes buckets per fixture. 60 is where appearance points go from 1 to 2.
BUCKETS = ["p_0", "p_1_59", "p_60"]
NAIVE_WINDOW = 5


def minutes_bucket(panel: pd.DataFrame) -> pd.Series:
    """0 = didn't play, 1 = 1-59 minutes, 2 = 60+ (per fixture)."""
    per_fixture = panel["minutes"] / panel["n_fixtures"]
    return pd.Series(
        np.select([per_fixture <= 0, per_fixture < 60], [0, 1], default=2),
        index=panel.index,
    )


class HurdleModel:
    """P(minutes bucket) x E[points per fixture | bucket]."""

    def fit(self, train: pd.DataFrame) -> "HurdleModel":
        X = design_matrix(train)
        bucket = minutes_bucket(train)
        self.minutes = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        self.minutes.fit(X, bucket)

        # A benched player scores 0 by definition, so only buckets 1 and 2
        # need a points model.
        per_fixture = train["total_points"] / train["n_fixtures"]
        self.points = {}
        for b in (1, 2):
            in_bucket = bucket == b
            model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
            self.points[b] = model.fit(X[in_bucket], per_fixture[in_bucket])
        return self

    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame:
        proba = self.minutes.predict_proba(design_matrix(test))
        return pd.DataFrame(proba, columns=BUCKETS, index=test.index)

    def points_if(self, test: pd.DataFrame, bucket: int) -> np.ndarray:
        return self.points[bucket].predict(design_matrix(test))

    def expected_points(self, test: pd.DataFrame, proba: pd.DataFrame | None = None) -> np.ndarray:
        proba = self.predict_proba(test) if proba is None else proba
        per_fixture = (proba["p_1_59"] * self.points_if(test, 1)
                       + proba["p_60"] * self.points_if(test, 2))
        return per_fixture.to_numpy() * test["n_fixtures"].to_numpy()


def hurdle_fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return HurdleModel().fit(train).expected_points(test)


def naive_minutes_proba(panel: pd.DataFrame) -> pd.DataFrame:
    """Bucket frequencies over the last five gameweeks, add-one smoothed.

    The smoothing keeps a player who started five in a row from getting
    P(benched) = 0, which would make log loss infinite the day he is.
    """
    n = panel["gws_observed"].clip(upper=NAIVE_WINDOW)
    play = panel["play_rate_5"].fillna(0)
    full = panel["mins60_rate_5"].fillna(0)
    counts = pd.DataFrame({
        "p_0": (1 - play) * n,
        "p_1_59": (play - full) * n,
        "p_60": full * n,
    })
    return (counts + 1).div(n + len(BUCKETS), axis=0)


def log_loss(proba: pd.DataFrame, bucket: pd.Series) -> pd.Series:
    """Per-row multiclass log loss: -log P(the bucket that happened)."""
    p = proba.to_numpy()[np.arange(len(proba)), bucket.to_numpy()]
    return pd.Series(-np.log(np.clip(p, 1e-15, 1)), index=proba.index)


def brier(proba: pd.DataFrame, bucket: pd.Series) -> pd.Series:
    onehot = np.eye(len(BUCKETS))[bucket.to_numpy()]
    return pd.Series(((proba.to_numpy() - onehot) ** 2).sum(axis=1), index=proba.index)


def minutes_backtest(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-level walk-forward minutes forecasts: model and naive, side by side."""
    played = panel[panel["total_points"].notna()]
    rows = []
    for t in sorted(played["period"].unique()):
        if t < MIN_TRAIN_GWS:
            continue
        train = played[played["period"] < t]
        test = played[(played["period"] == t) & (played["gws_observed"] > 0)]
        model = HurdleModel().fit(train)
        out = test[["period", "season", "round", "element", "web_name",
                    "position", "start_rate_5"]].copy()
        out["bucket"] = minutes_bucket(test)
        out[BUCKETS] = model.predict_proba(test)
        out[[f"naive_{b}" for b in BUCKETS]] = naive_minutes_proba(test).to_numpy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def summarise_minutes(bt: pd.DataFrame) -> pd.DataFrame:
    naive = bt[[f"naive_{b}" for b in BUCKETS]].set_axis(BUCKETS, axis=1)
    per_row = pd.DataFrame({
        "period": bt["period"],
        "logloss_model": log_loss(bt[BUCKETS], bt["bucket"]),
        "logloss_naive": log_loss(naive, bt["bucket"]),
        "brier_model": brier(bt[BUCKETS], bt["bucket"]),
        "brier_naive": brier(naive, bt["bucket"]),
    })
    by_gw = per_row.groupby("period").mean()
    out = {}
    for metric in ("logloss", "brier"):
        diff = by_gw[f"{metric}_model"] - by_gw[f"{metric}_naive"]
        out[metric] = {
            "model": by_gw[f"{metric}_model"].mean(),
            "naive": by_gw[f"{metric}_naive"].mean(),
            "diff": diff.mean(),
            "t_stat": diff.mean() / diff.std() * np.sqrt(len(diff)),
            "model_wins": (diff < 0).mean(),
        }
    return pd.DataFrame(out).T


def availability(bootstrap_path: Path = DATA_DIR / "bootstrap_static.json") -> pd.DataFrame:
    """FPL's published chance of playing next round (NaN = no flag)."""
    with open(bootstrap_path) as f:
        elements = pd.DataFrame(json.load(f)["elements"])
    return elements[["id", "chance_of_playing_next_round", "status", "news"]].rename(
        columns={"id": "element", "chance_of_playing_next_round": "chance_pct"}
    )


def apply_availability(proba: pd.DataFrame, chance_pct: pd.Series) -> pd.DataFrame:
    """Cap P(plays) at FPL's flag: scale both playing buckets, rest goes to 0."""
    scale = (chance_pct.fillna(100) / 100).clip(0, 1)
    out = proba.copy()
    out["p_1_59"] *= scale
    out["p_60"] *= scale
    out["p_0"] = 1 - out["p_1_59"] - out["p_60"]
    return out


def forecast_next(panel: pd.DataFrame) -> pd.DataFrame:
    played = panel[panel["total_points"].notna()]
    upcoming = panel[panel["total_points"].isna()].merge(availability(), on="element", how="left")

    model = HurdleModel().fit(played)
    proba = apply_availability(model.predict_proba(upcoming), upcoming["chance_pct"])
    upcoming[BUCKETS] = proba
    upcoming["pts_if_60"] = model.points_if(upcoming, 2) * upcoming["n_fixtures"]
    upcoming["xpts"] = model.expected_points(upcoming, proba)
    upcoming["xpts_ridge"] = ridge_fit_predict(played, upcoming)

    cols = ["element", "web_name", "position", "team", "season", "round", "price_m",
            "difficulty", "n_fixtures", "chance_pct", "news", *BUCKETS,
            "pts_if_60", "xpts", "xpts_ridge"]
    return upcoming[cols].sort_values("xpts", ascending=False, ignore_index=True)


def main():
    panel = add_period(pd.read_csv(PANEL_PATH))

    bt = minutes_backtest(panel)
    bt.to_csv(MINUTES_BACKTEST_PATH, index=False)
    print(f"Minutes model, walk-forward over {bt['period'].nunique()} gameweeks "
          f"(lower is better; saved to {MINUTES_BACKTEST_PATH})")
    print(summarise_minutes(bt).round(4).to_string())

    results = walk_forward(panel, {"hurdle": hurdle_fit_predict, "ridge": ridge_fit_predict})
    results.to_csv(BACKTEST_PATH, index=False)
    print("\nExpected points: hurdle vs. baseline ridge")
    print(compare(results, "hurdle", "ridge").round(3).to_string())
    print("\nExpected points: hurdle vs. naive 5-GW mean")
    print(compare(results, "hurdle", "naive").round(3).to_string())

    forecast = forecast_next(panel)
    forecast.to_csv(FORECAST_PATH, index=False)
    print(f"\nTop 15 for {forecast['season'].iloc[0]} GW{forecast['round'].iloc[0]} "
          f"(saved to {FORECAST_PATH}):")
    show = ["web_name", "position", "price_m", "chance_pct", *BUCKETS, "pts_if_60", "xpts", "xpts_ridge"]
    print(forecast[show].head(15).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
