"""
Baseline expected-points model with a walk-forward backtest.

The model is a ridge regression on the lagged panel features. It has to beat
the naive estimate every FPL manager already uses (mean points over the last
five gameweeks) before anything fancier earns its place.

Walk-forward, like a trading backtest: for each gameweek t, fit only on
gameweeks before t, forecast t, then step forward. Nothing from t or later
is ever seen in training.

Scored per gameweek on:
  mae / rmse          forecast error in points
  ic                  rank correlation of forecast vs. points scored
  ic_starters         same, among regular starters only (the hard part)
  top20_pts           mean points of the 20 highest-forecast players — the
                      "return" of a long-only top-20 portfolio

Writes data/backtest_results.csv (per gameweek) and data/forecast_next_gw.csv.

Run build_dataset.py first.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL_PATH = DATA_DIR / "player_gw_panel.csv"
BACKTEST_PATH = DATA_DIR / "backtest_results.csv"
FORECAST_PATH = DATA_DIR / "forecast_next_gw.csv"

NUMERIC_FEATURES = [
    "pts_mean_3", "pts_mean_5", "pts_std_5", "pts_sharpe_5", "pts_per_m_5",
    "mins_mean_3", "play_rate_5", "start_rate_5", "mins60_rate_5",
    "mins_last", "start_streak", "xgi_per90_5",
    "price_m", "difficulty", "home_share",
]
POSITIONS = ["GKP", "DEF", "MID", "FWD"]

# Gameweeks of history needed before the first forecast is scored.
MIN_TRAIN_GWS = 10
RIDGE_ALPHA = 10.0
TOP_K = 20
STARTER_RATE = 0.8


def add_period(panel: pd.DataFrame) -> pd.DataFrame:
    """Number every (season, round) in time order: 0, 1, 2, ..."""
    order = panel[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    order["period"] = np.arange(len(order))
    return panel.merge(order, on=["season", "round"])


def design_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Model inputs. Missing history becomes 0 plus a flag, not a dropped row."""
    X = panel[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.assign(has_history=(panel["gws_observed"] > 0).astype(float))
    X = X.fillna(0.0)
    for pos in POSITIONS:
        X[f"pos_{pos}"] = (panel["position"] == pos).astype(float)
    return X


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Ridge on points per fixture, scaled back up by fixtures that gameweek.

    Modelling per fixture means a double gameweek doubles the forecast instead
    of looking like an outlier the model has to learn.
    """
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(design_matrix(train), train["total_points"] / train["n_fixtures"])
    return model.predict(design_matrix(test)) * test["n_fixtures"].to_numpy()


def naive_forecast(panel: pd.DataFrame) -> pd.Series:
    """What most managers eyeball: mean points over the last five gameweeks."""
    return panel["pts_mean_5"].fillna(0.0)


def rank_ic(forecast: pd.Series, actual: pd.Series) -> float:
    return forecast.rank().corr(actual.rank())


def score(gw: pd.DataFrame, col: str) -> dict:
    err = gw[col] - gw["total_points"]
    starters = gw[gw["start_rate_5"] >= STARTER_RATE]
    return {
        "mae": err.abs().mean(),
        "rmse": np.sqrt((err ** 2).mean()),
        "ic": rank_ic(gw[col], gw["total_points"]),
        "ic_starters": rank_ic(starters[col], starters["total_points"]),
        "top20_pts": gw.nlargest(TOP_K, col)["total_points"].mean(),
    }


def compare(results: pd.DataFrame, method: str = "model", benchmark: str = "naive") -> pd.DataFrame:
    """Mean of each metric for two methods, plus a paired t-stat of the gap.

    Paired by gameweek, like comparing two strategies' returns over the same
    periods: |t| above ~2 means the gap is consistent, not a few lucky weeks.
    """
    metrics = ["mae", "rmse", "ic", "ic_starters", "top20_pts"]
    wide = results.pivot(index="period", columns="method", values=metrics)
    model = wide.xs(method, axis=1, level="method")
    naive = wide.xs(benchmark, axis=1, level="method")
    diff = model - naive
    # For errors lower is better; for IC and top-20 points higher is better.
    better = diff.copy()
    better[["mae", "rmse"]] *= -1
    return pd.DataFrame({
        method: model.mean(),
        benchmark: naive.mean(),
        "diff": diff.mean(),
        "t_stat": diff.mean() / diff.std() * np.sqrt(len(diff)),
        f"{method}_wins": (better > 0).mean(),
    })


def walk_forward(panel: pd.DataFrame, methods: dict | None = None) -> pd.DataFrame:
    """Expanding-window backtest: train on periods < t, forecast period t.

    `methods` maps a name to a fit_predict(train, test) function; the naive
    forecast is always scored alongside as the benchmark.
    """
    methods = methods or {"model": fit_predict}
    played = panel[panel["total_points"].notna()]
    rows = []
    for t in sorted(played["period"].unique()):
        if t < MIN_TRAIN_GWS:
            continue
        train = played[played["period"] < t]
        # Score only players with some history: neither method can rank a
        # debutant, and both would just guess 0.
        test = played[(played["period"] == t) & (played["gws_observed"] > 0)].copy()
        for name, method_fit_predict in methods.items():
            test[name] = method_fit_predict(train, test)
        test["naive"] = naive_forecast(test)

        season, rnd = test[["season", "round"]].iloc[0]
        for method in [*methods, "naive"]:
            rows.append({"period": t, "season": season, "round": rnd,
                         "method": method, "n": len(test), **score(test, method)})
    return pd.DataFrame(rows)


def forecast_next(panel: pd.DataFrame) -> pd.DataFrame:
    """Fit on every played gameweek and forecast the upcoming one."""
    played = panel[panel["total_points"].notna()]
    upcoming = panel[panel["total_points"].isna()].copy()
    upcoming["xpts"] = fit_predict(played, upcoming)
    upcoming["naive"] = naive_forecast(upcoming)
    cols = ["element", "web_name", "position", "team", "season", "round",
            "price_m", "difficulty", "n_fixtures", "start_rate_5", "xpts", "naive"]
    return upcoming[cols].sort_values("xpts", ascending=False, ignore_index=True)


def coefficients(panel: pd.DataFrame) -> pd.Series:
    """Standardised coefficients of the full-sample fit (per fixture)."""
    played = panel[panel["total_points"].notna()]
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    X = design_matrix(played)
    model.fit(X, played["total_points"] / played["n_fixtures"])
    coefs = pd.Series(model[-1].coef_, index=X.columns)
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index)


def main():
    panel = add_period(pd.read_csv(PANEL_PATH))

    results = walk_forward(panel)
    results.to_csv(BACKTEST_PATH, index=False)

    print(f"Walk-forward backtest over {results['period'].nunique()} gameweeks "
          f"(saved to {BACKTEST_PATH})")
    print(compare(results).round(3).to_string())

    print("\nStandardised coefficients (points per fixture):")
    print(coefficients(panel).round(3).to_string())

    forecast = forecast_next(panel)
    forecast.to_csv(FORECAST_PATH, index=False)
    print(f"\nTop 15 forecasts for {forecast['season'].iloc[0]} GW{forecast['round'].iloc[0]} "
          f"(saved to {FORECAST_PATH}):")
    print(forecast.head(15).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
