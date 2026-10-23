"""
Builds the player-gameweek panel: one row per player per gameweek, with
lagged features and the points scored that gameweek as the target.

Quant framing — each player is an asset, each gameweek a period:
  total_points        the period "return" we want to predict (target)
  pts_mean_*          trailing mean return  -> naive expected-return estimate
  pts_std_5           trailing volatility   -> risk
  pts_sharpe_5        (mean - 2) / std      -> risk-adjusted excess return; the
                      2 appearance points a starter banks for playing 60' are
                      FPL's "risk-free rate"
  pts_per_m_5         mean / price          -> return per £m of budget
  play_rate_5,
  start_rate_5,
  mins_mean_3         rotation risk         -> chance the asset pays out at all
  xgi_per90_5         underlying chance quality, the "fundamentals" behind
                      noisy realised points (like earnings vs. price moves)
  difficulty, is_home exposure to this period's fixture

No look-ahead: every pts_/mins_/rate/xgi feature in a row only uses
gameweeks *before* that row (shift by one), exactly like a backtest that
only trades on information available at the deadline. Fixture, price and
home/away are known before the deadline, so they are not lagged.

Rows for the next, unplayed gameweek are appended with an empty target, so
the same features can be fed to a model to forecast it.

Run fetch_data.py and fetch_history.py first.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "element_summary"
OUT_PATH = DATA_DIR / "player_gw_panel.csv"

# Points for playing 60+ minutes: the return you get just for "holding" a
# starter, so it plays the role of the risk-free rate in the Sharpe ratio.
RISK_FREE_PTS = 2

# Per-match stats summed into a gameweek (a double gameweek has two matches).
COUNT_COLS = [
    "total_points", "minutes", "starts", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "bonus", "bps", "saves",
    "defensive_contribution", "yellow_cards", "red_cards",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded",
]


def load_json(name: str):
    with open(DATA_DIR / name) as f:
        return json.load(f)


def load_match_history(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Every match row from every saved element-summary file."""
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        with open(path) as f:
            rows.extend(json.load(f)["history"])
    if not rows:
        raise FileNotFoundError(f"No history in {raw_dir} — run fetch_history.py")

    history = pd.DataFrame(rows)
    # xG-type stats arrive as strings ("0.35").
    for col in COUNT_COLS:
        history[col] = pd.to_numeric(history[col])
    return history


def add_fixture_info(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Attach the player's team and the FPL difficulty rating of each match."""
    fx = fixtures[["id", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]]
    out = matches.merge(fx, left_on="fixture", right_on="id", how="left", suffixes=("", "_fx"))

    home = out["was_home"].astype(bool)
    # Team comes from the fixture, not bootstrap, so mid-season transfers
    # are attributed to the club the player was actually at.
    out["team"] = np.where(home, out["team_h"], out["team_a"])
    out["opponent_team"] = np.where(home, out["team_a"], out["team_h"])
    out["difficulty"] = np.where(home, out["team_h_difficulty"], out["team_a_difficulty"])
    return out.drop(columns=["id", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"])


def upcoming_matches(bootstrap: dict, fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per player per fixture in the next gameweek, target left empty."""
    next_events = [e["id"] for e in bootstrap["events"] if e["is_next"]]
    if not next_events:
        return pd.DataFrame()
    next_gw = next_events[0]

    players = pd.DataFrame(bootstrap["elements"])[["id", "team", "now_cost"]]
    gw_fixtures = fixtures[fixtures["event"] == next_gw]

    sides = []
    for side, other in (("h", "a"), ("a", "h")):
        s = gw_fixtures[["id", f"team_{side}", f"team_{other}", f"team_{side}_difficulty"]]
        s = s.rename(columns={
            "id": "fixture",
            f"team_{side}": "team",
            f"team_{other}": "opponent_team",
            f"team_{side}_difficulty": "difficulty",
        })
        s["was_home"] = side == "h"
        sides.append(s)
    team_fixtures = pd.concat(sides, ignore_index=True)

    # Players at a club with no fixture (blank gameweek) get no row.
    out = players.merge(team_fixtures, on="team").rename(
        columns={"id": "element", "now_cost": "value"}
    )
    out["round"] = next_gw
    for col in COUNT_COLS:
        out[col] = np.nan
    return out


def to_gameweeks(matches: pd.DataFrame) -> pd.DataFrame:
    """Collapse matches into one row per player per gameweek."""
    agg = {col: lambda s: s.sum(min_count=1) for col in COUNT_COLS}
    agg.update({
        "team": "last",
        "value": "last",
        "difficulty": "mean",
        "was_home": "mean",
        "fixture": "count",
    })
    gw = (
        matches.groupby(["element", "round"], as_index=False)
        .agg(agg)
        .rename(columns={"fixture": "n_fixtures", "was_home": "home_share"})
    )
    return gw.sort_values(["element", "round"], ignore_index=True)


def add_features(gw: pd.DataFrame) -> pd.DataFrame:
    """Trailing features, each lagged one gameweek per player (no look-ahead)."""
    gw = gw.sort_values(["element", "round"], ignore_index=True).copy()
    by_player = gw.groupby("element", group_keys=False)

    def lagged_rolling(col: str, window: int, stat: str, min_periods: int = 1):
        return by_player[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).agg(stat)
        )

    gw["played"] = (gw["minutes"] > 0).astype(float).where(gw["minutes"].notna())

    gw["pts_mean_3"] = lagged_rolling("total_points", 3, "mean")
    gw["pts_mean_5"] = lagged_rolling("total_points", 5, "mean")
    gw["pts_std_5"] = lagged_rolling("total_points", 5, "std", min_periods=3)
    gw["pts_sharpe_5"] = (gw["pts_mean_5"] - RISK_FREE_PTS) / gw["pts_std_5"].replace(0, np.nan)

    gw["price_m"] = gw["value"] / 10
    gw["pts_per_m_5"] = gw["pts_mean_5"] / gw["price_m"]

    gw["mins_mean_3"] = lagged_rolling("minutes", 3, "mean")
    gw["play_rate_5"] = lagged_rolling("played", 5, "mean")
    gw["start_rate_5"] = lagged_rolling("starts", 5, "mean")

    xgi_5 = lagged_rolling("expected_goal_involvements", 5, "sum")
    mins_5 = lagged_rolling("minutes", 5, "sum")
    gw["xgi_per90_5"] = 90 * xgi_5 / mins_5.replace(0, np.nan)

    gw["gws_observed"] = by_player["total_points"].transform(
        lambda s: s.shift(1).notna().cumsum()
    )
    return gw.drop(columns=["played"])


def build_panel() -> pd.DataFrame:
    bootstrap = load_json("bootstrap_static.json")
    fixtures = pd.DataFrame(load_json("fixtures.json"))

    played = add_fixture_info(load_match_history(), fixtures)
    matches = pd.concat([played, upcoming_matches(bootstrap, fixtures)], ignore_index=True)
    panel = add_features(to_gameweeks(matches))

    players = pd.DataFrame(bootstrap["elements"])[["id", "web_name", "element_type"]]
    positions = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    players["position"] = players["element_type"].map(positions)
    panel = panel.merge(
        players.drop(columns="element_type"), left_on="element", right_on="id"
    ).drop(columns="id")

    front = ["element", "web_name", "position", "team", "round", "n_fixtures",
             "price_m", "difficulty", "home_share"]
    return panel[front + [c for c in panel.columns if c not in front]]


def main():
    panel = build_panel()
    panel.to_csv(OUT_PATH, index=False)

    played = panel["total_points"].notna()
    print(f"Saved {len(panel)} rows to {OUT_PATH}")
    print(f"  played gameweeks: {panel.loc[played, 'round'].min()}–"
          f"{panel.loc[played, 'round'].max()} ({played.sum()} rows)")
    print(f"  upcoming rows to forecast: {(~played).sum()}")


if __name__ == "__main__":
    main()
