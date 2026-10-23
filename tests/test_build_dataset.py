import numpy as np
import pandas as pd

from build_dataset import COUNT_COLS, add_features, to_gameweeks


def make_matches(points_by_round, element=1, minutes=90, season="2026-27"):
    rows = []
    for rnd, pts in points_by_round.items():
        row = {col: 0.0 for col in COUNT_COLS}
        row.update(total_points=pts, minutes=minutes, starts=1,
                   expected_goal_involvements=0.5)
        row.update(code=element, element=element, season=season, round=rnd, fixture=rnd, team=1,
                   value=60, difficulty=3, was_home=True)
        rows.append(row)
    return pd.DataFrame(rows)


FEATURES = ["pts_mean_3", "pts_mean_5", "pts_std_5", "pts_sharpe_5",
            "pts_per_m_5", "mins_mean_3", "play_rate_5", "start_rate_5",
            "xgi_per90_5", "gws_observed", "mins60_rate_5", "mins_last", "start_streak"]


def test_no_look_ahead():
    """Changing a gameweek's result must not change that gameweek's features."""
    base = make_matches({1: 2, 2: 8, 3: 5, 4: 1, 5: 12})
    shocked = base.copy()
    shocked.loc[shocked["round"] >= 4, "total_points"] = 99

    a = add_features(to_gameweeks(base)).set_index("round")
    b = add_features(to_gameweeks(shocked)).set_index("round")

    pd.testing.assert_frame_equal(a.loc[:4, FEATURES], b.loc[:4, FEATURES])
    assert not np.isclose(a.loc[5, "pts_mean_3"], b.loc[5, "pts_mean_3"])


def test_features_use_only_previous_gameweeks():
    gw = add_features(to_gameweeks(make_matches({1: 2, 2: 8, 3: 5, 4: 1})))
    gw = gw.set_index("round")

    assert np.isnan(gw.loc[1, "pts_mean_3"])
    assert gw.loc[4, "pts_mean_3"] == (2 + 8 + 5) / 3
    assert gw.loc[4, "gws_observed"] == 3
    assert gw.loc[4, "pts_per_m_5"] == (2 + 8 + 5) / 3 / 6.0
    # Sharpe is excess over the 2 appearance points, divided by volatility.
    assert np.isclose(gw.loc[4, "pts_sharpe_5"], (5 - 2) / np.std([2, 8, 5], ddof=1))


def test_double_gameweek_is_one_row():
    matches = make_matches({1: 3, 2: 4})
    second_match = matches[matches["round"] == 2].assign(fixture=99, total_points=6)
    gw = to_gameweeks(pd.concat([matches, second_match])).set_index("round")

    assert gw.loc[2, "n_fixtures"] == 2
    assert gw.loc[2, "total_points"] == 10
    assert gw.loc[2, "minutes"] == 180


def test_upcoming_row_keeps_empty_target():
    matches = make_matches({1: 3, 2: 4})
    upcoming = matches.iloc[[0]].assign(round=3)
    upcoming[COUNT_COLS] = np.nan
    gw = add_features(to_gameweeks(pd.concat([matches, upcoming]))).set_index("round")

    assert np.isnan(gw.loc[3, "total_points"])
    assert gw.loc[3, "pts_mean_3"] == 3.5


def test_features_roll_over_seasons():
    """GW1 of a new season uses the end of last season, keyed by player code."""
    last = make_matches({37: 6, 38: 10}, element=250, season="2025-26")
    this = make_matches({1: 3}, element=12, season="2026-27")
    this["code"] = 250  # same player, new season id
    gw = add_features(to_gameweeks(pd.concat([this, last])))

    gw1 = gw[(gw["season"] == "2026-27") & (gw["round"] == 1)].iloc[0]
    assert gw1["pts_mean_3"] == 8
    assert gw1["gws_observed"] == 2
    assert gw1["element"] == 12


def test_minutes_features():
    matches = make_matches({1: 2, 2: 2, 3: 1, 4: 2, 5: 2})
    matches.loc[matches["round"] == 3, ["minutes", "starts"]] = [20, 0]
    gw = add_features(to_gameweeks(matches)).set_index("round")

    assert gw.loc[3, "start_streak"] == 2
    assert gw.loc[4, "start_streak"] == 0      # benched in GW3
    assert gw.loc[5, "start_streak"] == 1
    assert gw.loc[4, "mins_last"] == 20
    assert gw.loc[4, "mins60_rate_5"] == 2 / 3


def test_xgi_per90_is_shrunk_for_tiny_samples():
    matches = make_matches({1: 2, 2: 2}, minutes=3)
    matches["expected_goal_involvements"] = 0.5
    gw = add_features(to_gameweeks(matches)).set_index("round")
    # 0.5 xGI in 3 minutes is 15 per 90 unshrunk; the prior pulls it near 0.3.
    assert gw.loc[2, "xgi_per90_5"] < 0.5
