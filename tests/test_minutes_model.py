import numpy as np
import pandas as pd

from minutes_model import (
    BUCKETS, apply_availability, log_loss, minutes_bucket, naive_minutes_proba,
)


def test_minutes_bucket_is_per_fixture():
    panel = pd.DataFrame({"minutes": [0, 45, 59, 60, 90, 100], "n_fixtures": [1, 1, 1, 1, 1, 2]})
    # 100 minutes over a double gameweek is 50 per fixture: a sub-60 bucket.
    assert minutes_bucket(panel).tolist() == [0, 1, 1, 2, 2, 1]


def test_naive_proba_is_smoothed_and_sums_to_one():
    panel = pd.DataFrame({"gws_observed": [5, 0, 2], "play_rate_5": [1.0, np.nan, 0.5],
                          "mins60_rate_5": [1.0, np.nan, 0.0]})
    proba = naive_minutes_proba(panel)
    assert np.allclose(proba.sum(axis=1), 1)
    assert proba.loc[0, "p_0"] > 0           # five straight starts is not certainty
    assert np.allclose(proba.loc[1], 1 / 3)  # no history: uniform
    assert np.isclose(proba.loc[0, "p_60"], 6 / 8)


def test_availability_flag_caps_chance_of_playing():
    proba = pd.DataFrame({"p_0": [0.1, 0.1, 0.1], "p_1_59": [0.1, 0.1, 0.1], "p_60": [0.8, 0.8, 0.8]})
    out = apply_availability(proba, pd.Series([np.nan, 50.0, 0.0]))
    assert np.allclose(out.sum(axis=1), 1)
    assert np.allclose(out.loc[0], proba.loc[0])   # no flag: unchanged
    assert np.isclose(out.loc[1, "p_60"], 0.4)
    assert np.isclose(out.loc[2, "p_0"], 1.0)      # ruled out


def test_log_loss_scores_the_bucket_that_happened():
    proba = pd.DataFrame([[0.5, 0.25, 0.25]], columns=BUCKETS)
    assert np.isclose(log_loss(proba, pd.Series([0])).iloc[0], -np.log(0.5))
    assert np.isclose(log_loss(proba, pd.Series([2])).iloc[0], -np.log(0.25))
