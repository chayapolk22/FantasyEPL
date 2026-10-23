import numpy as np
import pandas as pd

from simulator import (
    FACTORS, POSITIONS, GameweekSimulator, _empirical, fit_loadings, pit,
)


class StubHurdle:
    """Fixed forecasts so the simulator can be tested without fitting."""

    def __init__(self, p0, p_sub, p60, pts_sub, pts_full):
        self.p = (p0, p_sub, p60)
        self.pts = {1: pts_sub, 2: pts_full}

    def predict_proba(self, test):
        p0, p_sub, p60 = self.p
        return pd.DataFrame({"p_0": p0, "p_1_59": p_sub, "p_60": p60}, index=test.index)

    def points_if(self, test, bucket):
        return np.full(len(test), self.pts[bucket], dtype=float)


def make_sim(p0=0.0, p_sub=0.0, p60=1.0, pts_full=5.0, loadings=None):
    sim = GameweekSimulator()
    sim.hurdle = StubHurdle(p0, p_sub, p60, 1.0, pts_full)
    sim.sub_residuals = {pos: np.array([-1.0, 0.0, 1.0]) for pos in POSITIONS}
    sim.blank_curve = {pos: (0.0, -0.3) for pos in POSITIONS}      # P(blank) falls with forecast
    sim.blank_values = {pos: np.array([1.0, 2.0, 2.0, 2.0]) for pos in POSITIONS}
    sim.return_excess = {pos: np.array([1.0, 2.0, 4.0, 8.0, 12.0]) for pos in POSITIONS}
    sim.loadings = loadings if loadings is not None else pd.DataFrame(0.0, index=POSITIONS, columns=FACTORS)
    return sim


def make_players(n=4, team=1, position="DEF"):
    return pd.DataFrame({"position": [position] * n, "team": [team] * n,
                         "season": ["2026-27"] * n, "n_fixtures": [1] * n})


def test_full_game_points_keep_the_forecast_mean():
    sim = make_sim()
    u = np.random.default_rng(0).random((200_000, 3))
    pred = np.array([3.0, 5.0, 8.0])
    pts = sim.full_game_points("FWD", pred, u)
    assert np.allclose(pts.mean(axis=0), pred, atol=0.05)
    # Better forecasts blank less often.
    blank_rate = (pts <= 2).mean(axis=0)
    assert blank_rate[0] > blank_rate[1] > blank_rate[2]


def test_benched_players_score_zero():
    sim = make_sim(p0=1.0, p60=0.0)
    sims = sim.simulate(make_players(), n_sims=500)
    assert (sims == 0).all()


def test_double_gameweek_doubles_the_mean():
    sim = make_sim(pts_full=4.0)
    players = make_players(2)
    players.loc[1, "n_fixtures"] = 2
    sims = sim.simulate(players, n_sims=50_000)
    assert np.isclose(sims[:, 1].mean() / sims[:, 0].mean(), 2, atol=0.05)


def test_team_factor_correlates_teammates_only():
    loadings = pd.DataFrame([[0.8, 0.0]] * 4, index=POSITIONS, columns=FACTORS)
    sim = make_sim(loadings=loadings)
    players = pd.concat([make_players(2, team=1), make_players(1, team=2)], ignore_index=True)
    sims = sim.simulate(players, n_sims=50_000)
    corr = np.corrcoef(sims.T)
    assert corr[0, 1] > 0.3            # teammates
    assert abs(corr[0, 2]) < 0.03      # different teams

    independent = sim.simulate(players, n_sims=50_000, correlated=False)
    assert abs(np.corrcoef(independent.T)[0, 1]) < 0.03


def test_fit_loadings_recovers_a_known_structure():
    B = pd.DataFrame([[0.8, 0.1], [0.7, 0.2], [0.2, 0.3], [0.0, 0.4]], index=POSITIONS, columns=FACTORS)
    rows = []
    for i, p in enumerate(POSITIONS):
        for q in POSITIONS[i:]:
            rows.append((p, q, B.loc[p] @ B.loc[q], 1000))
    corr = pd.DataFrame(rows, columns=["p", "q", "mean", "size"]).set_index(["p", "q"])
    fitted = fit_loadings(corr)
    implied = fitted.to_numpy() @ fitted.to_numpy().T
    assert np.allclose(implied, B.to_numpy() @ B.to_numpy().T, atol=0.02)


def test_empirical_draws_only_observed_values():
    values = np.array([0.0, 2.0, 2.0, 6.0])
    assert set(_empirical(values, np.linspace(0, 1, 101))) <= set(values)


def test_pit_is_uniform_for_a_matching_distribution():
    rng = np.random.default_rng(1)
    sims = rng.normal(size=(5_000, 2_000))
    actual = rng.normal(size=2_000)
    assert abs(pit(sims, actual).mean() - 0.5) < 0.02
