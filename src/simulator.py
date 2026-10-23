"""
Correlated Monte Carlo gameweek simulator.

Turns the hurdle model's point forecasts into full distributions: each
simulated gameweek draws every player's minutes bucket, then his points.

  1. Minutes: a bucket (0 / 1-59 / 60+) from the minutes model's
     probabilities, injury flags included.
  2. Points, given minutes. For 60+ minutes a second hurdle, blank or
     return, the way most of the FPL distribution actually looks:
       P(blank)  logistic in the forecast, per position: blanks (<= 2
                 points) fall smoothly from ~75% for weak forwards to ~30%
                 for elite ones; a smooth curve carries that into the thin
                 elite tail, where buckets of data would be too small
       blank     a position's real blank scores (2, 1, 0, -1 ...)
       return    a position's real return distribution, rescaled so the
                 overall mean is exactly the hurdle model's forecast: the
                 simulator adds shape, never shifts the point forecast
     For 1-59 minutes: the forecast plus a bootstrapped residual.
  3. Correlation: teammates' residuals share two team factors, the same
     idea as sector factors for stocks:
        defence  clean sheets: goalkeepers and defenders move together
        attack   team goals: midfielders and forwards move together
     Each position has a loading on each factor, fitted to the observed
     correlation between teammates' residuals. They enter through a
     Gaussian copula: a correlated uniform draw u picks the outcome, low u
     meaning a blank and high u a haul, so teammates' good and bad days
     line up.

Without step 3 every player would be independent, and a triple-up on one
defence would look far safer than it is. That is concentration risk.

Validated walk-forward (fit on gameweeks < t, simulate t):
  coverage    share of actual scores inside the simulated 80% interval
  pit         where the actual score falls in its simulated distribution
              (a well-calibrated simulator puts it uniformly in 0-1)
  team cover  80% interval coverage of each team's total points, with and
              without the factors (correlation is what makes team totals
              swing, so this is the test of step 3)

Writes data/sim_backtest.csv, data/sim_backtest_players.csv and
data/sim_next_gw.csv.
Run build_dataset.py first.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from baseline_model import MIN_TRAIN_GWS, add_period
from minutes_model import HurdleModel, apply_availability, availability, minutes_bucket

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL_PATH = DATA_DIR / "player_gw_panel.csv"
SIM_BACKTEST_PATH = DATA_DIR / "sim_backtest.csv"
SIM_BACKTEST_PLAYERS_PATH = DATA_DIR / "sim_backtest_players.csv"
SIM_NEXT_PATH = DATA_DIR / "sim_next_gw.csv"

POSITIONS = ["GKP", "DEF", "MID", "FWD"]
FACTORS = ["defence", "attack"]
MIN_RESIDUALS = 50          # smaller (position, bucket) groups pool positions
APPEARANCE_PTS = 2          # the floor for 60+ minutes, absent cards etc.
N_SIMS = 10_000
BACKTEST_SIMS = 2_000
INTERVAL = (0.10, 0.90)     # the 80% interval
BLANK_MAX = 2               # 2 or fewer points: no attacking or defensive return
HAUL_MIN = 10


def normal_scores(x: pd.Series) -> pd.Series:
    """Map values to standard normal by rank (the copula's marginal step)."""
    return pd.Series(norm.ppf(x.rank() / (len(x) + 1)), index=x.index)


def teammate_correlations(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean product of teammates' normal-score residuals, by position pair.

    `rows` holds one row per player who played 60+ in a single-fixture
    gameweek, with a `z` column. Pairs come from the same team and gameweek.
    """
    x = rows[["season", "round", "team", "element", "position", "z"]]
    pairs = x.merge(x, on=["season", "round", "team"])
    pairs = pairs[pairs["element_x"] < pairs["element_y"]]
    pairs = pairs.assign(prod=pairs["z_x"] * pairs["z_y"])
    # Unordered pairs: (DEF, MID) and (MID, DEF) are the same thing.
    key = pairs[["position_x", "position_y"]].apply(sorted, axis=1, result_type="expand")
    return pairs.groupby([key[0], key[1]])["prod"].agg(["mean", "size"])


def fit_loadings(corr: pd.DataFrame) -> pd.DataFrame:
    """Two-factor loadings per position: corr(p, q) ~ sum_k B[p,k] * B[q,k]."""
    pairs = [(p, q, row["mean"], row["size"]) for (p, q), row in corr.iterrows()]

    def residuals(flat):
        B = dict(zip(POSITIONS, flat.reshape(len(POSITIONS), len(FACTORS))))
        res = [np.sqrt(n) * (B[p] @ B[q] - c) for p, q, c, n in pairs]
        # A loading vector can't explain more than all of a player's variance.
        res += [1e3 * max(0.0, b @ b - 0.95) for b in B.values()]
        return np.array(res)

    # Start goalkeepers and defenders on defence, attackers on attack.
    start = np.array([[0.7, 0.0], [0.7, 0.1], [0.2, 0.3], [0.0, 0.3]])
    fit = least_squares(residuals, start.ravel(), bounds=(-0.97, 0.97))
    return pd.DataFrame(fit.x.reshape(len(POSITIONS), len(FACTORS)),
                        index=POSITIONS, columns=FACTORS)


class GameweekSimulator:
    def fit(self, train: pd.DataFrame) -> "GameweekSimulator":
        self.hurdle = HurdleModel().fit(train)

        bucket = minutes_bucket(train)
        played = train[bucket > 0].assign(bucket=bucket[bucket > 0])
        played["actual"] = played["total_points"] / played["n_fixtures"]
        played["pred"] = np.nan
        for b in (1, 2):
            rows = played["bucket"] == b
            played.loc[rows, "pred"] = self.hurdle.points_if(played[rows], b)
        played["resid"] = played["actual"] - played["pred"]

        # 1-59 minutes: sorted residuals per position (pooled if thin).
        sub = played[played["bucket"] == 1]
        pooled = np.sort(sub["resid"].to_numpy())
        self.sub_residuals = {
            pos: np.sort(r.to_numpy()) if len(r) >= MIN_RESIDUALS else pooled
            for pos, r in ((pos, sub.loc[sub["position"] == pos, "resid"]) for pos in POSITIONS)
        }

        # 60+ minutes: blank-or-return per position, from single-fixture
        # gameweeks so a double gameweek's sum doesn't blur the shape.
        full = played[(played["bucket"] == 2) & (played["n_fixtures"] == 1)].copy()
        self.blank_curve, self.blank_values, self.return_excess = {}, {}, {}
        for pos in POSITIONS:
            rows = full[full["position"] == pos]
            blank = rows["actual"] <= BLANK_MAX
            curve = LogisticRegression(C=1e3).fit(rows[["pred"]].to_numpy(), blank)
            self.blank_curve[pos] = (curve.intercept_[0], curve.coef_[0, 0])
            self.blank_values[pos] = np.sort(rows.loc[blank, "actual"].to_numpy())
            self.return_excess[pos] = np.sort(rows.loc[~blank, "actual"].to_numpy() - APPEARANCE_PTS)

        full["z"] = full.groupby("position")["resid"].transform(normal_scores)
        self.correlations = teammate_correlations(full)
        self.loadings = fit_loadings(self.correlations)
        return self

    def p_blank(self, position: str, pred: np.ndarray) -> np.ndarray:
        a, b = self.blank_curve[position]
        return 1 / (1 + np.exp(-(a + b * pred)))

    def full_game_points(self, position: str, pred: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Points for 60+ minutes from uniforms u (rows: sims, cols: players)."""
        p_blank = np.clip(self.p_blank(position, pred), 1e-3, 1 - 1e-3)
        blanks, excess = self.blank_values[position], self.return_excess[position]
        # Scale returns so p_blank * E[blank] + (1 - p_blank) * E[return] = pred.
        mean_return = (pred - p_blank * blanks.mean()) / (1 - p_blank)
        scale = np.maximum(mean_return - APPEARANCE_PTS, 0.05) / excess.mean()

        is_blank = u < p_blank
        blank_pts = _empirical(blanks, u / p_blank)
        return_pts = APPEARANCE_PTS + scale * _empirical(excess, (u - p_blank) / (1 - p_blank))
        return np.where(is_blank, blank_pts, return_pts)

    def simulate(self, test: pd.DataFrame, n_sims: int = N_SIMS, seed: int = 0,
                 proba: pd.DataFrame | None = None, correlated: bool = True) -> np.ndarray:
        """Simulated points, shape (n_sims, len(test)), columns in test order."""
        rng = np.random.default_rng(seed)
        proba = self.hurdle.predict_proba(test) if proba is None else proba
        p0 = proba["p_0"].to_numpy()
        p01 = p0 + proba["p_1_59"].to_numpy()
        mean_if = {b: self.hurdle.points_if(test, b) for b in (1, 2)}

        position = test["position"].to_numpy()
        B = self.loadings.loc[position].to_numpy() if correlated else np.zeros((len(test), 2))
        idio = np.sqrt(np.clip(1 - (B ** 2).sum(axis=1), 0, 1))
        _, team_idx = np.unique(test[["season", "team"]].astype(str).agg("-".join, axis=1),
                                return_inverse=True)
        n_fixtures = test["n_fixtures"].to_numpy()

        total = np.zeros((n_sims, len(test)))
        for f in range(int(n_fixtures.max())):
            # Fresh team factors per fixture: a double gameweek is two draws.
            factors = rng.standard_normal((n_sims, team_idx.max() + 1, len(FACTORS)))
            z = (np.einsum("sjk,jk->sj", factors[:, team_idx], B)
                 + idio * rng.standard_normal((n_sims, len(test))))
            u = norm.cdf(z)

            draw = rng.random((n_sims, len(test)))
            bucket = (draw >= p0).astype(int) + (draw >= p01)

            pts = np.zeros((n_sims, len(test)))
            for pos in POSITIONS:
                cols = position == pos
                if not cols.any():
                    continue
                cameo = mean_if[1][cols] + _quantile(self.sub_residuals[pos], u[:, cols])
                full = self.full_game_points(pos, mean_if[2][cols], u[:, cols])
                b = bucket[:, cols]
                pts[:, cols] = np.select([b == 1, b == 2], [cameo, full], 0.0)
            total += pts * (n_fixtures > f)
        return total


def _quantile(sorted_values: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Empirical quantile function, interpolated: u in (0, 1) -> value."""
    n = len(sorted_values)
    return np.interp(u * (n - 1), np.arange(n), sorted_values)


def _empirical(sorted_values: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Empirical quantile function without interpolation, so 2s stay 2s.

    floor(u * n) gives every observed value the same 1/n chance.
    """
    n = len(sorted_values)
    idx = np.clip(np.floor(u * n), 0, n - 1).astype(int)
    return sorted_values[idx]


def summarise(sims: np.ndarray, players: pd.DataFrame) -> pd.DataFrame:
    lo, hi = np.quantile(sims, INTERVAL, axis=0)
    out = players.reset_index(drop=True).copy()
    out["xpts"] = sims.mean(axis=0)
    out["std"] = sims.std(axis=0)
    out["p10"], out["p50"], out["p90"] = lo, np.median(sims, axis=0), hi
    out["p_blank"] = (sims <= BLANK_MAX).mean(axis=0)
    out["p_haul"] = (sims >= HAUL_MIN).mean(axis=0)
    return out


def pit(sims: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Mid-PIT: where actual falls in its simulated distribution (0-1)."""
    return (sims < actual).mean(axis=0) + 0.5 * (sims == actual).mean(axis=0)


def team_totals(sims: np.ndarray, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    teams = test["team"].to_numpy()
    ids = np.unique(teams)
    sim_tot = np.stack([sims[:, teams == t].sum(axis=1) for t in ids], axis=1)
    actual = np.array([test.loc[teams == t, "total_points"].sum() for t in ids])
    return sim_tot, actual


def sim_backtest(panel: pd.DataFrame, n_sims: int = BACKTEST_SIMS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward calibration of the simulated distributions.

    Returns per-gameweek metrics (with and without team factors) and the
    per-player PIT / blank / haul forecasts of the correlated version.
    """
    played = panel[panel["total_points"].notna()]
    rows, player_rows = [], []
    for t in sorted(played["period"].unique()):
        if t < MIN_TRAIN_GWS:
            continue
        sim = GameweekSimulator().fit(played[played["period"] < t])
        test = played[(played["period"] == t) & (played["gws_observed"] > 0)]
        actual = test["total_points"].to_numpy()
        season, rnd = test[["season", "round"]].iloc[0]

        for correlated in (True, False):
            sims = sim.simulate(test, n_sims, seed=int(t), correlated=correlated)
            lo, hi = np.quantile(sims, INTERVAL, axis=0)
            inside = (actual >= lo) & (actual <= hi)
            played_mask = test["minutes"].to_numpy() > 0
            team_sims, team_actual = team_totals(sims, test)
            t_lo, t_hi = np.quantile(team_sims, INTERVAL, axis=0)
            rows.append({
                "period": t, "season": season, "round": rnd, "correlated": correlated,
                "coverage": inside.mean(),
                "coverage_played": inside[played_mask].mean(),
                "pit_mean": pit(sims, actual).mean(),
                "team_coverage": ((team_actual >= t_lo) & (team_actual <= t_hi)).mean(),
                "team_std_ratio": np.std(team_actual - team_sims.mean(axis=0)) / team_sims.std(axis=0).mean(),
            })
            if correlated:
                player_rows.append(pd.DataFrame({
                    "period": t, "element": test["element"].to_numpy(),
                    "position": test["position"].to_numpy(), "actual": actual,
                    "pit": pit(sims, actual), "xpts": sims.mean(axis=0),
                    # P(sim < actual) and P(sim <= actual): the bounds a
                    # randomised PIT draws between when outcomes tie (0s, 2s).
                    "pit_lo": (sims < actual).mean(axis=0),
                    "pit_hi": (sims <= actual).mean(axis=0),
                    "p_blank": (sims <= BLANK_MAX).mean(axis=0),
                    "p_haul": (sims >= HAUL_MIN).mean(axis=0),
                }))
    return pd.DataFrame(rows), pd.concat(player_rows, ignore_index=True)


def simulate_next(panel: pd.DataFrame, n_sims: int = N_SIMS) -> tuple[pd.DataFrame, np.ndarray, GameweekSimulator]:
    played = panel[panel["total_points"].notna()]
    upcoming = panel[panel["total_points"].isna()].merge(availability(), on="element", how="left")
    upcoming = upcoming.reset_index(drop=True)

    sim = GameweekSimulator().fit(played)
    proba = apply_availability(sim.hurdle.predict_proba(upcoming), upcoming["chance_pct"])
    sims = sim.simulate(upcoming, n_sims, proba=proba)

    cols = ["element", "web_name", "position", "team", "season", "round", "price_m", "chance_pct"]
    summary = summarise(sims, upcoming[cols])
    summary["p_play"] = 1 - proba["p_0"].to_numpy()
    return summary, sims, sim


def main():
    panel = add_period(pd.read_csv(PANEL_PATH))

    bt, bt_players = sim_backtest(panel)
    bt.to_csv(SIM_BACKTEST_PATH, index=False)
    bt_players.to_csv(SIM_BACKTEST_PLAYERS_PATH, index=False)
    print(f"Simulator walk-forward over {bt['period'].nunique()} gameweeks "
          f"(target: coverage 0.80, pit_mean 0.50, team_std_ratio 1.0)")
    print(bt.groupby("correlated")[["coverage", "coverage_played", "pit_mean",
                                    "team_coverage", "team_std_ratio"]].mean().round(3).to_string())

    print("\nTail calibration (correlated): forecast probability vs. how often it happened")
    for col, hit in (("p_blank", bt_players["actual"] <= BLANK_MAX),
                     ("p_haul", bt_players["actual"] >= HAUL_MIN)):
        bins = pd.cut(bt_players[col], [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 1.0], include_lowest=True)
        cal = pd.DataFrame({"forecast": bt_players[col], "actual": hit}).groupby(bins, observed=True)
        print(f"{col}:\n" + cal.agg(forecast=("forecast", "mean"), actual=("actual", "mean"),
                                    n=("actual", "size")).round(3).to_string())

    summary, sims, sim = simulate_next(panel)
    summary.to_csv(SIM_NEXT_PATH, index=False)
    print("\nTeam factor loadings (full sample):")
    print(sim.loadings.round(3).to_string())
    print(f"\nTop 15 for GW{summary['round'].iloc[0]} by simulated mean "
          f"({N_SIMS:,} sims, saved to {SIM_NEXT_PATH}):")
    show = ["web_name", "position", "price_m", "p_play", "xpts", "std", "p10", "p50", "p90", "p_blank", "p_haul"]
    print(summary.sort_values("xpts", ascending=False)[show].head(15).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
