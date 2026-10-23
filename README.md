# FantasyEPL
FantasyEPL

Monte Carlo and Bayesian modeling for Fantasy Premier League — stochastic season simulation, rotation-risk prediction, and portfolio-style squad optimization.

A side project exploring FPL through a quantitative finance lens: treating squad selection as portfolio construction, chip usage as optimal stopping, and mini-league standing as a relative-return problem.

## Why frame FPL this way?

| FPL concept | Quant equivalent |
|---|---|
| Player expected points | Expected return |
| Player points variance/uncertainty | Return volatility / risk |
| Players from the same club owned together | Correlated positions (concentration risk) |
| £100m squad constraint | Budget/leverage constraint |
| Weekly transfers | Portfolio rebalancing with transaction costs (points hits) |
| Chips (Bench Boost, Free Hit, Triple Captain, Wildcard) | Optionality — when to exercise a limited-use option |
| Captain choice | Leveraged position (2x) sizing decision |
| Mini-league standing | Relative-return / active-management problem, not absolute return |

## Architecture

**Layer 1 — Data & base models**
-  FPL API ingestion pipeline
-  Understat / FBref xG & xA scraping
-  Baseline points-prediction model (XGBoost)
-  Price-change classifier

**Layer 2 — Uncertainty modeling**
-  Bayesian / survival model for minutes-played probability (rotation risk)
-  Per-player predictive distributions (not just point estimates)

**Layer 3 — Monte Carlo & stochastic optimization (core)**
-  Correlated Monte Carlo season simulator
-  Mean-variance squad optimizer (Markowitz-style)
-  Multi-gameweek chip-timing MDP / optimal stopping
-  League-relative game-theoretic strategy simulation

**Layer 4 — Explainability**
-  SHAP-based reasoning surfaced alongside every recommendation
-  Fan charts / percentile bands instead of single-number outputs

**Layer 5 — Product layer**
-  Squad builder with save/load (Team 1 / Team 2)
-  LLM chat wrapper narrating model outputs (no invented numbers)
-  Chip planner UI
-  Mini-league dashboard

## Tech stack

- **Modeling:** Python — `pandas`, `numpy`, `scikit-learn`, `xgboost`, `pymc` / `lifelines` (survival analysis), `shap`
- **Optimization:** `cvxpy` (mean-variance), backward induction / DP for chip timing
- **Backend:** FastAPI
- **Frontend:** Streamlit (prototype) → possible migration later
- **Database:** SQLite (local) → Supabase/Postgres (later)

## Getting started

```bash
pip install -r requirements.txt
python src/fetch_data.py      # players, teams, gameweeks, fixtures
python src/fetch_history.py   # per-player match history (~3 min, saved to data/raw/)
python src/build_dataset.py   # -> data/player_gw_panel.csv
python -m pytest              # includes a no-look-ahead check on the features
```

## Data: the player-gameweek panel

`data/player_gw_panel.csv` has one row per player per gameweek, i.e. one row per asset per period. `total_points` is the period return, and it's left empty for the upcoming gameweek, which is the one to forecast.

| Feature | Quant meaning |
|---|---|
| `pts_mean_3`, `pts_mean_5` | Trailing mean return: naive expected-return estimate |
| `pts_std_5` | Trailing volatility |
| `pts_sharpe_5` | (mean − 2) / std: excess return over the 2 appearance points, FPL's risk-free rate |
| `pts_per_m_5` | Return per £m of budget |
| `mins_mean_3`, `play_rate_5`, `start_rate_5` | Rotation risk: will the asset pay out at all? |
| `xgi_per90_5` | Underlying chance quality: fundamentals behind noisy realised points |
| `difficulty`, `home_share`, `price_m` | Known before the deadline, so not lagged |

Every trailing feature is lagged by one gameweek, so a row only uses information available at that gameweek's deadline, the same rule as a backtest. `notebooks/02_player_panel_checks.ipynb` spot-checks the panel and measures each signal's information coefficient.
