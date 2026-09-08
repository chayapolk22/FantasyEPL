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
