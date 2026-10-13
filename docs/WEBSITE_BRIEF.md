# Brief: plan the xFPL transfer-planning website

Goal: a website where users choose an arbitrary range of upcoming gameweeks and see each player's expected
points (xP) over it, to judge value and plan transfers. The backend is the xP model in this repository
(https://github.com/ofraley24/xFPL). This session is for PLANNING: architecture, data flow, model work needed,
scope of a first version. Ask the owner before building or publishing anything.

## What exists
- `analysis/13_xp_model.py`: the final model (default flags). Component models (team ratings, goals,
  assists, conceded, saves, cards, minutes given a start, bonus ranked within the match, defensive
  contribution) assembled by nested Monte Carlo: for every player-fixture a full points distribution, xP,
  quantiles, P(5+), P(10+), epistemic and total variance. Formulation and results:
  docs/HIGH_END_PLAN.md, docs/XP_EVALUATION*.md, README.md.
- Predictions are made walk-forward for FINISHED fixtures only (dev 2019-20..2024-25, `--sealed` 2025-26,
  `--live` finished 2026-27 gameweeks). Output: data/analysis/xp_predictions_<mode>.parquet (data/ is not in git).
- Upcoming fixtures already carry team ratings and expected goals (pipeline/11_pi_ratings_to_fpl.py).
- `pipeline/06_fetch_fpl_live.py` takes timestamped snapshots of the FPL API (prices, ownership, injury
  status / chance of playing, fixtures); snapshots are not yet scheduled.

## Gaps the plan must address
1. Forecast mode: xP for UPCOMING fixtures, features frozen at the current deadline, over a multi-gameweek
   horizon (double and blank gameweeks included).
2. Availability: xP is CONDITIONAL ON STARTING. Transfers need P(start): a model, FPL's chance-of-playing
   flags, a user override, or a mix. Expected points then = P(start) x xP(start) + substitute appearances
   (not modelled yet).
3. Refresh: rerun before every deadline (schedule the FPL snapshot + model run) and publish results
   for the site to read.
4. Uncertainty over a range: summing xP is easy; intervals need the correlated epistemic part
   (season-total method in docs, "+ epistemic correlated").
5. Hosting, stack, user features (filters, price / position, compare, saved squads), and data terms
   (FPL API, Understat: check terms before redistributing derived data).

## Constraints
- Decision time is the gameweek deadline; nothing may use later information.
- Never commit data/ or credentials; ask before pushing.
