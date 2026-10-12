# xFPL: expected-points model for Fantasy Premier League

## Objective
Predict the Fantasy Premier League (FPL) points each player will score in each
gameweek, treated as a research project that must withstand academic scrutiny.
The approach has three stages:

1. **Fixture strength.** Estimate team strength separately for attack and
   defence, so that each fixture yields an expected-goals-for and
   expected-goals-against for both sides.
2. **Stratified scoring-mechanism models.** Model each way a player earns FPL
   points separately (minutes/appearance, goals, assists, clean sheets and
   goals conceded, saves, bonus, cards, penalties, defensive contributions).
3. **Expected points (xP).** Combine the components into xP per player per
   gameweek, validated out-of-sample against actual FPL points and against
   baselines.

## Current status
The headline target is **xP given the player starts**: availability is left to the manager, and the
model predicts the points a starter scores. It is evaluated walk-forward on 2019-20..2024-25
(`docs/XP_EVALUATION.md`), on 2025-26 (`docs/XP_EVALUATION_sealed.md`; this season was opened earlier
with the availability-weighted model, see commit f546f26) and on the finished 2026-27 gameweeks
(`docs/XP_EVALUATION_live.md`; held out until the high-end work, since scored several times, so the
next clean test is the gameweeks still to be played). The high-end diagnosis, fixes and rejected ideas,
each with its rule written before the run, are in `docs/HIGH_END_PLAN.md` (results:
`docs/HIGH_END_RESULTS*.md`). Leakage audit: `docs/AUDIT_TESTS.md`. The mathematical
formulation is in a separate document. `analysis/13_xp_model.py --unconditional` gives the
availability-weighted xP instead.
Figures are in `figures/`.

## Model (analysis/)
Run order after the data pipeline: `pipeline/03_build_matches.py --source merged` ->
`analysis/10` (team-rating hyperparameters fitted before 2019-08-01) -> `analysis/17` (ratings with COVID and
drift terms chosen as of each season, written to data/ratings_eval) -> `analysis/01`, `06`, `12`
(with `XFPL_RATINGS=data/ratings_eval`) -> `analysis/13` (development folds), `analysis/13 --sealed`,
`analysis/13 --live` -> `analysis/15 --tag dev|sealed|live` -> `analysis/20 dev|sealed|live` (high-end
comparison) -> `analysis/18` (audit).

| script | purpose |
|---|---|
| `01_build_scorer_assist_dataset.py` | Understat time-decayed player histories (all six leagues) and same-match goal split |
| `02`-`05`, `07`, `08` | model-development experiments: Poisson ablation ladders, xG fractions, penalties, hierarchical shrinkage, minutes models, split-half empirical Bayes |
| `06_build_minutes_dataset.py` | one row per player per fixture (zeros included), minutes features as of the gameweek deadline; price and ownership from the previous gameweek |
| `10_fit_pi_params_pre2019.py` | team-rating hyperparameters fitted on matches before the first test season |
| `12_build_master.py` | master table: all outcomes, deadline-as-of Understat and FPL histories, team ratings |
| `13_xp_model.py` | component models with explicit parameter uncertainty and nested Monte Carlo assembly of xP; by default conditional on the player starting (minutes model of when a starter is taken off), with bonus ranked within each simulated match and a walk-forward spline calibration of goals and assists (`--slope` linear, `--no-cal` none) |
| `14_dc_model.py` | defensive-contribution model, validated on 2016-19 counts |
| `15_evaluate_xp.py` | calibration and uncertainty figures, the high end of the distribution, benchmarks |
| `16_covid_team_eval.py`, `17_build_asof_ratings.py` | closed-doors home advantage and online drift: evaluation and season-by-season selection |
| `18_audit.py` | leakage audit: truncation test at the gameweek deadline, single-feature scan, protocol and integrity checks |
| `19_compare_variants.py` | first high-end round: baseline vs within-match goal dependence vs calibration slope (historical record) |
| `20_high_end_fixes.py` | second high-end round: before / pre-registered fixes / final, by scoring component and position |

Decision time is the gameweek deadline. No feature uses information recorded after it.

## Repository contents
| path | purpose |
|---|---|
| `pipeline/01_fetch_understat.py` | fetch fixtures, team history, player totals, and per-match shots and rosters from Understat (EPL, La Liga, Bundesliga, Serie A, Ligue 1, RFPL) into a local JSON cache; logs every fetch |
| `pipeline/02_merge_understat.py` | splice freshly fetched seasons into the historical snapshot, keeping the legacy CSV schema |
| `pipeline/02b_clean_understat.py` | remove duplicate shot/roster records that Understat itself serves; every removal is logged |
| `pipeline/05_validate_data.py` | 100+ automated data-quality checks, written to `docs/DATA_QC.md` |
| `pipeline/06_fetch_fpl_live.py` | timestamped snapshot of the official FPL API: players (injury status/news/chance of playing, prices, ownership, set-piece orders), teams, fixtures, scoring rules, and per-player per-match history (ICT, BPS, xG, xA, xGC, defensive contribution) |
| `pipeline/07_fetch_fpl_history.py` | FPL history 2016-17 onward from the vaastav archive |
| `pipeline/08_build_fpl_tables.py` | unified tables in `data/fpl/` (player-match, player ids, snapshot history, teams, events, fixtures, rules) |
| `pipeline/09_validate_fpl.py` | FPL QC, plus recovery of the scoring rules per season from the data (`docs/FPL_QC.md`, `docs/FPL_RULES.md`) |
| `pipeline/10_crosswalk_fpl_understat.py` | FPL <-> Understat crosswalk: fixtures (kickoff + score), teams (vote, one-to-one), players (name + minutes assignment, consolidated into a persistent FPL `code` -> Understat `player_id` map); report in `docs/CROSSWALK_QC.md` |
| `model_v1/pi_rating_model.py` | dual attack/defence pi-rating model giving expected goals for and against (Constantinou and Fenton style updates, xG/goals blend) |
| `model_v1/01_build_dataset.py` | per-league join of results and npxG used by `pipeline/03_build_matches.py` |
| `pipeline/11_pi_ratings_to_fpl.py` | replays the model over all matches, records pre-match attack/defence ratings, and attaches them to every FPL fixture (played and upcoming) with expected goals, P(clean sheet) and goals-conceded distribution; report in `docs/PI_RATINGS_QC.md`. Options `--bcd`, `--drift-lr`, `--ha-lr` add the closed-doors and drift terms; fixture ratings are taken from snapshots at each gameweek deadline. |
| `pipeline/03_build_matches.py` | builds the cross-league match table `matches_unified.csv` (uses `model_v1/01_build_dataset.py`) |
| `pipeline/config.py` | paths, league list, column schemas |
| `docs/data_fetch_log.jsonl` | provenance: when each league-season was fetched, with package version |
| `docs/dedup_log.csv` | audit trail of rows removed by the cleaning step |
| `docs/DATA_QC.md` | latest QC report |
| `understat_MANIFEST.sha256` | checksums of the frozen historical snapshot (2014 to 2026-09-01) |

## Running
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python pipeline/run_all.py
```
The raw and derived data (`understat/`, `data/`) are not stored in git.

## Data notes
- Source: Understat via `understatapi`. Check Understat's terms of use before
  redistributing any data or derived work.
- The historical snapshot (2014 to 2026-09-01) predates this repository; its
  integrity is tracked by checksum. Rows from season 2026 onward are replaced
  by freshly fetched data on each refresh.
- Known upstream defect: some matches are served with every shot and roster row
  duplicated. The cleaning step removes these and logs them.
