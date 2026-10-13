# Plan: xFPL transfer-planning website

Answers the brief in `docs/WEBSITE_BRIEF.md`. Written 2026-10-13.

## Decisions taken
| question | decision |
|---|---|
| audience | personal use only for now; public release (and the FPL / Understat terms review it needs) is deferred |
| target shown on the site | **xP given that the player starts**, the model's headline target. P(start) is deferred to a later version |
| horizon | backtest forecasts 1-8 gameweeks ahead before the site relies on them |
| repositories | research stays here (`xFPL`); the site lives in a separate repository (`xfpl-web`) that reads a published data file |
| FPL snapshots | scheduled from 2026-10-13: full snapshot 08:00, light snapshot (bootstrap + fixtures) 20:00, `pipeline/run_fpl_snapshot.sh` via launchd |

## Facts from the code that shape the plan
- `analysis/13_xp_model.py` trains each fold on earlier seasons only (`d.season < ts`), so every component
  (GLMs, minutes ensemble, bonus, calibration spline) is **fitted once per season**. Before a deadline only
  features and the simulation need to run.
- `analysis/06` and `12` build rows for **finished fixtures only**. Forecasting needs the same features, frozen
  at the current deadline, for unplayed fixtures. Upcoming team ratings already exist
  (`pipeline/11`, `is_prediction = True`), rated from the current state.
- Every evaluation so far is 1 gameweek ahead (features as of the deadline of the fixture's own gameweek).
  Accuracy at longer horizons is unknown.
- The correlated-epistemic range method is validated on start-conditional player-season totals
  (sd(z) 1.01 dev / 1.00 sealed; 80% and 95% coverage on target), the same target as the site, but only for
  totals over >= 10 fixtures. Short windows are untested.

## Architecture
```
Mac (launchd)                                         xfpl-web (static)
 FPL snapshot (08:00 full, 20:00 light) ─┐
 Understat refresh (after each GW)  ─────┼─► predict step ─► QC gates ─► forecast_<season>_gw<k>.json ─► browser:
 fitted season model (saved once)  ──────┘   (features frozen          (archived, never overwritten)     range sums,
                                              at the deadline)                                           filters, compare
```
- Runs on this Mac: `data/` is not in git and the pipeline depends on the local Understat snapshot.
- No backend in v1. The browser sums over the chosen gameweeks.
- Each deadline's file is archived unchanged as the pre-registered forecast and scored after the gameweek:
  the clean prospective test the README says is still owed.

## Work packages

### WP1. Separate fitting from prediction (research repo). DONE 2026-10-13
- `13_xp_model.py` now has `fit_season` (once per season; saved to `data/models/xp_model_<tag>_<season>.pkl`
  with a hash of the training data and the code commit) and `predict_rows` (any rows with the master
  schema; outcomes optional). Regression test `21_regression_check.py`: dev, sealed and live outputs
  (predictions and coefficients) are identical to the frozen pre-split baseline.
- Horizon rows without new feature code (`22_build_horizon_rows.py`): the master table already holds each
  player's state at every deadline (the first row at or after it, the double-gameweek rule generalised).
  Counts are copied, the clock and decayed sums move on to the target kickoff, Understat form is recomputed
  with 12's own as-of join, and team ratings are frozen at the deadline (`pipeline/11 --horizon`, via
  `17 --reuse-selection --horizon 8`). Rows with h = 1 reproduce `master.parquet` exactly in every season.
  Truncation test for h >= 2: `23_audit_horizon.py`, all PASS (`docs/AUDIT_HORIZON.md`).
- Live forecast rows (WP3) reuse the same construction: the anchor is then a placeholder row for the next
  gameweek built by 06/12 from the snapshot, and the targets are the schedule's unplayed fixtures.
- Double gameweeks: every fixture gets the deadline features (the existing rule). Blank gameweeks: no rows.
  Fixtures without a gameweek (postponed) are listed as unscheduled and not forecast.

### WP2. Horizon backtest (research repo, before the site uses horizons > 1)
Pre-registration: `docs/HORIZON_PLAN.md` (rule awaiting confirmation). Predictions:
`24_predict_horizons.py` (saved fold models; h = 1 reproduces the walk-forward predictions exactly).
Design, to be fixed in writing before the run, in the style of `docs/HIGH_END_PLAN.md`:
- **Rows:** for each deadline t of each test season and each h = 1..8, every player who **started** a
  fixture in gameweek t + h, with player features and team ratings both as of deadline t. h = 1 is the
  current evaluation.
- **Ratings frozen at t:** extend `pipeline/11` to emit, for each deadline t, ratings and expected goals
  for all later fixtures from the state at t (today it rates each fixture from its own deadline only).
- **Model:** the fitted fold models are unchanged; only the test rows change. The calibration spline
  is fitted on h = 1 out-of-sample rows and applied as is, which is what the live forecast will do.
- **Metrics by h:** bias, MAE, RMSE, within-gameweek rank correlation, calibration by xP decile, q10-q90
  and q25-q75 coverage, P(10+) calibration (the high end), and the same by position.
- **Window metrics** (what the site shows): for windows of L = 1..8 gameweeks starting at t + 1, summed over
  the player's started fixtures in the window. sd(z) and 80% / 95% coverage of the correlated-epistemic
  interval, against the aleatoric-only and independent versions.
- **Decomposition (optional):** freeze only player features, or only team ratings, to see which staleness
  costs more.
- **Leakage:** extend `18_audit.py`'s truncation test: rows for (t, h) must be identical after deleting
  every record later than deadline t.
- **Decision rule (proposed, confirm before running):** horizon h is supported if bias is within ±0.10
  points per row and 80% coverage lies in [0.77, 0.83]. The site's default range is the supported horizons,
  with longer horizons shown with a warning.
- **Order:** dev seasons 2019-20..2024-25 first; rule fixed; then sealed 2025-26 once.
- **Cost:** about 8 times the rows of the current evaluation, all in the simulation step.

### WP3. Forecast mode and export (research repo)
- `predict --forecast`: the current deadline's rows for every player x upcoming fixture within the
  horizon, simulated with the season's fitted model.
- QC gates (nothing is published on failure): player and fixture counts against the snapshot, no missing
  features, xP within a sane range, mean xP close to the recent live mean, new players without an Understat
  mapping flagged (they fall back to the FPL-only features, as now).
- Export `forecast_<season>_gw<k>_<utc>.json` (schema versioned) to the web repo's data folder and to an
  archive under `data/forecasts/`:
  - per player: FPL id and code, name, team, position, price, status, news, chance of playing (shown,
    **not** used by the model)
  - per player x fixture: gameweek, fixture id, opponent, home, kickoff, `xp`, `var_ale` (= var_total -
    var_epi), `sd_epi`, q10 / q25 / q50 / q75 / q90, P(5+), P(10+), component means (`e_app`, `e_goals`,
    `e_assists`, `e_cs`, `e_gc_pts`, `e_saves_pts`, `e_bonus`, `e_dc`, cards, ...), and the horizon h.
  - header: model version and fit hash, snapshot timestamp, deadline, supported horizon from WP2.
- After each gameweek, score the archived forecast against actual points (started players), adding to a
  running prospective record.

### WP4. Refresh automation
- Done: FPL snapshot twice a day (see Decisions), logs in `data/logs/fpl_snapshot.log`.
- Next: Understat refresh after each gameweek, then about 2 h before each deadline (read from the
  snapshot's `events`) snapshot → features → predict → QC → export. A run missed while the Mac sleeps
  fires on wake. If it fires after the deadline, it must not publish (decision time is the deadline).

### WP5. The site (`xfpl-web`)
- **Stack:** Vite + React + TypeScript, a table library (TanStack Table) and a small chart library. Static
  build, run locally for now (no hosting needed while it is personal). Forecast files are gitignored in the
  web repo.
- **v1:**
  - gameweek range picker (default: the supported horizon)
  - sortable table: range xP given starts, xP per £m, number of fixtures in the range (doubles and blanks
    visible), 80% interval
  - filters: position, team, price, minimum fixtures
  - per-fixture heat strip (opponent, home / away, xP)
  - player page: points distribution per fixture, component breakdown, FPL status and news
  - compare 2-4 players
  - toggle to exclude a player-fixture (e.g. a known injury), which drops it from the sum. That is
    consistent with the start-conditional target and needs no P(start) model
  - "about the model" and prospective track record pages
- **Range maths:** xP_range = Σ xp. Var_range = Σ var_ale + (Σ sd_epi)^2 (epistemic fully correlated
  within a player, as validated). The interval is a normal approximation unless WP2's window metrics show it
  fails for short windows, in which case the export adds joint draws indexed by parameter draw j (keeping
  the correlation) and the browser takes percentiles.
- The site states plainly that every figure assumes the player starts each fixture in the range.

### Later versions
- P(start): FPL chance-of-playing flags (history accruing from the snapshots) plus a model on the minutes
  features, with a user override. Expected points = P(start) x xP(start) + substitute appearances.
- Import an FPL squad by team id (needs a small proxy: the FPL API has no CORS), saved squads, and a transfer
  optimiser (integer program: budget, 3 per club, formation).
- Hosting and a public release, after a terms review (FPL API, Understat; publish model outputs only).

## Order of work
1. WP1 (fit / predict split with the regression test).
2. WP2 horizon backtest: design note with the rule, dev run, then sealed.
3. WP3 forecast and export, then WP4 automation.
4. WP5 site v1, built against an exported file.
Ask before creating the `xfpl-web` repository on GitHub or pushing either repository.
