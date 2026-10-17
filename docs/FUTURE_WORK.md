# Future work

Open items, grouped by area. Each says what is wrong or missing and the evidence for it, so it can be
prioritised later. Results referred to are in docs/HIGH_END_PLAN.md, docs/HORIZON_RESULTS.md and the
evaluation reports.

## Not yet completed (availability work, started 2026-10-17)
- **Finish the full-mixture backtest.** `31_full_xp.py` has scored 2024-25 only (actual/predicted 0.983,
  mixture PIT 0.507 / 0.809, within-GW rank correlation over all players 0.60 vs 0.53 for form); 2019-20 to
  2023-24 are running. Then write docs/FULL_XP_RESULTS.md from all six seasons and check the sanity rule.
- **Website toggle (xfpl-web).** Read forecast schema 3 (`p_start`, `p_sub`, `xp_sub`, `full`); a toggle between
  "xP · if starts" (default) and the full model on every view that ranks or sums xP; the Risk & upside page to
  show the full mixture distribution (the spike at 0 for rotation risks); labels that the availability flags
  are a heuristic and that intervals summed over gameweeks are not validated; update src/types.ts.
- **Evidence for the site.** Export the availability and full-xP validation (calibration, horizons, ranking)
  for the method pages (28_export_evidence.py), once the backtest is complete.
- **Track record of the full model.** Score archived schema-3 forecasts after each gameweek with the full
  expectation as well as the start-conditional one (29_track_record.py).
- **Observations from the 2024-25 backtest to confirm on all seasons:** predicted substitute minutes 2.4
  minutes too long (20.9 vs 18.6); the lowest P(start) deciles over-predicted by about 20% (small absolute
  values); defenders and goalkeepers over-predicted by 7-10% in the full model (consistent with the known
  2024-25 clean-sheet drift).
- **Long-horizon availability.** After recalibration, mean P(start) runs 2-3% high at h = 6-8 (0.324 vs 0.314
  at h = 8).

## Availability and substitutes
- **A dedicated model for substitute appearances.** xP given a substitute appearance currently reuses the
  start simulation with a substitute-minutes distribution. Substitutes enter in particular game states
  (chasing a game, protecting a lead), and the per-minute rates were estimated mostly on starters' minutes.
  A remodel would condition on game state and minute of entry, and simulate the substitute as joining a match
  already in progress.
- **Substitutes in the bonus ranking.** Substitutes are a probabilistic slot in the starters' bonus ranking;
  a substitute's own bonus comes from simulating him in place of a starter with short minutes.
- **Substitute minutes by player.** The substitute-minutes distribution is by position (and rule era) only;
  some players are habitually introduced early or late.
- **Availability correlated across gameweeks.** An injury or suspension usually spans several gameweeks, so
  the spread of a multi-gameweek total is wider than independent weeks imply. Expected values are unaffected;
  intervals over a horizon need a joint model of availability (e.g. a Markov chain over available / doubtful
  / out).
- **Double gameweeks share availability.** Both fixtures of a double gameweek depend on the same fitness and
  selection; the same correlation as above, within one gameweek.
- **Calibrate FPL's availability flags.** The flag rule (status and chance of playing) is a heuristic because
  no flag history exists before October 2026. Once the deadline snapshots cover enough gameweeks, estimate
  the effect of each flag on starting and appearing.
- **Team news and press conferences.** Not used; the site's toggle back to "xP if he starts" covers news the
  model cannot see.

## Points given a start
- **Forwards' bonus.** Under-predicted by 5 to 10% in seasons with unchanged BPS rules, by both bonus models
  tried; a joint match simulation and a BPS regression did not remove it.
- **BPS rule changes (2024-25, 2025-26, 2026-27).** Learned only as training data accumulate. Applying the
  published rule weights fixes the ranking of real events but exposes an opposite error in the simulated
  defenders' events, which has to be found first (audit the simulated defensive events season by season).
- **Clean-sheet drift.** Over-predicted in 2023-24 (obs/pred 0.79) and 2024-25 (0.89); not predictable from
  earlier data with the league goal rate or the recent residual.
- **Premium attackers in decline.** The 2025-26 top-10 gap is mostly goals of a few premium players whose
  output fell (one player is 27% of it). Candidate signals: age, recent minutes load.
- **Players who changed club.** Over-predicted by about 7% in gameweeks 1-10 (0.93 actual/xP; goals 0.84),
  converging by mid-season. Down-weight history from the previous club in the talent shares.
- **Promoted teams' starting rating.** New teams start at the league average and returning teams resume a
  stale rating; promoted sides scored 0.89 of predicted goals in gameweeks 1-5 (weak evidence, 120
  team-matches). A walk-forward prior from past promoted teams.
- **Mid-season refit.** Refit the coefficients after gameweeks 5 and 19 with the new season's data. Expected
  to be small; cheap to test.
- **Joint team simulation.** Team-mates' goals and assists are not tied to a team score. Single-player xP is
  unaffected, but covariances between players (stacks, captaincy against an opposing keeper) are not usable.
- **Model discrepancy in the uncertainty.** Epistemic variance covers parameter uncertainty within the model,
  not error in its form; size a discrepancy term from out-of-sample calibration error.
- **Minutes given a start.** Only modestly better than simple baselines.

## Evaluation and benchmarks
- **A clean FPL benchmark.** Compare against FPL's ep_next from the deadline snapshots once enough gameweeks
  have been collected; the archived xP field contains post-match information.
- **A clean test set.** 2026-27 gameweeks 1-6 have been scored several times; score the remaining gameweeks
  once, at the end of a fixed period.
- **Intervals over horizons.** Per-fixture quantiles are validated; sums over gameweeks are not
  (docs/HORIZON_RESULTS.md).
