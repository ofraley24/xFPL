# Availability and the full xP model: pre-registered plan

Written 2026-10-17, before any run. The site shows xP GIVEN A START by default; a toggle switches to the full
expectation over the three ways a fixture can go for a player:

    E[points] = P(start) x E[points | start] + P(sub) x E[points | sub] + P(no appearance) x 0

E[points | start] is the existing model. This plan adds P(start), P(sub) and E[points | sub], and the full
points distribution, the mixture P(start) PMF|start + P(sub) PMF|sub + P(none) delta_0.

## A. Availability model

- Rows: every registered player-fixture in master.parquet (zero-minute rows included). Class: start
  (started = 1), sub (minutes > 0, not started), none (0 minutes).
- Features, all as of the gameweek deadline: starts in the last 10 fixtures, full 90s and mean starts over the
  last 5, fixtures since the last start; mean minutes over the last 1 / 3 / 5 / 10 fixtures, shares played /
  60+ / 90 over the last 5, exponentially weighted minutes, absence streak, fixtures and days since the last
  appearance, minutes per fixture this and last season, career appearances, rows of history; squad position
  (price rank and gap to the team's most expensive player in the position, ownership share in the team,
  minutes rank in the position, active players in the position); price, ownership, home, rest days, five-sub
  era, season progress, fixtures in the gameweek, position. Counts, days, price and ownership log-transformed;
  missing values set to 0 with an indicator.
- Model: multinomial logistic regression (scikit-learn), standardised inputs, cubic splines (4 knots) on six
  key features (starts in the last 10, mean minutes over the last 5, absence streak, days since the last
  appearance, price rank, weighted minutes), linear terms for the rest, L2 penalty C = 1.
- Walk-forward: development seasons 2019-20..2024-25, each fitted on every earlier season from 2016-17;
  2025-26 and 2026-27 reported only. A live model is fitted on every season before the current one.
- Baseline: the player's own recent shares, start = starts in the last 10 fixtures / fixtures available (up
  to 10), appearance = share of the last 5 fixtures played (raised to at least the start share), sub =
  appearance - start, none = 1 - appearance; the three shares smoothed towards the position's training rates
  with the weight of one fixture. (Clarified before any run: the master table has no "mean starts over the
  last 5"; st_mean5 is mean minutes per start.)
- Rule: adopted if (i) multiclass log loss is below the baseline's in every development season, and
  (ii) on the pooled development rows, for P(start) and for P(sub), the mean prediction is within 0.01 of the
  observed rate and the calibration slope (logistic regression of the outcome on logit p) is within
  [0.9, 1.1].
- Horizons (report only): the same models scored on the all-player horizon rows (features frozen at the
  deadline, h = 1..8): log loss and calibration in the large by h.

## B. Points given a substitute appearance

- The existing simulator, unchanged, with a substitute-minutes distribution in place of the start-minutes
  model: bucket probabilities and within-bucket minutes from training substitute appearances of the same
  position, from training seasons of the same five-substitute era when at least one exists, else all.
- Check (report only; decided to use the current simulator for now, docs/FUTURE_WORK.md): on development
  substitute appearances, actual/predicted points overall and by position, and actual/predicted minutes.
  Flagged in the report if outside +-15%.

## C. The full expectation

- Development seasons, rows of players with minutes in their previous 10 fixtures (97% of appearances;
  players outside are predicted 0 and the points they scored are reported). xP | start and xP | sub for
  non-starters are simulated as in the live forecast: each team plays a probable XI ranked by starts in the
  last 10, recent minutes and price (as of the deadline), and every other player is simulated in his own copy
  of the match. For speed the backtest uses J = 10 parameter draws (live: 40); means are unaffected.
- Report: actual/predicted points overall, by P(start) decile and by position; within-gameweek rank
  correlation over all players; against a form baseline (mean points over the last 5 fixtures, zeros
  included). Sanity rule: overall actual/predicted within +-5%.

## D. Live availability flags (heuristic, not validated)

FPL status injured / suspended / not available: P(start) = P(sub) = 0. Chance of playing 0 / 25 / 50 / 75%:
P(start) and P(sub) multiplied by chance / 100, the remainder to no appearance. Over a horizon (clarified
before live use): FPL's chance of playing describes the next gameweek only, so the scaling applies to the next
gameweek; a return date in FPL's news ("Expected back 25 Oct", "Suspended until 1 Nov") sets P(start) =
P(sub) = 0 for every fixture kicking off before it; injured / suspended players without a date are 0 for the
next gameweek and modelled normally after it. No flag history exists before
October 2026, so this rule cannot be tested yet; it is replaced by a fitted adjustment once the snapshots
allow (docs/FUTURE_WORK.md).

## E. Horizon recalibration (added 2026-10-17 after the horizon report of A, before any recalibration run)

The report of A showed (after resetting 'days since last appearance' to its deadline value): calibration in
the large holds at every horizon (P(start) 0.322-0.328 predicted vs 0.322 observed) but probabilities are too
extreme beyond next week (calibration slope of P(start) 0.86 at h = 2 falling to 0.60 at h = 8; P(sub) 0.89 to
0.70). The model assumes the deadline state persists.

Change: for each horizon h >= 2, a multinomial logistic recalibration of the fold model's output, inputs
log(P(start)/P(none)) and log(P(sub)/P(none)) (vector scaling, 6 coefficients per h), fitted by maximum
likelihood on the out-of-sample horizon predictions of earlier seasons only (from 2019-20, the first season
with horizon rows); h = 1 is unchanged. Live: fitted on every season up to 2025-26.
Rule (development seasons 2020-21..2024-25, the seasons with at least one earlier season): adopted if log loss
improves at every h >= 2 and the calibration slopes of P(start) and P(sub) are within [0.9, 1.1] at every
h >= 2.
