# High-end over-prediction and yellow cards: diagnosis and pre-registered plan

Written 2026-10-12, before any of the runs below. Development folds only (2019-20..2024-25, starters).
2025-26 was opened earlier and is reported but not used to choose. 2026-27 GW1-6 is scored once, at the end.

## Diagnosis (from the existing dev / dev_slope / sealed_slope predictions)

| finding | evidence |
| --- | --- |
| Yellow cards: the GLM fits the mean count lambda, the simulator books a card when Poisson(lambda) > 0, i.e. P = 1 - exp(-lambda) | implied lambda 0.142 vs observed 0.144 (dev); predicted P 0.131 |
| Yellow cards: league-level rise in 2023-24 not captured | 2023-24 implied lambda 0.151 vs observed 0.162 |
| Bonus: players are simulated independently, so a fixture's bonus total is unconstrained | predicted bonus per fixture 5.4 to 7.3 by quintile, observed flat at ~6.1 |
| Bonus is the largest single part of the dev top-10 gap | top-10 gap 0.27 pts: bonus 0.10, goals 0.08, assists 0.04, clean sheets 0.04 |
| Epistemic draws on the log scale inflate means by exp(beta^2 sd^2 / 2), although the GLMs were fitted on posterior means | assists: simulated / point estimate 1.39 in the bottom decile, 1.01 in the top |
| Goal / assist rates over-predicted in a model's own top 1%, in-sample too | obs / pred: goals 0.82, assists 0.88 after the linear slope |
| Team lambda over-dispersed | pooled log E[goals] = a + b log lambda, b = 0.87 (se 0.03); 2025-26 b = 0.73 |
| Season / player drift | top-decile ratio 0.86 to 1.08 by season; one player is 27% of the 2025-26 top-10 gap |

Not pursued: a linear team-lambda recalibration. Every component GLM already has log lambda as a regressor,
so a + b log lambda is absorbed exactly by refitting; only the season-specific drift would change, and that is
not knowable before the season (walk-forward b for 2025-26 was 0.88, realised 0.73).

## Changes

Step 1, correctness fixes (adopted unless they move dev overall actual/xP away from 1 by more than 1 point or
move PIT central 50 / 80% coverage away from nominal by more than 2 points):
1. yellow and red cards: binary GLM with complementary log-log link (P = 1 - exp(-lambda), consistent with
   the simulator); yellow adds the league yellow rate per 90 over the 365 days before the deadline
2. bonus: joint allocation within the fixture. A Plackett-Luce model on the BPS order of starters, conditional
   on the simulated events; 3/2/1 to the top three in every draw, with BPS ties (competition ranking) and
   bonus taken by substitutes at their empirical training rates
3. mean-preserving epistemic draws (talent fractions and team lambda), and team-lambda noise shared within a
   match (the same draw is a team's attack and its opponent's defence)

Step 2, high-end calibration (on top of step 1): walk-forward calibration of the goal and assist index,
linear slope (current) vs natural cubic spline (knots at the 50/90/98% quantiles of the earlier pool).
Rule: choose the variant whose top-10-per-GW actual/xP is closest to 1, subject to overall actual/xP within
+-2% (else no worse than linear) and PIT central coverage within +-2 points; ties within 0.01 go to linear.

Step 3, recency: GLM-level walk-forward test on dev rows of 180-day vs 540-day shot share, quality and xA
share (and both). Adopt only if out-of-sample Poisson log-likelihood improves by more than 2 per added
parameter AND top-decile obs/pred is no worse; then a full simulation run.

Final: the chosen configuration is run once on 2025-26 (reported, flagged as opened) and once on 2026-27 GW1-6.

## Outcome (added after the runs)

Full tables: docs/HIGH_END_RESULTS.md (dev), HIGH_END_RESULTS_sealed.md, HIGH_END_RESULTS_live.md; figure
figures/fig9_high_end_fixes.png. Final configuration: `13_xp_model.py --curve` (step 1 + spline calibration).

- Step 1 adopted (dev overall actual/xP 0.978 -> 0.988; PIT 50/80% 0.492/0.790). Yellow cards actual/predicted
  1.098 -> 1.006; bonus per fixture now flat across quintiles (6.14-6.24 predicted vs 6.18-6.19 observed).
- Step 2: spline chosen by the rule (dev top-10 actual/xP 0.965 linear vs 0.977 spline).
- Step 3: rejected. 180-day fractions lower the out-of-sample log-likelihood (goals -0.6, assists -30.2);
  both windows together -1.1 / -0.7.

Deviations from the plan, all decided on development data and reported here:
1. Goals conceded is drawn once per team and match and thinned to each player's minutes, so team-mates share
   clean sheets. Found while checking bonus by position: with independent draws, defenders were credited with
   clean sheets their team-mates did not share, and their bonus was over-predicted by 13%.
2. Bonus ranking scores gained goals x position terms (BPS per goal is 12 / 18 / 24 for GKP-DEF / MID / FWD).
3. 2026-27 GW1-6 was scored twice: once with the planned configuration (top-10 actual/xP 1.038, overall 1.012)
   and once after deviations 1-2 (top-10 1.046, overall 1.013). Both are reported.

Investigated and not changed:
- Clean sheets are over-predicted in 2023-24 (actual/predicted 0.79) and 2024-25 (0.89). At the right mean,
  2023-24 had 10% fewer clean sheets than a Poisson implies, a break at the start of the season. Neither the
  league goal rate nor the recent residual (actual / predicted goals) over 60-365 days predicts it out of
  sample (both lower the log-likelihood; the residual mean-reverts).
- Team goals around a calibrated lambda are not over-dispersed (Pearson dispersion 1.00), so s_team mostly
  reflects the slope of 0.87, not true extra variance. This affects only the width of the uncertainty, and
  PIT coverage is already close to nominal, so it was left alone.
- Bonus by position still has a gap after the fixes: FWD actual/predicted 1.11, MID 0.95 (dev).
- The remaining 2025-26 top-10 gap (actual/xP 0.935) is mostly goals (0.25 pts per player) and is
  concentrated in a few premium players whose output fell (one player is 27% of it).

## Step 4: joint match simulation (pre-registered 2026-10-12, before any run)

Motivation: within a simulated draw, one team's goals were independent of the other team's goals conceded,
assists were independent of goals and team-mates' goals did not add up to a team score, so bonus was ranked on
matches that could not happen (forward bias in bonus: FWD actual/predicted 1.11, dev).

Change: in every draw, team goals = the opponent's full-match goals conceded (one score per match); goals are
allocated to the players on the pitch in proportion to their fitted scoring rates (open play + penalties),
with the remainder of the team rate going to substitutes / own goals; each goal is assisted with the training
rate of FPL assists per goal and the assister is drawn from team-mates other than the scorer in proportion to
their assist rates; goals conceded and clean sheets follow from the opponent's score (as now). Penalty misses,
cards, saves, own goals and defensive contributions stay as they are. Bonus is ranked on this match.

Rule (development folds): adopted if (a) FWD and DEF bonus actual/predicted are within +-5%, (b) top-10 per GW
actual/xP is no further from 1 than the current 0.982, (c) overall actual/xP within +-2% and PIT central
50 / 80% coverage within +-2 points of nominal. Expected goals and assists per player should be unchanged up
to simulation noise (checked, not part of the rule). 2025-26 and 2026-27 are reported only; 2026-27 has been
scored twice already and is no longer a clean test.

### Step 4 outcome: not adopted

Two implementations were run on the development folds (predictions kept in
data/analysis/before_fixes/xp_predictions_dev_curve_joint_rejected.parquet):
1. Team score = the opponent's conceded draw, goals shared out within it. Players' summed goal rates exceeded
   the team rate in 13-36% of draws, so allocation scaled them down: expected goals fell 2.6% and assists 1.7%,
   most at the top (top-1% expected goals 0.711 -> 0.668). Discarded as an artefact without evaluation.
2. Team score built bottom-up (player goals keep their Poisson means; substitutes / own goals add the rest of
   the team rate); it raises the opponent's conceded rate by 1.0-5.0% by season. Rule: (a) FWD bonus
   actual/predicted 1.105, DEF 0.906 -> FAIL; (b) top-10 1.007 (pass); (c) overall 0.996, PIT 0.494 / 0.792
   (pass). The top-10 gain partly came from expected assists falling at the top (scorer exclusion and the
   substitute share), another artefact.

Diagnosis: ranking the ACTUAL events with the bonus model, out of sample (2019-20..2024-25), gives FWD 1.068,
DEF 0.956, MID 0.995, GKP 1.022; a FWD who scores once gets 1.61 bonus against 1.49 predicted. So roughly
two-thirds of the forward gap is in the ranking model itself (BPS rewards actions that are not simulated and
cluster around forwards who score), not in the consistency of the simulated match. The code was reverted; the
reverted code reproduces the pre-joint predictions exactly (max |xP difference| = 0).

## Step 5: model BPS itself (pre-registered 2026-10-12, before any run)

Change: replace the Plackett-Luce bonus ranking with a model of each player's BPS score.
BPS_i = event part (OLS on the match events, by position: minutes 60+, goals x GKP-DEF / MID / FWD, assists,
clean sheet x GKP-DEF / MID, saves, goals conceded x GKP-DEF, penalties saved / missed, yellow, red, own goals)
+ player baseline (the player's shrunk lagged BPS per 90 x minutes / 90) + pre-match team strength
(log lambda for / against) + a residual drawn from the training residuals of the same position and 60+ status.
Fitted on training seasons only; coefficient uncertainty drawn from the OLS covariance. Simulated BPS is
rounded and ranked within the match; 3/2/1 by FPL's competition ranking, so ties arise from equal BPS;
substitutes take a slot at their training rate (as now).

Stage A (actual events, out of sample, development seasons, before any simulation): FWD and DEF bonus
actual/predicted within +-3%. If it fails, stop and report.
Stage B (full simulation, development folds): (a) FWD and DEF bonus actual/predicted within +-5%, (b) top-10
per GW actual/xP no further from 1 than 0.982, (c) overall actual/xP within +-2% and PIT 50 / 80% within
+-2 points. 2025-26 and 2026-27 reported only. Caveat: training data end in 2024-25, so any BPS rule change in
2025-26 cannot be learned for that season.

### Step 5 outcome: stopped at stage A

BPS regression (OLS, out-of-sample R^2 0.843-0.847, residual sd 4.2), actual events, development seasons:
bonus actual/predicted FWD 1.097, DEF 0.936 (MID 0.973, GKP 1.063) -> stage A FAILS (needs +-3%); worse than
the Plackett-Luce ranking on the same test (FWD 1.068, DEF 0.956). Not taken to simulation; code removed.

What the test shows: the BPS MEAN is about right for scorers (FWD one goal: residual -0.47 BPS, MID one goal:
0.00), yet forwards' bonus is still under-predicted. Out-of-sample residuals are mostly negative (BPS scale
drifts across seasons), and the forward gap swings by season (1.11, 1.15, 1.14, 1.05, 0.97, 1.17 for
2019-20..2024-25), as does the defender gap (0.90 .. 0.97, 0.85 in 2024-25).
Correction (checked against published rule histories): the BPS rules were unchanged from 2015-16 to 2023-24 and
changed in 2024-25, 2025-26 and 2026-27. Rule changes can therefore explain the 2024-25 values (FWD 1.17,
DEF 0.85) but not 2019-20..2023-24, where forwards are under-predicted in four of five seasons under fixed rules.
That part is a persistent bias of both bonus models, not drift.

## Step 6: rule-aware bonus + walk-forward position calibration (pre-registered 2026-10-12, before any run)

6b, rules: the BPS values of the SIMULATED events follow each season's published rules. Reference rules R0
(2015-16..2023-24); changes: 2024-25 GKP/DEF -4 BPS per goal conceded, penalty save 15 -> 9; 2025-26 penalty
goals 12 BPS for every position (was 12/18/24), penalty save 8 (+3 for the inside-box save); 2026-27 penalty
save 7 (+3). Save location is not simulated, so save BPS changes are not applied. In the ranking score each
row gets the offset kappa x (BPS under its season's rules - BPS under R0) for its events, in training and in
the simulation; kappa (ranking-score units per BPS) is estimated on R0 seasons from a ranking model whose
event part is the R0 rule BPS. The table is checked against the data first (per-season OLS of BPS on events).
6a, calibration: position offsets (DEF, MID, FWD; GKP = 0) added to the ranking score, estimated walk-forward
by maximum likelihood on the BPS order of earlier seasons, scored out of sample (each earlier season ranked by
the model fitted on the seasons before it). Earliest folds use 2017-18 and 2018-19 as the earlier seasons.

Stage A (actual events, out of sample, development seasons): FWD and DEF bonus actual/predicted within +-3%.
Stage B (simulation, development folds): (a) FWD and DEF bonus within +-5%, (b) top-10 per GW actual/xP no
further from 1 than 0.982, (c) overall within +-2%, PIT 50 / 80% within +-2 points. Each of 6b and 6a is kept
only if it moves FWD and DEF towards 1 in stage A. 2025-26 and 2026-27 reported only.

### Step 6 outcome: not adopted

Rule table checked against per-season OLS of BPS on events (coefficient changes: GKP/DEF per goal conceded
+0.9 -> -3.0 in 2024-25; penalty save 14.5 -> 8.9 in 2024-25; FWD penalty goal 24.2 -> 10.3 and MID 20.0 ->
14.7 in 2025-26), consistent with the published changes. kappa = 0.30-0.31 ranking-score units per BPS.

Stage A (actual events, out of sample), bonus actual/predicted:

| | base | + rules (6b) | + rules + position offsets (6a) |
| --- | --- | --- | --- |
| dev FWD / DEF | 1.059 / 0.938 | 1.054 / 0.974 | 1.050 / 0.970 |
| 2024-25 DEF | 0.854 | | 1.013 |
| 2025-26 DEF / MID | 0.875 / 1.082 | 0.960 / 0.981 | 0.959 / 0.981 |
| 2026-27 DEF / MID | 0.865 / 1.121 | 0.910 / 1.044 | 0.926 / 1.062 |

6b moves FWD and DEF towards 1 (kept for stage B); 6a moves DEF away (rejected). Stage A fails overall
(FWD 1.054 > 1.03): the remaining forward gap is in seasons with unchanged rules.

Stage B was run for information (rules only): dev top-10 0.980 (adopted 0.982, rule (b) fails); bonus by
position in the simulation: dev DEF 0.996, FWD 1.099, MID 0.926; 2025-26 DEF 1.131, MID 0.901; 2026-27 DEF
1.133, MID 0.907 (adopted: 0.99-1.00 / 1.01). The rule weights fix the ranking of actual events, but in the
simulation they expose an opposite error of similar size in the simulated defenders' events, which had been
cancelling the outdated weights. Not adopted; code reverted (reproduces the adopted predictions exactly).
Predictions kept in data/analysis/before_fixes/*_curve_rules_rejected.parquet.
