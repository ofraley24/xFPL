# Horizon backtest: pre-registration

Status: rule FIXED 2026-10-13, thresholds confirmed by the owner, committed before any horizon prediction
was scored. Predictions (`24_predict_horizons.py`, `data/analysis/xp_horizon.parquet`) were generated but not
compared with outcomes before this commit.

Question: how much worse is xP (given the player starts) when it is predicted h = 1..8 gameweeks ahead,
with every feature and the team ratings frozen at the earlier deadline? The answer sets the default and
maximum range on the website.

## Rows
- `analysis/22_build_horizon_rows.py` -> `data/analysis/horizon_rows.parquet`: for every deadline t of
  2019-20..2025-26 and h = 1..8, each player's fixture in gameweek t + h - 1 with features as of deadline t
  (construction and limitations in the script header). Rows with h = 1 reproduce `master.parquet` exactly
  (checked in the script).
- Team ratings frozen at deadline t: `17_build_asof_ratings.py --reuse-selection --horizon 8`
  (`fpl_team_fixture_ratings_asof.csv`; h = 1 reproduces the evaluation ratings exactly).
- Leakage: `analysis/23_audit_horizon.py` recomputes the features of 400 random rows with h >= 2 from records
  dated before the earlier deadline (`docs/AUDIT_HORIZON.md`, all PASS on 2026-10-13).
- Known limitations: fixture context (rest days, fixtures in the gameweek) is from the final fixture list,
  so a fixture moved after deadline t sits where it was played. Squad-competition features are computed
  among the players with a target row, so players who left the club before the target fixture drop out.
  Both make the backtest slightly optimistic about long horizons.

## Model
- The saved fold models (`data/models/xp_model_dev_<season>.pkl`, `xp_model_sealed_2025-26.pkl`): every
  component trained on earlier seasons, calibration spline fitted on earlier out-of-sample h = 1 rows,
  applied unchanged at every h (what the live forecast will do).
- Population: rows whose target fixture the player started (the model's target, xP given a start).
- Consistency check: h = 1 rows predicted with the fold's simulation seed must reproduce
  `xp_predictions_dev.parquet` / `xp_predictions_sealed.parquet`.

## Metrics
Per h, on the BALANCED PANEL: target fixtures in gameweek >= 8 (so every h = 1..8 exists) and player-fixtures
present at every h. Also reported on all rows.
- bias (mean actual - xP), MAE, RMSE, rank correlation of xP and points within each gameweek
- calibration by xP decile; q10-q90 and q25-q75 coverage per row; P(10+) calibration (high end); by position
- windows (what the site shows): for each deadline t and L = 1..8, each player's sum over his started
  fixtures in gameweeks t..t+L-1; standardised error z with variance = sum of aleatoric variances +
  (sum of epistemic sds)^2; sd(z), 80% and 95% coverage; also with aleatoric-only and independent epistemic
- decomposition (descriptive): h = 8 rows with only player features frozen, or only ratings frozen

## Decision rule (FIXED)
On the dev seasons (2019-20..2024-25), balanced panel:
1. Horizon h is SUPPORTED if |bias_h| <= 0.10 points per row and the 80% interval of windows of length
   L = h covers between 0.77 and 0.83 of outcomes.
   - bias_h = mean(actual - xP) over the balanced panel's rows at horizon h, dev seasons pooled.
   - windows of length L: every (season, deadline t, player) with at least one started fixture in gameweeks
     t..t+L-1, each fixture predicted from deadline t, for every t with t + L - 1 <= the season's last
     gameweek (complete windows only); dev seasons pooled. 80% interval: |z| <= 1.2816 with
     z = (sum actual - sum xP) / sqrt(sum var_aleatoric + (sum sd_epistemic)^2).
2. The site's default range is the largest h such that every horizon 1..h is supported. Longer ranges stay
   available with a warning.
3. If the window intervals fail but bias passes, the site shows sums without intervals beyond the supported
   length, and joint draws (plan, WP5) are the follow-up.
Then 2025-26 (sealed) is scored once with the same rule and reported whether it agrees or not.
