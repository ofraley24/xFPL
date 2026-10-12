# Audit of the xP pipeline (2026-10-12)

13/13 checks passed.

| check                                                                                | result   | detail                                                                                                                                                                                         |
|:-------------------------------------------------------------------------------------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| T1a Understat form recomputed from raw pre-kickoff rows                              | PASS     | 3610 comparisons, max rel err 2.49e-15                                                                                                                                                         |
| T1b FPL minutes features recomputed from earlier rows                                | PASS     | 398 players x 8 features, mismatches 0                                                                                                                                                         |
| T1c FPL event lags (L365) recomputed from earlier rows                               | PASS     | 400 rows x 4 features, mismatches 0                                                                                                                                                            |
| T1d team lambda reproduced by a replay truncated at the gameweek deadline            | PASS     | 10 fixtures, max |d lambda| = 0.00e+00                                                                                                                                                         |
| T2 no single feature near-perfectly predicts the outcome (AUC < 0.95)                | PASS     | max AUC played 0.764 (st_full5)                                                                                                                                                                |
| T3 dev predictions contain only seasons 2019-20..2024-25                             | PASS     | seasons: ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']                                                                                                                    |
| T3 team-rating hyperparameters fitted before the first test season                   | PASS     | cutoff 2019-08-01, 10354 matches                                                                                                                                                               |
| T3 drift settings chosen per season from earlier data only (selection table present) | PASS     | 2019-20: d=0.001, h=0.0; 2020-21: d=0.002, h=0.0005; 2021-22: d=0.004, h=0.001; 2022-23: d=0.004, h=0.0005; 2023-24: d=0.002, h=0.0005; 2024-25: d=0.004, h=0.0005; 2025-26: d=0.001, h=0.0005 |
| T3 xP model reads only seasons <= 2024-25 and fits each fold on earlier seasons      | PASS     | static check of 13_xp_model.py: load filter, fold split, EB estimated on seasons < test year                                                                                                   |
| T4 master key (season, element, fixture) unique                                      | PASS     | 257,252 rows                                                                                                                                                                                   |
| T4 every master row has team ratings                                                 | PASS     |                                                                                                                                                                                                |
| T3 price and ownership are taken from the previous gameweek                          | PASS     | value differs from the same-gameweek value on 8.5% of rows                                                                                                                                     |
| T4 the simulator's scoring rules reproduce actual total_points from realised events  | PASS     | 100.0000% of 50,160 rows                                                                                                                                                                       |

## Single-feature AUCs (T2)

| feature         |   AUC played |   AUC 5+ pts |
|:----------------|-------------:|-------------:|
| st_full5        |        0.764 |        0.521 |
| st_mean5        |        0.747 |        0.525 |
| f_full5         |        0.741 |        0.525 |
| ewm3            |        0.682 |        0.533 |
| m3              |        0.68  |        0.523 |
| m5              |        0.676 |        0.526 |
| m10             |        0.67  |        0.533 |
| std_mpf         |        0.664 |        0.53  |
| m1              |        0.663 |        0.515 |
| m5_rank_pos     |        0.65  |        0.535 |
| f_long5         |        0.646 |        0.526 |
| L180_log        |        0.646 |        0.551 |
| st_n10          |        0.645 |        0.533 |
| pos_MID         |        0.636 |        0.557 |
| pos_DEF         |        0.623 |        0.531 |
| n_active_pos    |        0.621 |        0.542 |
| L540_log        |        0.616 |        0.554 |
| price_gap_top   |        0.611 |        0.555 |
| prev_season_mpf |        0.608 |        0.53  |
| own_share_team  |        0.604 |        0.588 |
| selected_log    |        0.594 |        0.607 |
| value           |        0.592 |        0.575 |
| price_rank      |        0.572 |        0.575 |
| st_since        |        0.566 |        0.507 |
| five_subs       |        0.564 |        0.507 |
| pos_GKP         |        0.562 |        0.503 |
| pos_FWD         |        0.55  |        0.523 |
| f_play5         |        0.545 |        0.516 |
| career_apps     |        0.542 |        0.545 |
| n_prior_rows    |        0.534 |        0.538 |
| total_goals     |        0.527 |        0.505 |
| bcd             |        0.521 |        0.506 |
| fix_since_app   |        0.512 |        0.504 |
| absent_streak   |        0.512 |        0.504 |
| us_days_log     |        0.51  |        0.506 |
| days_since_app  |        0.51  |        0.505 |
| supremacy       |        0.509 |        0.628 |
| season_progress |        0.507 |        0.505 |
| home            |        0.504 |        0.529 |
| rest_days       |        0.503 |        0.505 |
| opp_rest_days   |        0.502 |        0.501 |
| next_gap_days   |        0.502 |        0.502 |
| fix_in_event    |        0.502 |        0.505 |
