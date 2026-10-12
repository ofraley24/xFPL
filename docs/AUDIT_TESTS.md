# Audit of the xP pipeline (2026-10-12)

13/13 checks passed.

| check                                                                                | result   | detail                                                                                                                                                                                         |
|:-------------------------------------------------------------------------------------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| T1a Understat form recomputed from raw pre-kickoff rows                              | PASS     | 3610 comparisons, max rel err 2.49e-15                                                                                                                                                         |
| T1b FPL minutes features recomputed from earlier rows                                | PASS     | 398 players x 8 features, mismatches 0                                                                                                                                                         |
| T1c FPL event lags (L365) recomputed from earlier rows                               | PASS     | 400 rows x 4 features, mismatches 0                                                                                                                                                            |
| T1d team lambda reproduced by a replay truncated at the gameweek deadline            | PASS     | 10 fixtures, max |d lambda| = 0.00e+00                                                                                                                                                         |
| T2 no single feature near-perfectly predicts the outcome (AUC < 0.95)                | PASS     | max AUC played 0.828 (m3)                                                                                                                                                                      |
| T3 dev predictions contain only seasons 2019-20..2024-25                             | PASS     | seasons: ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']                                                                                                                    |
| T3 team-rating hyperparameters fitted before the first test season                   | PASS     | cutoff 2019-08-01, 10354 matches                                                                                                                                                               |
| T3 drift settings chosen per season from earlier data only (selection table present) | PASS     | 2019-20: d=0.001, h=0.0; 2020-21: d=0.002, h=0.0005; 2021-22: d=0.004, h=0.001; 2022-23: d=0.004, h=0.0005; 2023-24: d=0.002, h=0.0005; 2024-25: d=0.004, h=0.0005; 2025-26: d=0.001, h=0.0005 |
| T3 xP model reads only seasons <= 2024-25 and fits each fold on earlier seasons      | PASS     | static check of 13_xp_model.py: load filter, fold split, EB estimated on seasons < test year                                                                                                   |
| T4 master key (season, element, fixture) unique                                      | PASS     | 257,252 rows                                                                                                                                                                                   |
| T4 every master row has team ratings                                                 | PASS     |                                                                                                                                                                                                |
| T3 price and ownership are taken from the previous gameweek                          | PASS     | value differs from the same-gameweek value on 8.5% of rows                                                                                                                                     |
| T4 the simulator's scoring rules reproduce actual total_points from realised events  | PASS     | 100.0000% of 101,821 rows                                                                                                                                                                      |

## Single-feature AUCs (T2)

| feature         |   AUC played |   AUC 5+ pts |
|:----------------|-------------:|-------------:|
| m3              |        0.828 |        0.717 |
| m1              |        0.823 |        0.706 |
| ewm3            |        0.811 |        0.72  |
| f_play5         |        0.81  |        0.683 |
| m5              |        0.809 |        0.712 |
| absent_streak   |        0.795 |        0.656 |
| fix_since_app   |        0.795 |        0.656 |
| f_long5         |        0.785 |        0.704 |
| days_since_app  |        0.777 |        0.644 |
| m10             |        0.774 |        0.701 |
| std_mpf         |        0.771 |        0.697 |
| us_days_log     |        0.767 |        0.64  |
| f_full5         |        0.738 |        0.685 |
| m5_rank_pos     |        0.734 |        0.68  |
| L180_log        |        0.721 |        0.683 |
| own_share_team  |        0.676 |        0.685 |
| selected_log    |        0.669 |        0.694 |
| L540_log        |        0.658 |        0.647 |
| prev_season_mpf |        0.631 |        0.616 |
| value           |        0.622 |        0.617 |
| price_rank      |        0.622 |        0.644 |
| career_apps     |        0.592 |        0.591 |
| price_gap_top   |        0.588 |        0.614 |
| n_prior_rows    |        0.529 |        0.55  |
| pos_DEF         |        0.524 |        0.529 |
| pos_MID         |        0.519 |        0.555 |
| five_subs       |        0.512 |        0.509 |
| season_progress |        0.512 |        0.507 |
| opp_rest_days   |        0.507 |        0.501 |
| rest_days       |        0.507 |        0.507 |
| n_active_pos    |        0.507 |        0.555 |
| next_gap_days   |        0.506 |        0.501 |
| supremacy       |        0.505 |        0.608 |
| p_win           |        0.505 |        0.608 |
| pos_GKP         |        0.503 |        0.513 |
| bcd             |        0.502 |        0.507 |
| pos_FWD         |        0.502 |        0.512 |
| total_goals     |        0.501 |        0.507 |
| fix_in_event    |        0.501 |        0.504 |
| home            |        0.5   |        0.524 |
