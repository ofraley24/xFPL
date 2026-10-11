# FPL data QC (2026-10-11)

51/51 checks passed.

## Points identity by season

| season   |   rows |   R2 |   exact_match |
|:---------|-------:|-----:|--------------:|
| 2016-17  |  23679 |    1 |             1 |
| 2017-18  |  22467 |    1 |             1 |
| 2018-19  |  21790 |    1 |             1 |
| 2019-20  |  22560 |    1 |             1 |
| 2020-21  |  24365 |    1 |             1 |
| 2021-22  |  25447 |    1 |             1 |
| 2022-23  |  26505 |    1 |             1 |
| 2023-24  |  29725 |    1 |             1 |
| 2024-25  |  27283 |    1 |             1 |
| 2025-26  |  29747 |    1 |             1 |
| 2026-27  |   3883 |    1 |             1 |

## Recovered points per event (regression on data; see FPL_RULES_RECOVERED.csv)

| feature    |   2016-17 |   2017-18 |   2018-19 |   2019-20 |   2020-21 |   2021-22 |   2022-23 |   2023-24 |   2024-25 |   2025-26 |   2026-27 |
|:-----------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| assist     |         3 |         3 |         3 |         3 |         3 |         3 |         3 |         3 |         3 |         3 |         3 |
| bonus      |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |
| cs_DEF     |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |
| cs_FWD     |         0 |        -0 |         0 |        -0 |        -0 |         0 |         0 |         0 |        -0 |         0 |        -0 |
| cs_GKP     |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |
| cs_MID     |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |
| dc10_DEF   |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |         2 |         2 |
| dc10_FWD   |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |        -0 |         1 |
| dc10_MID   |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |        -0 |         0 |
| dc12_DEF   |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |        -0 |         0 |
| dc12_FWD   |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |         2 |         1 |
| dc12_MID   |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |       nan |         2 |         2 |
| gc2_DEF    |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |
| gc2_FWD    |        -0 |         0 |        -0 |        -0 |         0 |        -0 |         0 |         0 |         0 |         0 |        -0 |
| gc2_GKP    |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |
| gc2_MID    |        -0 |         0 |        -0 |         0 |         0 |         0 |        -0 |        -0 |         0 |        -0 |        -0 |
| goal_DEF   |         6 |         6 |         6 |         6 |         6 |         6 |         6 |         6 |         6 |         6 |         6 |
| goal_FWD   |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |         4 |
| goal_GKP   |       nan |       nan |       nan |       nan |         6 |       nan |       nan |       nan |       nan |       nan |       nan |
| goal_MID   |         5 |         5 |         5 |         5 |         5 |         5 |         5 |         5 |         5 |         5 |         5 |
| long_play  |         2 |         2 |         2 |         2 |         2 |         2 |         2 |         2 |         2 |         2 |         2 |
| own_goal   |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |
| pen_miss   |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |        -2 |
| pen_save   |         5 |         5 |         5 |         5 |         5 |         5 |         5 |         5 |         5 |         5 |         5 |
| red        |        -3 |        -3 |        -3 |        -3 |        -3 |        -3 |        -3 |        -3 |        -3 |        -3 |        -3 |
| saves3_GKP |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |
| short_play |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |         1 |
| yellow     |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |        -1 |

## Checks

| check | result | detail |
|---|---|---|
| F1 key (season,element,fixture,round) unique | PASS |  |
| F1 every row has a persistent code | PASS | 100.0000% |
| F1 every row has a position (managers labelled MNG) | PASS |  |
| F1 fixture_finished present for every row | PASS |  |
| F1 minutes in [0,150] | PASS |  |
| F2 2016-17 expected_goals coverage (n/a pre-2022-23) | PASS | 0.0% |
| F2 2017-18 expected_goals coverage (n/a pre-2022-23) | PASS | 0.0% |
| F2 2018-19 expected_goals coverage (n/a pre-2022-23) | PASS | 0.0% |
| F2 2019-20 expected_goals coverage (n/a pre-2022-23) | PASS | 0.0% |
| F2 2020-21 expected_goals coverage (n/a pre-2022-23) | PASS | 0.0% |
| F2 2021-22 expected_goals coverage (n/a pre-2022-23) | PASS | 0.0% |
| F2 2022-23 expected_goals coverage (expected) | PASS | 100.0% |
| F2 2023-24 expected_goals coverage (expected) | PASS | 100.0% |
| F2 2024-25 expected_goals coverage (expected) | PASS | 100.0% |
| F2 2025-26 expected_goals coverage (expected) | PASS | 100.0% |
| F2 2025-26 defensive_contribution coverage | PASS | 100.0% |
| F2 2026-27 expected_goals coverage (expected) | PASS | 100.0% |
| F2 2026-27 defensive_contribution coverage | PASS | 100.0% |
| F3 2026-27 summed match history == bootstrap season total (total_points) | PASS | 100.00% of 527 players match |
| F3 2026-27 summed match history == bootstrap season total (minutes) | PASS | 100.00% of 527 players match |
| F3 2026-27 summed match history == bootstrap season total (goals_scored) | PASS | 100.00% of 527 players match |
| F3 2026-27 summed match history == bootstrap season total (assists) | PASS | 100.00% of 527 players match |
| F4 live API vs vaastav GW1 agree (minutes) | PASS | 100.00% of 485 |
| F4 live API vs vaastav GW1 agree (total_points) | PASS | 100.00% of 485 |
| F4 live API vs vaastav GW1 agree (goals_scored) | PASS | 100.00% of 485 |
| F4 live API vs vaastav GW1 agree (bps) | PASS | 100.00% of 485 |
| F4 live API vs vaastav GW1 agree (expected_goals) | PASS | 100.00% of 485 |
| F5 2016-17 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2017-18 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2018-19 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2019-20 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2020-21 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2021-22 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2022-23 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2023-24 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2024-25 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2025-26 points identity: R2=1.0, exact=100.0% | PASS |  |
| F5 2026-27 points identity: R2=1.0, exact=100.0% | PASS |  |
| F6 2026-27 recovered 'goal_DEF' == API published (6) | PASS | recovered 6.0 |
| F6 2026-27 recovered 'goal_MID' == API published (5) | PASS | recovered 5.0 |
| F6 2026-27 recovered 'goal_FWD' == API published (4) | PASS | recovered 4.0 |
| F6 2026-27 recovered 'cs_GKP' == API published (4) | PASS | recovered 4.0 |
| F6 2026-27 recovered 'cs_DEF' == API published (4) | PASS | recovered 4.0 |
| F6 2026-27 recovered 'cs_MID' == API published (1) | PASS | recovered 1.0 |
| F6 2026-27 recovered 'cs_FWD' == API published (0) | PASS | recovered -0.0 |
| F6 2026-27 recovered 'assist' == API published (3) | PASS | recovered 3.0 |
| F6 2026-27 recovered 'pen_save' == API published (5) | PASS | recovered 5.0 |
| F6 2026-27 recovered 'pen_miss' == API published (-2) | PASS | recovered -2.0 |
| F6 2026-27 recovered 'red' == API published (-3) | PASS | recovered -3.0 |
| F6 2026-27 recovered 'yellow' == API published (-1) | PASS | recovered -1.0 |
| F6 2026-27 recovered 'own_goal' == API published (-2) | PASS | recovered -2.0 |
