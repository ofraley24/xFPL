# Full xP and points given a substitute appearance: results

Plan (written before any run): docs/AVAILABILITY_PLAN.md, parts B and C. Script: analysis/31_full_xp.py.

## B. Points given a substitute appearance (development substitute appearances)

|     |   appearances |   actual pts |   xP | sub |   actual/predicted |   actual min |   predicted min |
|:----|--------------:|-------------:|-----------:|-------------------:|-------------:|----------------:|
| DEF |           630 |        1.022 |      1.071 |              0.955 |       19.319 |          21.903 |
| FWD |           469 |        1.443 |      1.536 |              0.94  |       17.446 |          19.846 |
| GKP |             4 |        1     |      1.474 |              0.678 |       40.75  |          41.022 |
| MID |          1881 |        1.365 |      1.379 |              0.99  |       18.523 |          20.845 |
| all |          2984 |        1.305 |      1.339 |              0.974 |       18.552 |          20.939 |

Flag: outside +-15% for GKP.

For comparison, starters: actual/predicted 0.973 (8,258 starts).

## C. The full expectation (all players with minutes in their previous 10 fixtures)

**Sanity rule (overall actual/predicted within +-5%): PASS** (0.983; 17,562 player-fixtures).

By decile of P(start):

|                  |     0 |     1 |     2 |     3 |     4 |    5 |     6 |     7 |     8 |     9 |
|:-----------------|------:|------:|------:|------:|------:|-----:|------:|------:|------:|------:|
| p_start          | 0.027 | 0.068 | 0.127 | 0.214 | 0.344 | 0.53 | 0.703 | 0.813 | 0.886 | 0.94  |
| xp_full          | 0.204 | 0.461 | 0.786 | 1.113 | 1.513 | 2.01 | 2.465 | 2.81  | 3.007 | 3.455 |
| actual           | 0.164 | 0.349 | 0.65  | 1.087 | 1.549 | 2.05 | 2.456 | 2.735 | 2.994 | 3.49  |
| actual/predicted | 0.807 | 0.757 | 0.827 | 0.977 | 1.024 | 1.02 | 0.996 | 0.974 | 0.996 | 1.01  |

By position:

| position   |   rows |   xP full |   actual |   actual/predicted |
|:-----------|-------:|----------:|---------:|-------------------:|
| DEF        |   6111 |     1.59  |    1.484 |              0.933 |
| FWD        |   1946 |     2.001 |    2.07  |              1.035 |
| GKP        |   1242 |     2.178 |    1.969 |              0.904 |
| MID        |   8263 |     1.814 |    1.844 |              1.017 |

Ranking all players (every universe player-fixture, starters or not):

| predictor                  |   within-GW rank corr |     RMSE |   top-10 by GW: mean actual |
|:---------------------------|----------------------:|---------:|----------------------------:|
| full xP                    |                0.6012 |   2.3491 |                      5.6368 |
| xP given a start           |                0.2257 | nan      |                      4.8105 |
| form (last 5, zeros incl.) |                0.5252 |   2.5594 |                      4.7737 |

Mixture distribution, randomised PIT: central 50% 0.507, central 80% 0.809 (nominal 0.50 / 0.80).

Players outside the universe (no minutes in their previous 10 fixtures) are predicted 0: 9,721 player-fixtures, 324 appearances, 1.6% of all points scored.
