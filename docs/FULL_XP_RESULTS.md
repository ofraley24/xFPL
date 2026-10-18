# Full xP and points given a substitute appearance: results

Plan (written before any run): docs/AVAILABILITY_PLAN.md, parts B and C. Script: analysis/31_full_xp.py.

## B. Points given a substitute appearance (development substitute appearances)

|     |   appearances |   actual pts |   xP | sub |   actual/predicted |   actual min |   predicted min |
|:----|--------------:|-------------:|-----------:|-------------------:|-------------:|----------------:|
| DEF |          2845 |        1.067 |      1.112 |              0.959 |       19.838 |          23.176 |
| FWD |          3017 |        1.515 |      1.511 |              1.003 |       17.771 |          20.48  |
| GKP |            26 |        1.077 |      1.48  |              0.728 |       42     |          44.433 |
| MID |          8600 |        1.354 |      1.347 |              1.006 |       18.874 |          21.223 |
| all |         14488 |        1.331 |      1.335 |              0.997 |       18.875 |          21.493 |

Flag: outside +-15% for GKP.

For comparison, starters: actual/predicted 0.988 (49,430 starts).

## C. The full expectation (all players with minutes in their previous 10 fixtures)

**Sanity rule (overall actual/predicted within +-5%): PASS** (0.988; 101,821 player-fixtures).

By decile of P(start):

|                  |     0 |     1 |     2 |     3 |     4 |     5 |     6 |     7 |     8 |     9 |
|:-----------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| p_start          | 0.03  | 0.075 | 0.142 | 0.239 | 0.378 | 0.572 | 0.734 | 0.83  | 0.896 | 0.943 |
| xp_full          | 0.209 | 0.47  | 0.79  | 1.145 | 1.594 | 2.162 | 2.583 | 2.842 | 3.058 | 3.51  |
| actual           | 0.181 | 0.423 | 0.715 | 1.122 | 1.6   | 2.209 | 2.534 | 2.805 | 3.072 | 3.476 |
| actual/predicted | 0.866 | 0.899 | 0.905 | 0.979 | 1.004 | 1.022 | 0.981 | 0.987 | 1.004 | 0.99  |

By position:

| position   |   rows |   xP full |   actual |   actual/predicted |
|:-----------|-------:|----------:|---------:|-------------------:|
| DEF        |  35854 |     1.704 |    1.633 |              0.958 |
| FWD        |  13174 |     1.968 |    2.003 |              1.018 |
| GKP        |   6830 |     2.429 |    2.292 |              0.943 |
| MID        |  45963 |     1.813 |    1.829 |              1.009 |

Ranking all players (every universe player-fixture, starters or not):

| predictor                  |   within-GW rank corr |     RMSE |   top-10 by GW: mean actual |
|:---------------------------|----------------------:|---------:|----------------------------:|
| full xP                    |                0.5841 |   2.4381 |                      5.2352 |
| xP given a start           |                0.1985 | nan      |                      4.6493 |
| form (last 5, zeros incl.) |                0.5026 |   2.6571 |                      4.4304 |

Mixture distribution, randomised PIT: central 50% 0.501, central 80% 0.799 (nominal 0.50 / 0.80).

Players outside the universe (no minutes in their previous 10 fixtures) are predicted 0: 54,005 player-fixtures, 1,869 appearances, 1.9% of all points scored.
