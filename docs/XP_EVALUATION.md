# xP evaluation (development 2019-20..2024-25, GIVEN THE PLAYER STARTS: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25)

Scoring rules (incl. defensive contribution from 2025-26) reproduce actual points on 100.0000% of rows.

Rows evaluated: 50,160 player-fixtures in which the player STARTED. The model conditions on the start; everything else is as of the gameweek deadline.

## Accuracy vs benchmarks (rows where the archived FPL xP exists: 2020-21, 2021-22, 2022-23, 2023-24, 2024-25)

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                |  0.0314 | 3.006  | 2.2509 |               0.3167 |                          5.9497 |
| FPL xP (vaastav)          | -0.2775 | 2.921  | 2.0694 |               0.5087 |                          7.8063 |
| form: mean of last 5      | -0.6442 | 3.4502 | 2.4473 |               0.1498 |                          4.8169 |
| points per start (last 5) |  0.0267 | 3.3151 | 2.4357 |               0.1675 |                          4.8275 |

All evaluated rows (2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25):

|                           |   bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|-------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                | 0.0436 | 2.9981 | 2.2487 |               0.3144 |                          5.9626 |
| points per start (last 5) | 0.0307 | 3.3103 | 2.4324 |               0.1624 |                          4.8595 |

Fig 1: 92% of the 25 bins' 95% CIs contain the bin's mean prediction.

## Calibration across feature bins (Fig 2)

73% of 55 bins have a 95% CI that contains the mean prediction. Bins where it does not:

| feature            | bin     |     n |   pred |   obs |   ratio obs/pred | CI contains pred   |
|:-------------------|:--------|------:|-------:|------:|-----------------:|:-------------------|
| position           | GKP     |  4560 |  3.674 | 3.533 |            0.962 | False              |
| position           | DEF     | 18854 |  3.079 | 2.988 |            0.971 | False              |
| price band (£m)    | 4.6-5.5 | 22056 |  3.188 | 3.128 |            0.981 | False              |
| starts in last 10  | 0-3     |  9608 |  3.186 | 3.052 |            0.958 | False              |
| starts in last 10  | 4-6     | 10797 |  3.315 | 3.207 |            0.968 | False              |
| minutes last match | 0       |  5360 |  3.238 | 3.11  |            0.96  | False              |
| minutes last match | 90      | 30286 |  3.386 | 3.339 |            0.986 | False              |
| gameweek           | 1-5     |  6534 |  3.383 | 3.29  |            0.972 | False              |
| gameweek           | 6-13    | 10230 |  3.375 | 3.31  |            0.981 | False              |
| ownership quintile | Q1 low  | 10032 |  2.781 | 2.625 |            0.944 | False              |
| form (last-5 pts)  | ≤1      |  9533 |  2.977 | 2.833 |            0.952 | False              |
| form (last-5 pts)  | 1-2     | 11760 |  3.099 | 3.041 |            0.981 | False              |
| season             | 2019-20 |  8360 |  3.451 | 3.347 |            0.97  | False              |
| season             | 2024-25 |  8360 |  3.336 | 3.245 |            0.973 | False              |
| fixtures in GW     | single  | 46079 |  3.379 | 3.323 |            0.984 | False              |

## Component calibration in the large (Fig 3)

| component             |   mean predicted |   mean observed |   ratio |
|:----------------------|-----------------:|----------------:|--------:|
| minutes               |          84.6431 |         83.786  |  0.9899 |
| appearance points     |           1.9514 |          1.9404 |  0.9944 |
| goals                 |           0.11   |          0.1118 |  1.0159 |
| assists               |           0.0988 |          0.1006 |  1.0191 |
| clean sheet (60+ min) |           0.28   |          0.2686 |  0.9594 |
| bonus points          |           0.2811 |          0.2788 |  0.9918 |
| save points (GKP)     |           0.6619 |          0.6697 |  1.0118 |
| yellow cards          |           0.1433 |          0.144  |  1.0051 |

## Uncertainty diagnostics (Fig 4)

| quantity                                       |   value |
|:-----------------------------------------------|--------:|
| PIT mean (0.5 ideal)                           |  0.4908 |
| PIT sd (0.289 ideal)                           |  0.2929 |
| share PIT in central 50%                       |  0.4912 |
| share PIT in central 80%                       |  0.7868 |
| [q25,q75] covers actual (incl. ties)           |  0.6646 |
| [q10,q90] covers actual (incl. ties)           |  0.869  |
| season totals sd(z) aleatoric only             |  1.0934 |
| season totals sd(z) +epistemic independent     |  1.0872 |
| season totals sd(z) +epistemic correlated      |  1.0075 |
| season totals 80% coverage (correlated)        |  0.8121 |
| season totals 95% coverage (correlated)        |  0.9579 |
| median epistemic share of variance (row level) |  0.0091 |

Epistemic uncertainty by amount of Understat history (should fall with experience):

| L540_time     |     n |   mean_xp |   mean_sd_epistemic |   mean_epistemic_share |
|:--------------|------:|----------:|--------------------:|-----------------------:|
| <5 full games |  2474 |    2.9401 |              0.3096 |                 0.0178 |
| 5-15          |  5188 |    3.0339 |              0.2666 |                 0.0136 |
| 15-30         | 10665 |    3.1635 |              0.246  |                 0.0116 |
| 30+           | 31833 |    3.5469 |              0.2494 |                 0.0105 |

## Benchmark validity

The archived FPL xP ranks THIS match far better than this model but falls to the same level on the player's NEXT match, which no forecast can see; a genuine pre-deadline forecast should degrade by a similar amount to the others. This pattern indicates the archived value contains post-deadline information about its own gameweek, so it is not a fair same-gameweek benchmark. A clean FPL benchmark is being collected from deadline snapshots (ep_next) from 2026-10-11.

## Per-season benchmark (Fig 5)

| season   | model                     |   RMSE |   Spearman |   Spearman next match |
|:---------|:--------------------------|-------:|-----------:|----------------------:|
| 2020-21  | this model                | 3.0095 |     0.2937 |                0.1486 |
| 2020-21  | FPL xP                    | 2.9183 |     0.495  |                0.212  |
| 2020-21  | form (last 5)             | 3.4659 |     0.1277 |                0.215  |
| 2020-21  | points per start (last 5) | 3.297  |     0.1459 |                0.1553 |
| 2021-22  | this model                | 3.0172 |     0.3316 |                0.1612 |
| 2021-22  | FPL xP                    | 2.9793 |     0.5063 |                0.2197 |
| 2021-22  | form (last 5)             | 3.5188 |     0.1424 |                0.1932 |
| 2021-22  | points per start (last 5) | 3.3401 |     0.1751 |                0.1617 |
| 2022-23  | this model                | 2.9797 |     0.2895 |                0.1419 |
| 2022-23  | FPL xP                    | 2.9724 |     0.5016 |                0.1963 |
| 2022-23  | form (last 5)             | 3.4179 |     0.1346 |                0.2011 |
| 2022-23  | points per start (last 5) | 3.3115 |     0.1388 |                0.1497 |
| 2023-24  | this model                | 3.0627 |     0.3398 |                0.1869 |
| 2023-24  | FPL xP                    | 2.858  |     0.5257 |                0.2222 |
| 2023-24  | form (last 5)             | 3.4815 |     0.1672 |                0.2248 |
| 2023-24  | points per start (last 5) | 3.3596 |     0.1896 |                0.1977 |
| 2024-25  | this model                | 2.9597 |     0.3281 |                0.2106 |
| 2024-25  | FPL xP                    | 2.875  |     0.515  |                0.2416 |
| 2024-25  | form (last 5)             | 3.3646 |     0.1765 |                0.2378 |
| 2024-25  | points per start (last 5) | 3.2668 |     0.1872 |                0.189  |

## High end of the distribution (Fig 6)

|   top-k by xP per gameweek |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |   P(5+) predicted |   P(5+) observed |
|---------------------------:|-------:|----------:|--------------:|------------:|---------:|----------:|------------------:|-----------------:|
|                          5 |   1135 |     6.642 |         6.654 |       1.002 |    6.35  |     6.94  |             0.573 |            0.587 |
|                         10 |   2270 |     6.073 |         5.963 |       0.982 |    5.767 |     6.144 |             0.536 |            0.539 |
|                         20 |   4540 |     5.498 |         5.337 |       0.971 |    5.195 |     5.49  |             0.5   |            0.484 |
|                         40 |   9080 |     4.941 |         4.834 |       0.978 |    4.717 |     4.935 |             0.461 |            0.453 |

| subset             |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |
|:-------------------|-------:|----------:|--------------:|------------:|---------:|----------:|
| price >= £8.0m     |   2775 |     5.597 |         5.463 |       0.976 |    5.291 |     5.637 |
| xP >= 4            |  11291 |     4.791 |         4.678 |       0.977 |    4.57  |     4.76  |
| all evaluated rows |  50160 |     3.382 |         3.339 |       0.987 |    3.317 |     3.362 |

Rank correlation within each gameweek's top 50 by xP: 0.169
