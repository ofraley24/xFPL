# xP evaluation (development 2019-20..2024-25, GIVEN THE PLAYER STARTS: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25)

Scoring rules (incl. defensive contribution from 2025-26) reproduce actual points on 100.0000% of rows.

Rows evaluated: 50,160 player-fixtures in which the player STARTED. The model conditions on the start; everything else is as of the gameweek deadline.

## Accuracy vs benchmarks (rows where the archived FPL xP exists: 2020-21, 2021-22, 2022-23, 2023-24, 2024-25)

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                |  0.1041 | 3.0092 | 2.2727 |               0.3158 |                          5.9354 |
| FPL xP (vaastav)          | -0.2775 | 2.921  | 2.0694 |               0.5087 |                          7.8063 |
| form: mean of last 5      | -0.6442 | 3.4502 | 2.4473 |               0.1498 |                          4.8169 |
| points per start (last 5) |  0.0267 | 3.3151 | 2.4357 |               0.1675 |                          4.8275 |

All evaluated rows (2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25):

|                           |   bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|-------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                | 0.1115 | 3.0014 | 2.2692 |               0.313  |                          5.97   |
| points per start (last 5) | 0.0307 | 3.3103 | 2.4324 |               0.1624 |                          4.8595 |

Fig 1: 60% of the 25 bins' 95% CIs contain the bin's mean prediction.

## Calibration across feature bins (Fig 2)

25% of 55 bins have a 95% CI that contains the mean prediction. Bins where it does not:

| feature             | bin           |     n |   pred |   obs |   ratio obs/pred | CI contains pred   |
|:--------------------|:--------------|------:|-------:|------:|-----------------:|:-------------------|
| position            | GKP           |  4560 |  3.655 | 3.533 |            0.967 | False              |
| position            | DEF           | 18854 |  3.142 | 2.988 |            0.951 | False              |
| position            | MID           | 21360 |  3.494 | 3.419 |            0.979 | False              |
| position            | FWD           |  5386 |  4.184 | 4.083 |            0.976 | False              |
| price band (£m)     | ≤4.5          | 14045 |  2.829 | 2.709 |            0.957 | False              |
| price band (£m)     | 4.6-5.5       | 22056 |  3.237 | 3.128 |            0.966 | False              |
| price band (£m)     | 8.1-10        |  1451 |  5.247 | 4.951 |            0.944 | False              |
| price band (£m)     | >10           |  1087 |  6.962 | 6.261 |            0.899 | False              |
| team λ quintile     | Q2            | 10032 |  3.054 | 2.956 |            0.968 | False              |
| team λ quintile     | Q3            | 10032 |  3.373 | 3.256 |            0.965 | False              |
| team λ quintile     | Q5 strong     | 10032 |  4.494 | 4.33  |            0.963 | False              |
| opponent λ quintile | Q1 weak opp   | 10032 |  4.503 | 4.266 |            0.947 | False              |
| opponent λ quintile | Q4            | 10032 |  3.051 | 2.933 |            0.961 | False              |
| opponent λ quintile | Q5 strong opp | 10032 |  2.519 | 2.384 |            0.946 | False              |
| venue               | away          | 25080 |  3.256 | 3.147 |            0.967 | False              |
| venue               | home          | 25080 |  3.644 | 3.53  |            0.969 | False              |
| starts in last 10   | 0-3           |  9608 |  3.259 | 3.052 |            0.937 | False              |
| starts in last 10   | 4-6           | 10797 |  3.384 | 3.207 |            0.948 | False              |
| starts in last 10   | 7-8           | 10085 |  3.512 | 3.406 |            0.97  | False              |
| minutes last match  | 0             |  5360 |  3.316 | 3.11  |            0.938 | False              |
| minutes last match  | 31-59         |  2399 |  3.368 | 3.197 |            0.949 | False              |
| minutes last match  | 90            | 30286 |  3.449 | 3.339 |            0.968 | False              |
| gameweek            | 1-5           |  6534 |  3.461 | 3.29  |            0.951 | False              |
| gameweek            | 6-13          | 10230 |  3.447 | 3.31  |            0.96  | False              |
| gameweek            | 14-26         | 17182 |  3.45  | 3.362 |            0.975 | False              |
| gameweek            | 27-33         |  8206 |  3.442 | 3.332 |            0.968 | False              |
| gameweek            | 34+           |  8008 |  3.455 | 3.372 |            0.976 | False              |
| ownership quintile  | Q1 low        | 10032 |  2.839 | 2.625 |            0.925 | False              |
| ownership quintile  | Q2            | 10032 |  3.072 | 2.98  |            0.97  | False              |
| ownership quintile  | Q4            | 10032 |  3.652 | 3.562 |            0.975 | False              |
| ownership quintile  | Q5 high       | 10032 |  4.368 | 4.235 |            0.97  | False              |
| form (last-5 pts)   | ≤1            |  9533 |  3.042 | 2.833 |            0.931 | False              |
| form (last-5 pts)   | 1-2           | 11760 |  3.16  | 3.041 |            0.962 | False              |
| form (last-5 pts)   | 2-3           | 10860 |  3.358 | 3.25  |            0.968 | False              |
| season              | 2019-20       |  8360 |  3.495 | 3.347 |            0.958 | False              |
| season              | 2020-21       |  8360 |  3.517 | 3.438 |            0.978 | False              |
| season              | 2021-22       |  8360 |  3.498 | 3.412 |            0.975 | False              |
| season              | 2022-23       |  8360 |  3.449 | 3.336 |            0.967 | False              |
| season              | 2023-24       |  8360 |  3.357 | 3.254 |            0.969 | False              |
| season              | 2024-25       |  8360 |  3.385 | 3.245 |            0.959 | False              |
| fixtures in GW      | single        | 46079 |  3.446 | 3.323 |            0.964 | False              |

## Component calibration in the large (Fig 3)

| component             |   mean predicted |   mean observed |   ratio |
|:----------------------|-----------------:|----------------:|--------:|
| minutes               |          84.6231 |         83.786  |  0.9901 |
| appearance points     |           1.9516 |          1.9404 |  0.9943 |
| goals                 |           0.1179 |          0.1118 |  0.9482 |
| assists               |           0.1039 |          0.1006 |  0.9689 |
| clean sheet (60+ min) |           0.2777 |          0.2686 |  0.9673 |
| bonus points          |           0.2905 |          0.2788 |  0.9598 |
| save points (GKP)     |           0.6637 |          0.6697 |  1.0092 |
| yellow cards          |           0.1311 |          0.144  |  1.0986 |

## Uncertainty diagnostics (Fig 4)

| quantity                                       |   value |
|:-----------------------------------------------|--------:|
| PIT mean (0.5 ideal)                           |  0.4864 |
| PIT sd (0.289 ideal)                           |  0.2916 |
| share PIT in central 50%                       |  0.492  |
| share PIT in central 80%                       |  0.7899 |
| [q25,q75] covers actual (incl. ties)           |  0.666  |
| [q10,q90] covers actual (incl. ties)           |  0.8745 |
| season totals sd(z) aleatoric only             |  1.068  |
| season totals sd(z) +epistemic independent     |  1.0617 |
| season totals sd(z) +epistemic correlated      |  0.9816 |
| season totals 80% coverage (correlated)        |  0.8086 |
| season totals 95% coverage (correlated)        |  0.9654 |
| median epistemic share of variance (row level) |  0.0093 |

Epistemic uncertainty by amount of Understat history (should fall with experience):

| L540_time     |     n |   mean_xp |   mean_sd_epistemic |   mean_epistemic_share |
|:--------------|------:|----------:|--------------------:|-----------------------:|
| <5 full games |  2474 |    3.0345 |              0.342  |                 0.0195 |
| 5-15          |  5188 |    3.1052 |              0.2818 |                 0.0142 |
| 15-30         | 10665 |    3.2213 |              0.2592 |                 0.0122 |
| 30+           | 31833 |    3.6156 |              0.2572 |                 0.0105 |

## Benchmark validity

The archived FPL xP ranks THIS match far better than this model but falls to the same level on the player's NEXT match, which no forecast can see; a genuine pre-deadline forecast should degrade by a similar amount to the others. This pattern indicates the archived value contains post-deadline information about its own gameweek, so it is not a fair same-gameweek benchmark. A clean FPL benchmark is being collected from deadline snapshots (ep_next) from 2026-10-11.

## Per-season benchmark (Fig 5)

| season   | model                     |   RMSE |   Spearman |   Spearman next match |
|:---------|:--------------------------|-------:|-----------:|----------------------:|
| 2020-21  | this model                | 3.0154 |     0.2894 |                0.1428 |
| 2020-21  | FPL xP                    | 2.9183 |     0.495  |                0.212  |
| 2020-21  | form (last 5)             | 3.4659 |     0.1277 |                0.215  |
| 2020-21  | points per start (last 5) | 3.297  |     0.1459 |                0.1553 |
| 2021-22  | this model                | 3.021  |     0.3325 |                0.1578 |
| 2021-22  | FPL xP                    | 2.9793 |     0.5063 |                0.2197 |
| 2021-22  | form (last 5)             | 3.5188 |     0.1424 |                0.1932 |
| 2021-22  | points per start (last 5) | 3.3401 |     0.1751 |                0.1617 |
| 2022-23  | this model                | 2.9841 |     0.2883 |                0.1379 |
| 2022-23  | FPL xP                    | 2.9724 |     0.5016 |                0.1963 |
| 2022-23  | form (last 5)             | 3.4179 |     0.1346 |                0.2011 |
| 2022-23  | points per start (last 5) | 3.3115 |     0.1388 |                0.1497 |
| 2023-24  | this model                | 3.0653 |     0.3397 |                0.189  |
| 2023-24  | FPL xP                    | 2.858  |     0.5257 |                0.2222 |
| 2023-24  | form (last 5)             | 3.4815 |     0.1672 |                0.2248 |
| 2023-24  | points per start (last 5) | 3.3596 |     0.1896 |                0.1977 |
| 2024-25  | this model                | 2.9591 |     0.3285 |                0.2078 |
| 2024-25  | FPL xP                    | 2.875  |     0.515  |                0.2416 |
| 2024-25  | form (last 5)             | 3.3646 |     0.1765 |                0.2378 |
| 2024-25  | points per start (last 5) | 3.2668 |     0.1872 |                0.189  |

## High end of the distribution (Fig 6)

|   top-k by xP per gameweek |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |   P(5+) predicted |   P(5+) observed |
|---------------------------:|-------:|----------:|--------------:|------------:|---------:|----------:|------------------:|-----------------:|
|                          5 |   1135 |     7.146 |         6.518 |       0.912 |    6.241 |     6.837 |             0.609 |            0.581 |
|                         10 |   2270 |     6.438 |         5.97  |       0.927 |    5.783 |     6.167 |             0.563 |            0.539 |
|                         20 |   4540 |     5.748 |         5.356 |       0.932 |    5.219 |     5.513 |             0.52  |            0.492 |
|                         40 |   9080 |     5.099 |         4.841 |       0.949 |    4.75  |     4.938 |             0.473 |            0.455 |

| subset             |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |
|:-------------------|-------:|----------:|--------------:|------------:|---------:|----------:|
| price >= £8.0m     |   2775 |     5.892 |         5.463 |       0.927 |    5.279 |     5.628 |
| xP >= 4            |  11832 |     4.887 |         4.67  |       0.955 |    4.58  |     4.769 |
| all evaluated rows |  50160 |     3.45  |         3.339 |       0.968 |    3.316 |     3.362 |

Rank correlation within each gameweek's top 50 by xP: 0.164
