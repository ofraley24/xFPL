# xP evaluation (2025-26 (opened earlier with the unconditional model), GIVEN THE PLAYER STARTS: 2025-26)

Scoring rules (incl. defensive contribution from 2025-26) reproduce actual points on 100.0000% of rows.

Rows evaluated: 8,360 player-fixtures in which the player STARTED. The model conditions on the start; everything else is as of the gameweek deadline.

## Accuracy vs benchmarks (rows where the archived FPL xP exists: 2025-26)

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                |  0.0804 | 3.0854 | 2.373  |               0.2008 |                          5.2684 |
| FPL xP (vaastav)          | -2.7076 | 4.3092 | 3.1615 |               0.5138 |                          4.6789 |
| form: mean of last 5      | -0.7295 | 3.5726 | 2.5868 |               0.0793 |                          4.3316 |
| points per start (last 5) | -0.0122 | 3.4091 | 2.5557 |               0.0783 |                          4.1211 |

All evaluated rows (2025-26):

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                |  0.0804 | 3.0854 | 2.373  |               0.2008 |                          5.2684 |
| points per start (last 5) | -0.0122 | 3.4091 | 2.5557 |               0.0783 |                          4.1211 |

Fig 1: 96% of the 25 bins' 95% CIs contain the bin's mean prediction.

## Calibration across feature bins (Fig 2)

86% of 50 bins have a 95% CI that contains the mean prediction. Bins where it does not:

| feature            | bin     |    n |   pred |   obs |   ratio obs/pred | CI contains pred   |
|:-------------------|:--------|-----:|-------:|------:|-----------------:|:-------------------|
| position           | GKP     |  760 |  3.578 | 3.364 |            0.94  | False              |
| price band (£m)    | 5.6-6.5 | 1496 |  3.98  | 3.801 |            0.955 | False              |
| price band (£m)    | >10     |  107 |  6.711 | 5.636 |            0.84  | False              |
| minutes last match | 60-89   | 1678 |  3.713 | 3.512 |            0.946 | False              |
| gameweek           | 6-13    | 1760 |  3.716 | 3.547 |            0.954 | False              |
| season             | 2025-26 | 8360 |  3.705 | 3.625 |            0.978 | False              |
| fixtures in GW     | single  | 8140 |  3.701 | 3.623 |            0.979 | False              |

## Component calibration in the large (Fig 3)

| component                        |   mean predicted |   mean observed |   ratio |
|:---------------------------------|-----------------:|----------------:|--------:|
| minutes                          |          83.0178 |         82.7028 |  0.9962 |
| appearance points                |           1.9367 |          1.9294 |  0.9963 |
| goals                            |           0.1154 |          0.1038 |  0.8998 |
| assists                          |           0.1009 |          0.0971 |  0.9626 |
| clean sheet (60+ min)            |           0.2702 |          0.2677 |  0.9909 |
| bonus points                     |           0.2794 |          0.2756 |  0.9864 |
| save points (GKP)                |           0.6501 |          0.5882 |  0.9047 |
| yellow cards                     |           0.1382 |          0.1452 |  1.0508 |
| defensive contribution (P of +2) |           0.1729 |          0.1854 |  1.0725 |

## Uncertainty diagnostics (Fig 4)

| quantity                                       |   value |
|:-----------------------------------------------|--------:|
| PIT mean (0.5 ideal)                           |  0.4916 |
| PIT sd (0.289 ideal)                           |  0.2908 |
| share PIT in central 50%                       |  0.4962 |
| share PIT in central 80%                       |  0.7947 |
| [q25,q75] covers actual (incl. ties)           |  0.6538 |
| [q10,q90] covers actual (incl. ties)           |  0.8737 |
| season totals sd(z) aleatoric only             |  1.0493 |
| season totals sd(z) +epistemic independent     |  1.0447 |
| season totals sd(z) +epistemic correlated      |  0.9898 |
| season totals 80% coverage (correlated)        |  0.8171 |
| season totals 95% coverage (correlated)        |  0.944  |
| median epistemic share of variance (row level) |  0.0063 |

Epistemic uncertainty by amount of Understat history (should fall with experience):

| L540_time     |    n |   mean_xp |   mean_sd_epistemic |   mean_epistemic_share |
|:--------------|-----:|----------:|--------------------:|-----------------------:|
| <5 full games |  307 |    3.2993 |              0.3205 |                 0.0158 |
| 5-15          |  738 |    3.2951 |              0.2389 |                 0.0104 |
| 15-30         | 1684 |    3.4875 |              0.2314 |                 0.0095 |
| 30+           | 5631 |    3.8461 |              0.2178 |                 0.0079 |

## Benchmark validity

The archived FPL xP ranks THIS match far better than this model but falls to the same level on the player's NEXT match, which no forecast can see; a genuine pre-deadline forecast should degrade by a similar amount to the others. This pattern indicates the archived value contains post-deadline information about its own gameweek, so it is not a fair same-gameweek benchmark. A clean FPL benchmark is being collected from deadline snapshots (ep_next) from 2026-10-11.

## Per-season benchmark (Fig 5)

| season   | model                     |   RMSE |   Spearman |   Spearman next match |
|:---------|:--------------------------|-------:|-----------:|----------------------:|
| 2025-26  | this model                | 3.0854 |     0.2008 |                0.1168 |
| 2025-26  | FPL xP                    | 4.3092 |     0.5138 |                0.1793 |
| 2025-26  | form (last 5)             | 3.5726 |     0.0793 |                0.1518 |
| 2025-26  | points per start (last 5) | 3.4091 |     0.0783 |                0.1083 |

## High end of the distribution (Fig 6)

|   top-k by xP per gameweek |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |   P(5+) predicted |   P(5+) observed |
|---------------------------:|-------:|----------:|--------------:|------------:|---------:|----------:|------------------:|-----------------:|
|                          5 |    190 |     6.539 |         5.847 |       0.894 |    5.369 |     6.345 |             0.565 |            0.532 |
|                         10 |    380 |     5.995 |         5.268 |       0.879 |    4.92  |     5.663 |             0.526 |            0.474 |
|                         20 |    760 |     5.495 |         4.929 |       0.897 |    4.588 |     5.201 |             0.487 |            0.437 |
|                         40 |   1520 |     5.014 |         4.626 |       0.923 |    4.378 |     4.836 |             0.447 |            0.412 |

| subset             |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |
|:-------------------|-------:|----------:|--------------:|------------:|---------:|----------:|
| price >= £8.0m     |    364 |     5.686 |         5.03  |       0.885 |    4.661 |     5.532 |
| xP >= 4            |   2527 |     4.678 |         4.501 |       0.962 |    4.34  |     4.667 |
| all evaluated rows |   8360 |     3.705 |         3.625 |       0.978 |    3.566 |     3.686 |

Rank correlation within each gameweek's top 50 by xP: 0.082
