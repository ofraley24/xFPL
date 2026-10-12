# xP evaluation (2026-27 finished gameweeks (untouched test), GIVEN THE PLAYER STARTS: 2026-27)

Scoring rules (incl. defensive contribution from 2025-26) reproduce actual points on 100.0000% of rows.

Rows evaluated: 1,276 player-fixtures in which the player STARTED. The model conditions on the start; everything else is as of the gameweek deadline.

## Accuracy vs benchmarks (rows where the archived FPL xP exists: 2026-27)

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                |  0.006  | 3.1961 | 2.4427 |               0.1941 |                          5.8    |
| form: mean of last 5      | -0.9143 | 3.7121 | 2.6826 |               0.1104 |                          4.75   |
| points per start (last 5) | -0.0122 | 3.5546 | 2.6488 |               0.0847 |                          4.4667 |

All evaluated rows (2026-27):

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                |  0.006  | 3.1961 | 2.4427 |               0.1941 |                          5.8    |
| points per start (last 5) | -0.0122 | 3.5546 | 2.6488 |               0.0847 |                          4.4667 |

Fig 1: 88% of the 25 bins' 95% CIs contain the bin's mean prediction.

## Calibration across feature bins (Fig 2)

100% of 46 bins have a 95% CI that contains the mean prediction. Bins where it does not:

| feature   | bin   | n   | pred   | obs   | ratio obs/pred   | CI contains pred   |
|-----------|-------|-----|--------|-------|------------------|--------------------|

## Component calibration in the large (Fig 3)

| component                        |   mean predicted |   mean observed |   ratio |
|:---------------------------------|-----------------:|----------------:|--------:|
| minutes                          |          82.858  |         82.3582 |  0.994  |
| appearance points                |           1.9375 |          1.9412 |  1.0019 |
| goals                            |           0.1134 |          0.1113 |  0.9816 |
| assists                          |           0.1045 |          0.1089 |  1.0419 |
| clean sheet (60+ min)            |           0.2646 |          0.2868 |  1.0841 |
| bonus points                     |           0.2813 |          0.2798 |  0.9945 |
| save points (GKP)                |           0.6521 |          0.6466 |  0.9916 |
| yellow cards                     |           0.138  |          0.1513 |  1.0961 |
| defensive contribution (P of +2) |           0.1698 |          0.1707 |  1.0053 |

## Uncertainty diagnostics (Fig 4)

| quantity                                       |    value |
|:-----------------------------------------------|---------:|
| PIT mean (0.5 ideal)                           |   0.4991 |
| PIT sd (0.289 ideal)                           |   0.2934 |
| share PIT in central 50%                       |   0.4898 |
| share PIT in central 80%                       |   0.7837 |
| [q25,q75] covers actual (incl. ties)           |   0.6332 |
| [q10,q90] covers actual (incl. ties)           |   0.8613 |
| season totals sd(z) aleatoric only             | nan      |
| season totals sd(z) +epistemic independent     | nan      |
| season totals sd(z) +epistemic correlated      | nan      |
| season totals 80% coverage (correlated)        | nan      |
| season totals 95% coverage (correlated)        | nan      |
| median epistemic share of variance (row level) |   0.0052 |

Epistemic uncertainty by amount of Understat history (should fall with experience):

| L540_time     |   n |   mean_xp |   mean_sd_epistemic |   mean_epistemic_share |
|:--------------|----:|----------:|--------------------:|-----------------------:|
| <5 full games | 156 |    3.1974 |              0.2933 |                 0.0156 |
| 5-15          | 131 |    3.3163 |              0.2336 |                 0.0102 |
| 15-30         | 269 |    3.5591 |              0.1942 |                 0.0076 |
| 30+           | 720 |    3.9194 |              0.194  |                 0.0066 |

## Benchmark validity

The archived FPL xP ranks THIS match far better than this model but falls to the same level on the player's NEXT match, which no forecast can see; a genuine pre-deadline forecast should degrade by a similar amount to the others. This pattern indicates the archived value contains post-deadline information about its own gameweek, so it is not a fair same-gameweek benchmark. A clean FPL benchmark is being collected from deadline snapshots (ep_next) from 2026-10-11.

## Per-season benchmark (Fig 5)

| season   | model                     |   RMSE |   Spearman |   Spearman next match |
|:---------|:--------------------------|-------:|-----------:|----------------------:|
| 2026-27  | this model                | 3.1961 |     0.1941 |                0.1386 |
| 2026-27  | form (last 5)             | 3.7121 |     0.1104 |                0.1131 |
| 2026-27  | points per start (last 5) | 3.5546 |     0.0847 |                0.1128 |

## High end of the distribution (Fig 6)

|   top-k by xP per gameweek |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |   P(5+) predicted |   P(5+) observed |
|---------------------------:|-------:|----------:|--------------:|------------:|---------:|----------:|------------------:|-----------------:|
|                          5 |     30 |     6.552 |         6.8   |       1.038 |    4.766 |     8.968 |             0.561 |            0.567 |
|                         10 |     60 |     5.954 |         5.8   |       0.974 |    4.622 |     7.141 |             0.519 |            0.483 |
|                         20 |    120 |     5.444 |         5.117 |       0.94  |    4.159 |     5.903 |             0.488 |            0.425 |
|                         40 |    240 |     4.981 |         4.8   |       0.964 |    4.282 |     5.314 |             0.448 |            0.421 |

| subset             |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |
|:-------------------|-------:|----------:|--------------:|------------:|---------:|----------:|
| price >= £8.0m     |     57 |     5.702 |         5.333 |       0.935 |    3.941 |     6.964 |
| xP >= 4            |    378 |     4.68  |         4.571 |       0.977 |    4.152 |     4.998 |
| all evaluated rows |   1276 |     3.693 |         3.687 |       0.998 |    3.543 |     3.825 |

Rank correlation within each gameweek's top 50 by xP: 0.106
