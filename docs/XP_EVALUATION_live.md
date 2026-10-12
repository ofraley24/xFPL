# xP evaluation (2026-27 finished gameweeks (untouched test), GIVEN THE PLAYER STARTS: 2026-27)

Scoring rules (incl. defensive contribution from 2025-26) reproduce actual points on 100.0000% of rows.

Rows evaluated: 1,276 player-fixtures in which the player STARTED. The model conditions on the start; everything else is as of the gameweek deadline.

## Accuracy vs benchmarks (rows where the archived FPL xP exists: 2026-27)

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                | -0.047  | 3.2007 | 2.4342 |               0.2004 |                          5.95   |
| form: mean of last 5      | -0.9143 | 3.7121 | 2.6826 |               0.1104 |                          4.75   |
| points per start (last 5) | -0.0122 | 3.5546 | 2.6488 |               0.0847 |                          4.4667 |

All evaluated rows (2026-27):

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                | -0.047  | 3.2007 | 2.4342 |               0.2004 |                          5.95   |
| points per start (last 5) | -0.0122 | 3.5546 | 2.6488 |               0.0847 |                          4.4667 |

Fig 1: 96% of the 25 bins' 95% CIs contain the bin's mean prediction.

## Calibration across feature bins (Fig 2)

100% of 46 bins have a 95% CI that contains the mean prediction. Bins where it does not:

| feature   | bin   | n   | pred   | obs   | ratio obs/pred   | CI contains pred   |
|-----------|-------|-----|--------|-------|------------------|--------------------|

## Component calibration in the large (Fig 3)

| component                        |   mean predicted |   mean observed |   ratio |
|:---------------------------------|-----------------:|----------------:|--------:|
| minutes                          |          82.8515 |         82.3582 |  0.994  |
| appearance points                |           1.9355 |          1.9412 |  1.003  |
| goals                            |           0.1075 |          0.1113 |  1.0348 |
| assists                          |           0.1015 |          0.1089 |  1.0732 |
| clean sheet (60+ min)            |           0.2642 |          0.2868 |  1.0859 |
| bonus points                     |           0.2801 |          0.2798 |  0.9988 |
| save points (GKP)                |           0.6539 |          0.6466 |  0.9887 |
| yellow cards                     |           0.1519 |          0.1513 |  0.9959 |
| defensive contribution (P of +2) |           0.1698 |          0.1707 |  1.0054 |

## Uncertainty diagnostics (Fig 4)

| quantity                                       |    value |
|:-----------------------------------------------|---------:|
| PIT mean (0.5 ideal)                           |   0.5003 |
| PIT sd (0.289 ideal)                           |   0.2949 |
| share PIT in central 50%                       |   0.4796 |
| share PIT in central 80%                       |   0.7821 |
| [q25,q75] covers actual (incl. ties)           |   0.6317 |
| [q10,q90] covers actual (incl. ties)           |   0.8644 |
| season totals sd(z) aleatoric only             | nan      |
| season totals sd(z) +epistemic independent     | nan      |
| season totals sd(z) +epistemic correlated      | nan      |
| season totals 80% coverage (correlated)        | nan      |
| season totals 95% coverage (correlated)        | nan      |
| median epistemic share of variance (row level) |   0.0043 |

Epistemic uncertainty by amount of Understat history (should fall with experience):

| L540_time     |   n |   mean_xp |   mean_sd_epistemic |   mean_epistemic_share |
|:--------------|----:|----------:|--------------------:|-----------------------:|
| <5 full games | 156 |    3.0768 |              0.2408 |                 0.013  |
| 5-15          | 131 |    3.2928 |              0.1892 |                 0.0084 |
| 15-30         | 269 |    3.5259 |              0.1822 |                 0.0069 |
| 30+           | 720 |    3.8684 |              0.1821 |                 0.0063 |

## Benchmark validity

The archived FPL xP ranks THIS match far better than this model but falls to the same level on the player's NEXT match, which no forecast can see; a genuine pre-deadline forecast should degrade by a similar amount to the others. This pattern indicates the archived value contains post-deadline information about its own gameweek, so it is not a fair same-gameweek benchmark. A clean FPL benchmark is being collected from deadline snapshots (ep_next) from 2026-10-11.

## Per-season benchmark (Fig 5)

| season   | model                     |   RMSE |   Spearman |   Spearman next match |
|:---------|:--------------------------|-------:|-----------:|----------------------:|
| 2026-27  | this model                | 3.2007 |     0.2004 |                0.1416 |
| 2026-27  | form (last 5)             | 3.7121 |     0.1104 |                0.1131 |
| 2026-27  | points per start (last 5) | 3.5546 |     0.0847 |                0.1128 |

## High end of the distribution (Fig 6)

|   top-k by xP per gameweek |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |   P(5+) predicted |   P(5+) observed |
|---------------------------:|-------:|----------:|--------------:|------------:|---------:|----------:|------------------:|-----------------:|
|                          5 |     30 |     6.116 |         5.767 |       0.943 |    3.777 |     7.898 |             0.53  |            0.5   |
|                         10 |     60 |     5.688 |         5.95  |       1.046 |    4.561 |     7.167 |             0.503 |            0.517 |
|                         20 |    120 |     5.266 |         5.067 |       0.962 |    4.178 |     5.927 |             0.472 |            0.392 |
|                         40 |    240 |     4.864 |         4.821 |       0.991 |    4.301 |     5.321 |             0.44  |            0.421 |

| subset             |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |
|:-------------------|-------:|----------:|--------------:|------------:|---------:|----------:|
| price >= £8.0m     |     57 |     5.47  |         5.333 |       0.975 |    4.143 |     6.907 |
| xP >= 4            |    353 |     4.638 |         4.547 |       0.98  |    4.047 |     5.014 |
| all evaluated rows |   1276 |     3.64  |         3.687 |       1.013 |    3.538 |     3.821 |

Rank correlation within each gameweek's top 50 by xP: 0.115
