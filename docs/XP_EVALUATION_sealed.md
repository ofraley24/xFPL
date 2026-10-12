# xP evaluation (2025-26 (opened earlier with the unconditional model), GIVEN THE PLAYER STARTS: 2025-26)

Scoring rules (incl. defensive contribution from 2025-26) reproduce actual points on 100.0000% of rows.

Rows evaluated: 8,360 player-fixtures in which the player STARTED. The model conditions on the start; everything else is as of the gameweek deadline.

## Accuracy vs benchmarks (rows where the archived FPL xP exists: 2025-26)

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                |  0.0445 | 3.0841 | 2.3688 |               0.2029 |                          5.3895 |
| FPL xP (vaastav)          | -2.7076 | 4.3092 | 3.1615 |               0.5138 |                          4.6789 |
| form: mean of last 5      | -0.7295 | 3.5726 | 2.5868 |               0.0793 |                          4.3316 |
| points per start (last 5) | -0.0122 | 3.4091 | 2.5557 |               0.0783 |                          4.1211 |

All evaluated rows (2025-26):

|                           |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:--------------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model                |  0.0445 | 3.0841 | 2.3688 |               0.2029 |                          5.3895 |
| points per start (last 5) | -0.0122 | 3.4091 | 2.5557 |               0.0783 |                          4.1211 |

Fig 1: 84% of the 25 bins' 95% CIs contain the bin's mean prediction.

## Calibration across feature bins (Fig 2)

92% of 50 bins have a 95% CI that contains the mean prediction. Bins where it does not:

| feature            | bin     |    n |   pred |   obs |   ratio obs/pred | CI contains pred   |
|:-------------------|:--------|-----:|-------:|------:|-----------------:|:-------------------|
| position           | GKP     |  760 |  3.572 | 3.364 |            0.942 | False              |
| price band (£m)    | 5.6-6.5 | 1496 |  3.967 | 3.801 |            0.958 | False              |
| minutes last match | 60-89   | 1678 |  3.69  | 3.512 |            0.952 | False              |
| gameweek           | 6-13    | 1760 |  3.677 | 3.547 |            0.965 | False              |

## Component calibration in the large (Fig 3)

| component                        |   mean predicted |   mean observed |   ratio |
|:---------------------------------|-----------------:|----------------:|--------:|
| minutes                          |          83.036  |         82.7028 |  0.996  |
| appearance points                |           1.9373 |          1.9294 |  0.9959 |
| goals                            |           0.111  |          0.1038 |  0.9357 |
| assists                          |           0.0993 |          0.0971 |  0.9785 |
| clean sheet (60+ min)            |           0.2716 |          0.2677 |  0.9856 |
| bonus points                     |           0.2789 |          0.2756 |  0.9883 |
| save points (GKP)                |           0.649  |          0.5882 |  0.9063 |
| yellow cards                     |           0.1525 |          0.1452 |  0.9525 |
| defensive contribution (P of +2) |           0.1726 |          0.1854 |  1.0741 |

## Uncertainty diagnostics (Fig 4)

| quantity                                       |   value |
|:-----------------------------------------------|--------:|
| PIT mean (0.5 ideal)                           |  0.4958 |
| PIT sd (0.289 ideal)                           |  0.2907 |
| share PIT in central 50%                       |  0.4965 |
| share PIT in central 80%                       |  0.7931 |
| [q25,q75] covers actual (incl. ties)           |  0.6502 |
| [q10,q90] covers actual (incl. ties)           |  0.8714 |
| season totals sd(z) aleatoric only             |  1.0591 |
| season totals sd(z) +epistemic independent     |  1.0546 |
| season totals sd(z) +epistemic correlated      |  0.9998 |
| season totals 80% coverage (correlated)        |  0.8112 |
| season totals 95% coverage (correlated)        |  0.9499 |
| median epistemic share of variance (row level) |  0.0063 |

Epistemic uncertainty by amount of Understat history (should fall with experience):

| L540_time     |    n |   mean_xp |   mean_sd_epistemic |   mean_epistemic_share |
|:--------------|-----:|----------:|--------------------:|-----------------------:|
| <5 full games |  307 |    3.2462 |              0.2832 |                 0.0141 |
| 5-15          |  738 |    3.2484 |              0.242  |                 0.0109 |
| 15-30         | 1684 |    3.4582 |              0.2193 |                 0.0088 |
| 30+           | 5631 |    3.8106 |              0.2142 |                 0.0079 |

## Benchmark validity

The archived FPL xP ranks THIS match far better than this model but falls to the same level on the player's NEXT match, which no forecast can see; a genuine pre-deadline forecast should degrade by a similar amount to the others. This pattern indicates the archived value contains post-deadline information about its own gameweek, so it is not a fair same-gameweek benchmark. A clean FPL benchmark is being collected from deadline snapshots (ep_next) from 2026-10-11.

## Per-season benchmark (Fig 5)

| season   | model                     |   RMSE |   Spearman |   Spearman next match |
|:---------|:--------------------------|-------:|-----------:|----------------------:|
| 2025-26  | this model                | 3.0841 |     0.2029 |                0.118  |
| 2025-26  | FPL xP                    | 4.3092 |     0.5138 |                0.1793 |
| 2025-26  | form (last 5)             | 3.5726 |     0.0793 |                0.1518 |
| 2025-26  | points per start (last 5) | 3.4091 |     0.0783 |                0.1083 |

## High end of the distribution (Fig 6)

|   top-k by xP per gameweek |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |   P(5+) predicted |   P(5+) observed |
|---------------------------:|-------:|----------:|--------------:|------------:|---------:|----------:|------------------:|-----------------:|
|                          5 |    190 |     6.182 |         5.674 |       0.918 |    5.17  |     6.165 |             0.534 |            0.505 |
|                         10 |    380 |     5.765 |         5.389 |       0.935 |    4.989 |     5.786 |             0.506 |            0.497 |
|                         20 |    760 |     5.347 |         4.958 |       0.927 |    4.676 |     5.256 |             0.473 |            0.441 |
|                         40 |   1520 |     4.932 |         4.615 |       0.936 |    4.399 |     4.807 |             0.439 |            0.404 |

| subset             |   rows |   mean xP |   mean actual |   actual/xP |   CI low |   CI high |
|:-------------------|-------:|----------:|--------------:|------------:|---------:|----------:|
| price >= £8.0m     |    364 |     5.462 |         5.03  |       0.921 |    4.628 |     5.438 |
| xP >= 4            |   2480 |     4.637 |         4.516 |       0.974 |    4.315 |     4.686 |
| all evaluated rows |   8360 |     3.669 |         3.625 |       0.988 |    3.568 |     3.684 |

Rank correlation within each gameweek's top 50 by xP: 0.088
