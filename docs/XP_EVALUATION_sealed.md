# xP evaluation (SEALED HOLDOUT, run once: 2025-26)

Run note: the first execution of the sealed fold stopped with an error in the defensive-contribution draw (no prior for goalkeepers) before any prediction was produced; after that fix the fold was run once to completion. No sealed outcome was seen before the fix.

Scoring rules (incl. defensive contribution from 2025-26) reproduce actual points on 100.0000% of rows.

Rows evaluated (universe): 17,124. Rows outside the universe (predicted 0): 12,623, which scored 657 of 34,382 points (1.91%).

## Accuracy vs benchmarks (rows where the archived FPL xP exists: 2025-26)

|                      |    bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:---------------------|--------:|-------:|-------:|---------------------:|--------------------------------:|
| this model           |  0.0327 | 2.4791 | 1.6256 |               0.6027 |                          4.5632 |
| FPL xP (vaastav)     | -1.4158 | 3.0958 | 1.7848 |               0.7357 |                          3.5421 |
| form: mean of last 5 |  0.0377 | 2.7462 | 1.789  |               0.4872 |                          3.7895 |

All evaluated rows (2025-26):

|            |    MAE |   RMSE |   bias |   top-10 by GW: mean actual pts |   within-GW Spearman |
|:-----------|-------:|-------:|-------:|--------------------------------:|---------------------:|
| this model | 1.6256 | 2.4791 | 0.0327 |                          4.5632 |               0.6027 |

Fig 1: 84% of the 25 bins' 95% CIs contain the bin's mean prediction.

## Calibration across feature bins (Fig 2)

81% of 52 bins have a 95% CI that contains the mean prediction. Bins where it does not:

| feature            | bin     |     n |   pred |   obs |   ratio obs/pred | CI contains pred   |
|:-------------------|:--------|------:|-------:|------:|-----------------:|:-------------------|
| position           | GKP     |  1129 |  2.317 | 2.163 |            0.934 | False              |
| position           | MID     |  7944 |  1.994 | 1.933 |            0.969 | False              |
| price band (£m)    | 5.6-6.5 |  2754 |  2.456 | 2.351 |            0.957 | False              |
| price band (£m)    | 8.1-10  |   322 |  3.84  | 3.36  |            0.875 | False              |
| price band (£m)    | >10     |   169 |  4.75  | 3.805 |            0.801 | False              |
| P(plays) band      | ≤.2     |  2443 |  0.216 | 0.177 |            0.82  | False              |
| P(plays) band      | .8-.9   |  3100 |  2.662 | 2.548 |            0.957 | False              |
| ownership quintile | Q5 high |  3425 |  3.351 | 3.239 |            0.967 | False              |
| form (last-5 pts)  | 1-2     |  3346 |  2.071 | 1.972 |            0.952 | False              |
| season             | 2025-26 | 17124 |  2.002 | 1.969 |            0.984 | False              |

## Component calibration in the large (Fig 3)

| component                        |   mean predicted |   mean observed |   ratio |
|:---------------------------------|-----------------:|----------------:|--------:|
| minutes                          |          42.8568 |         43.0211 |  1.0038 |
| appearance points                |           1.0948 |          1.1036 |  1.008  |
| goals                            |           0.0641 |          0.0579 |  0.9038 |
| assists                          |           0.056  |          0.0541 |  0.9663 |
| clean sheet (60+ min)            |           0.1302 |          0.1289 |  0.9904 |
| bonus points                     |           0.1417 |          0.1391 |  0.9817 |
| save points (GKP)                |           0.4193 |          0.3782 |  0.902  |
| yellow cards                     |           0.0757 |          0.0819 |  1.0824 |
| defensive contribution (P of +2) |           0.0808 |          0.0875 |  1.0826 |

## Uncertainty diagnostics (Fig 4)

| quantity                                       |   value |
|:-----------------------------------------------|--------:|
| PIT mean (0.5 ideal)                           |  0.501  |
| PIT sd (0.289 ideal)                           |  0.2877 |
| share PIT in central 50%                       |  0.5062 |
| share PIT in central 80%                       |  0.8018 |
| [q25,q75] covers actual (incl. ties)           |  0.7417 |
| [q10,q90] covers actual (incl. ties)           |  0.9074 |
| season totals sd(z) aleatoric only             |  1.0108 |
| season totals sd(z) +epistemic independent     |  1.0071 |
| season totals sd(z) +epistemic correlated      |  0.9503 |
| season totals 80% coverage (correlated)        |  0.8315 |
| season totals 95% coverage (correlated)        |  0.9685 |
| median epistemic share of variance (row level) |  0.0042 |

Epistemic uncertainty by amount of Understat history (should fall with experience):

| L540_time     |    n |   mean_xp |   mean_sd_epistemic |   mean_epistemic_share |
|:--------------|-----:|----------:|--------------------:|-----------------------:|
| <5 full games | 1840 |    0.7249 |              0.0901 |                 0.0072 |
| 5-15          | 2049 |    1.4178 |              0.1325 |                 0.0075 |
| 15-30         | 4070 |    1.6894 |              0.139  |                 0.0069 |
| 30+           | 9165 |    2.5282 |              0.1742 |                 0.007  |

## Benchmark validity

The archived FPL xP ranks THIS match far better than this model but falls to the same level on the player's NEXT match, which no forecast can see; a genuine pre-deadline forecast should degrade by a similar amount to the others. This pattern indicates the archived value contains post-deadline information about its own gameweek, so it is not a fair same-gameweek benchmark. A clean FPL benchmark is being collected from deadline snapshots (ep_next) from 2026-10-11.

## Per-season benchmark (Fig 5)

| season   | model         |   RMSE |   Spearman |   Spearman next match |
|:---------|:--------------|-------:|-----------:|----------------------:|
| 2025-26  | this model    | 2.4791 |     0.6027 |                0.5201 |
| 2025-26  | FPL xP        | 3.0958 |     0.7357 |                0.5241 |
| 2025-26  | form (last 5) | 2.7462 |     0.4872 |                0.4326 |
