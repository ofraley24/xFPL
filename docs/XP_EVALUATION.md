# xP evaluation (development: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25)


Scoring rules (incl. defensive contribution from 2025-26) reproduce actual points on 100.0000% of rows.

Rows evaluated (universe): 101,821. Rows outside the universe (predicted 0): 54,005, which scored 3,494 of 188,159 points (1.86%).

## Accuracy vs benchmarks (rows where the archived FPL xP exists: 2020-21, 2021-22, 2022-23, 2023-24, 2024-25)

|                      |   bias |   RMSE |    MAE |   within-GW Spearman |   top-10 by GW: mean actual pts |
|:---------------------|-------:|-------:|-------:|---------------------:|--------------------------------:|
| this model           | 0.0276 | 2.4293 | 1.5487 |               0.5912 |                          5.2503 |
| FPL xP (vaastav)     | 0.0363 | 2.2471 | 1.3764 |               0.7092 |                          7.5344 |
| form: mean of last 5 | 0.0418 | 2.6534 | 1.6814 |               0.5017 |                          4.4439 |

All evaluated rows (2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25):

|            |    MAE |   RMSE |   bias |   top-10 by GW: mean actual pts |   within-GW Spearman |
|:-----------|-------:|-------:|-------:|--------------------------------:|---------------------:|
| this model | 1.5542 | 2.4311 | 0.0305 |                          5.2771 |               0.5925 |

Fig 1: 84% of the 25 bins' 95% CIs contain the bin's mean prediction.

## Calibration across feature bins (Fig 2)

58% of 57 bins have a 95% CI that contains the mean prediction. Bins where it does not:

| feature             | bin           |     n |   pred |   obs |   ratio obs/pred | CI contains pred   |
|:--------------------|:--------------|------:|-------:|------:|-----------------:|:-------------------|
| position            | GKP           |  6830 |  2.377 | 2.292 |            0.964 | False              |
| position            | DEF           | 35854 |  1.692 | 1.633 |            0.965 | False              |
| price band (£m)     | 8.1-10        |  2310 |  3.602 | 3.37  |            0.936 | False              |
| price band (£m)     | >10           |  1433 |  5.533 | 4.902 |            0.886 | False              |
| opponent λ quintile | Q1 weak opp   | 20365 |  2.409 | 2.316 |            0.961 | False              |
| opponent λ quintile | Q5 strong opp | 20364 |  1.35  | 1.303 |            0.965 | False              |
| P(plays) band       | .2-.4         | 14971 |  0.642 | 0.585 |            0.911 | False              |
| P(plays) band       | >.9           | 24533 |  3.275 | 3.229 |            0.986 | False              |
| minutes last match  | 0             | 36258 |  0.618 | 0.577 |            0.934 | False              |
| minutes last match  | 60-89         | 12087 |  2.559 | 2.624 |            1.025 | False              |
| minutes last match  | 90            | 35220 |  2.999 | 2.93  |            0.977 | False              |
| gameweek            | 1-5           | 12842 |  1.845 | 1.786 |            0.968 | False              |
| gameweek            | 6-13          | 21334 |  1.8   | 1.762 |            0.979 | False              |
| gameweek            | 14-26         | 34772 |  1.857 | 1.826 |            0.983 | False              |
| ownership quintile  | Q1 low        | 20365 |  0.887 | 0.858 |            0.967 | False              |
| ownership quintile  | Q2            | 20364 |  1.306 | 1.259 |            0.964 | False              |
| ownership quintile  | Q5 high       | 20364 |  3.198 | 3.126 |            0.977 | False              |
| form (last-5 pts)   | ≤1            | 43647 |  0.852 | 0.79  |            0.928 | False              |
| form (last-5 pts)   | 1-2           | 20961 |  1.973 | 1.935 |            0.981 | False              |
| season              | 2019-20       | 16250 |  1.917 | 1.872 |            0.976 | False              |
| season              | 2020-21       | 16432 |  1.932 | 1.877 |            0.972 | False              |
| season              | 2021-22       | 16811 |  1.859 | 1.829 |            0.984 | False              |
| season              | 2024-25       | 17562 |  1.795 | 1.753 |            0.976 | False              |
| fixtures in GW      | single        | 93516 |  1.842 | 1.807 |            0.981 | False              |

## Component calibration in the large (Fig 3)

| component             |   mean predicted |   mean observed |   ratio |
|:----------------------|-----------------:|----------------:|--------:|
| minutes               |          43.2993 |         43.3683 |  1.0016 |
| appearance points     |           1.0755 |          1.0866 |  1.0104 |
| goals                 |           0.0633 |          0.0614 |  0.9694 |
| assists               |           0.0563 |          0.0554 |  0.9847 |
| clean sheet (60+ min) |           0.1345 |          0.1314 |  0.9767 |
| bonus points          |           0.1458 |          0.1415 |  0.9704 |
| save points (GKP)     |           0.4299 |          0.4315 |  1.0038 |
| yellow cards          |           0.0702 |          0.0789 |  1.1234 |

## Uncertainty diagnostics (Fig 4)

| quantity                                       |   value |
|:-----------------------------------------------|--------:|
| PIT mean (0.5 ideal)                           |  0.5008 |
| PIT sd (0.289 ideal)                           |  0.2873 |
| share PIT in central 50%                       |  0.503  |
| share PIT in central 80%                       |  0.8054 |
| [q25,q75] covers actual (incl. ties)           |  0.7578 |
| [q10,q90] covers actual (incl. ties)           |  0.9105 |
| season totals sd(z) aleatoric only             |  1.0673 |
| season totals sd(z) +epistemic independent     |  1.0618 |
| season totals sd(z) +epistemic correlated      |  0.9749 |
| season totals 80% coverage (correlated)        |  0.8206 |
| season totals 95% coverage (correlated)        |  0.9681 |
| median epistemic share of variance (row level) |  0.0068 |

Epistemic uncertainty by amount of Understat history (should fall with experience):

| L540_time     |     n |   mean_xp |   mean_sd_epistemic |   mean_epistemic_share |
|:--------------|------:|----------:|--------------------:|-----------------------:|
| <5 full games | 11920 |    0.7721 |              0.122  |                 0.0099 |
| 5-15          | 14250 |    1.3232 |              0.1528 |                 0.0093 |
| 15-30         | 24775 |    1.5621 |              0.1622 |                 0.009  |
| 30+           | 50876 |    2.3785 |              0.2182 |                 0.01   |

## Benchmark validity

The archived FPL xP ranks THIS match far better than this model but falls to the same level on the player's NEXT match, which no forecast can see; a genuine pre-deadline forecast should degrade by a similar amount to the others. This pattern indicates the archived value contains post-deadline information about its own gameweek, so it is not a fair same-gameweek benchmark. A clean FPL benchmark is being collected from deadline snapshots (ep_next) from 2026-10-11.

## Per-season benchmark (Fig 5)

| season   | model         |   RMSE |   Spearman |   Spearman next match |
|:---------|:--------------|-------:|-----------:|----------------------:|
| 2020-21  | this model    | 2.5017 |     0.5663 |                0.4948 |
| 2020-21  | FPL xP        | 2.2914 |     0.6941 |                0.4789 |
| 2020-21  | form (last 5) | 2.7024 |     0.494  |                0.4498 |
| 2021-22  | this model    | 2.4945 |     0.5744 |                0.4915 |
| 2021-22  | FPL xP        | 2.367  |     0.6809 |                0.4712 |
| 2021-22  | form (last 5) | 2.7314 |     0.4818 |                0.4347 |
| 2022-23  | this model    | 2.3836 |     0.6016 |                0.5234 |
| 2022-23  | FPL xP        | 2.2677 |     0.7269 |                0.5202 |
| 2022-23  | form (last 5) | 2.619  |     0.5097 |                0.4701 |
| 2023-24  | this model    | 2.4292 |     0.6031 |                0.5148 |
| 2023-24  | FPL xP        | 2.1548 |     0.7205 |                0.5029 |
| 2023-24  | form (last 5) | 2.6573 |     0.4982 |                0.4562 |
| 2024-25  | this model    | 2.3401 |     0.6111 |                0.5374 |
| 2024-25  | FPL xP        | 2.1559 |     0.7263 |                0.5229 |
| 2024-25  | form (last 5) | 2.5594 |     0.5252 |                0.4775 |
