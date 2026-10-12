# High-end and yellow-card fixes: results (dev)

Pre-registered plan: docs/HIGH_END_PLAN.md. Starters only; ratios are actual/predicted points with fixture-cluster bootstrap 95% intervals.

**Step 1 (correctness fixes): adopted.**

**Step 2 (calibration): chosen = spline.**

| variant   |   overall actual/xP |   overall lo |   overall hi |   top-20 |   top-10 |   top-10 lo |   top-10 hi |   top-5 |   £8m+ |   PIT 50% |   PIT 80% |   RMSE |   within-GW Spearman |   top-10 mean actual |   yellow obs/pred | bonus/fixture lowest quintile pred|obs   | bonus/fixture highest quintile pred|obs   | top-10 bonus pred|obs   |
|:----------|--------------------:|-------------:|-------------:|---------:|---------:|------------:|------------:|--------:|-------:|----------:|----------:|-------:|---------------------:|---------------------:|------------------:|:-----------------------------------------|:------------------------------------------|:------------------------|
| before    |              0.9781 |       0.9711 |       0.985  |   0.9621 |   0.9572 |      0.9245 |      0.9913 |  0.9563 | 0.9619 |    0.4894 |    0.7894 | 3.0008 |               0.3116 |               5.9317 |            1.0978 | 5.40 | 6.09                              | 7.31 | 6.18                               | 0.815 | 0.715           |
| fixes     |              0.9878 |       0.9812 |       0.9943 |   0.9667 |   0.965  |      0.9302 |      0.998  |  0.9687 | 0.9672 |    0.4922 |    0.7904 | 2.9991 |               0.3106 |               5.9454 |            1.0056 | 6.14 | 6.18                              | 6.24 | 6.19                               | 0.809 | 0.731           |
| final     |              0.9871 |       0.98   |       0.9937 |   0.9707 |   0.9818 |      0.949  |      1.0141 |  1.0017 | 0.9761 |    0.49   |    0.7856 | 2.9981 |               0.3144 |               5.9626 |            1.0051 | 6.14 | 6.11                              | 6.23 | 6.15                               | 0.803 | 0.723           |

## Top 10 per gameweek: predicted minus actual points by component

|             |   before |   fixes |   final |
|:------------|---------:|--------:|--------:|
| appearance  |    0.014 |   0.014 |   0.013 |
| goals       |    0.084 |   0.037 |   0.004 |
| assists     |    0.044 |   0.053 |  -0.001 |
| clean sheet |    0.038 |   0.05  |   0.031 |
| conceded    |    0.004 |   0.005 |   0.004 |
| bonus       |    0.1   |   0.077 |   0.08  |
| cards       |   -0.018 |  -0.02  |  -0.018 |
| other       |   -0.001 |  -0.001 |  -0.002 |

## Yellow cards, actual/predicted by season

| season   |   before |   fixes |   final |
|:---------|---------:|--------:|--------:|
| 2019-20  |    1.087 |   1.018 |   1.017 |
| 2020-21  |    0.97  |   0.926 |   0.926 |
| 2021-22  |    1.133 |   1.08  |   1.08  |
| 2022-23  |    1.11  |   1.018 |   1.018 |
| 2023-24  |    1.169 |   1.036 |   1.034 |
| 2024-25  |    1.106 |   0.959 |   0.958 |
