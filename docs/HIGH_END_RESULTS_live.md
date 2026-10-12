# High-end and yellow-card fixes: results (live)

Pre-registered plan: docs/HIGH_END_PLAN.md. Starters only; ratios are actual/predicted points with fixture-cluster bootstrap 95% intervals.

| variant   |   overall actual/xP |   overall lo |   overall hi |   top-20 |   top-10 |   top-10 lo |   top-10 hi |   top-5 |   £8m+ |   PIT 50% |   PIT 80% |   RMSE |   within-GW Spearman |   top-10 mean actual |   yellow obs/pred | bonus/fixture lowest quintile pred|obs   | bonus/fixture highest quintile pred|obs   | top-10 bonus pred|obs   |
|:----------|--------------------:|-------------:|-------------:|---------:|---------:|------------:|------------:|--------:|-------:|----------:|----------:|-------:|---------------------:|---------------------:|------------------:|:-----------------------------------------|:------------------------------------------|:------------------------|
| before    |              1.0037 |       0.9647 |       1.0462 |   0.9075 |   0.9932 |      0.803  |      1.1926 |  1.1125 | 0.9673 |    0.4851 |    0.7813 | 3.1936 |               0.1958 |               5.7333 |            1.0945 | 5.32 | 6.00                              | 7.03 | 6.33                               | 0.697 | 0.717           |
| final     |              1.0129 |       0.9682 |       1.0527 |   0.9622 |   1.046  |      0.8191 |      1.2649 |  0.9428 | 0.9751 |    0.4867 |    0.7798 | 3.2007 |               0.2004 |               5.95   |            0.9959 | 6.15 | 6.42                              | 6.17 | 6.17                               | 0.712 | 0.767           |

## Top 10 per gameweek: predicted minus actual points by component

|             |   before |   final |
|:------------|---------:|--------:|
| appearance  |   -0.043 |  -0.041 |
| goals       |   -0.173 |  -0.24  |
| assists     |   -0.018 |  -0.097 |
| clean sheet |    0.224 |   0.139 |
| conceded    |    0.072 |   0.045 |
| bonus       |   -0.019 |  -0.054 |
| cards       |   -0.064 |  -0.055 |
| other       |    0.002 |   0.003 |
| def. contr. |    0.06  |   0.039 |

## Yellow cards, actual/predicted by season

| season   |   before |   final |
|:---------|---------:|--------:|
| 2026-27  |    1.095 |   0.996 |
