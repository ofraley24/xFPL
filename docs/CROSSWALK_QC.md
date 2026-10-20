# FPL <-> Understat crosswalk QC (2026-10-20)

## Fixtures (FPL fixture -> Understat match)

|         |   fpl_fixtures |   matched |   understat_matches |   fpl_unmatched |   dup_use |
|:--------|---------------:|----------:|--------------------:|----------------:|----------:|
| 2016-17 |            380 |       380 |                 380 |               0 |         0 |
| 2017-18 |            380 |       380 |                 380 |               0 |         0 |
| 2018-19 |            380 |       380 |                 380 |               0 |         0 |
| 2019-20 |            380 |       380 |                 380 |               0 |         0 |
| 2020-21 |            380 |       380 |                 380 |               0 |         0 |
| 2021-22 |            380 |       380 |                 380 |               0 |         0 |
| 2022-23 |            380 |       380 |                 380 |               0 |         0 |
| 2023-24 |            380 |       380 |                 380 |               0 |         0 |
| 2024-25 |            380 |       380 |                 380 |               0 |         0 |
| 2025-26 |            380 |       380 |                 380 |               0 |         0 |
| 2026-27 |             70 |        70 |                  70 |               0 |         0 |

Method used:

| season   |   score+time |   score+time+teams |
|:---------|-------------:|-------------------:|
| 2016-17  |          261 |                119 |
| 2017-18  |          238 |                142 |
| 2018-19  |          243 |                137 |
| 2019-20  |          236 |                144 |
| 2020-21  |          259 |                121 |
| 2021-22  |          269 |                111 |
| 2022-23  |          231 |                149 |
| 2023-24  |          265 |                115 |
| 2024-25  |          226 |                154 |
| 2025-26  |          224 |                156 |
| 2026-27  |           42 |                 28 |

## Teams

| season   |   teams |   min_vote_share | bijective   |
|:---------|--------:|-----------------:|:------------|
| 2016-17  |      20 |                1 | True        |
| 2017-18  |      20 |                1 | True        |
| 2018-19  |      20 |                1 | True        |
| 2019-20  |      20 |                1 | True        |
| 2020-21  |      20 |                1 | True        |
| 2021-22  |      20 |                1 | True        |
| 2022-23  |      20 |                1 | True        |
| 2023-24  |      20 |                1 | True        |
| 2024-25  |      20 |                1 | True        |
| 2025-26  |      20 |                1 | True        |
| 2026-27  |      20 |                1 | True        |

## Players: coverage of FPL appearances (minutes > 0) in matched fixtures

| season   |   fpl_rows_min>0 |   matched |   rate |   strong_or_pinned_rate |
|:---------|-----------------:|----------:|-------:|------------------------:|
| 2016-17  |            10474 |     10474 | 1      |                  1      |
| 2017-18  |            10448 |     10447 | 0.9999 |                  0.9999 |
| 2018-19  |            10480 |     10480 | 1      |                  1      |
| 2019-20  |            10614 |     10614 | 1      |                  1      |
| 2020-21  |            10393 |     10393 | 1      |                  1      |
| 2021-22  |            10485 |     10485 | 1      |                  0.9999 |
| 2022-23  |            11345 |     11345 | 1      |                  1      |
| 2023-24  |            11384 |     11384 | 1      |                  0.9999 |
| 2024-25  |            11566 |     11566 | 1      |                  1      |
| 2025-26  |            11492 |     11490 | 0.9998 |                  0.9996 |
| 2026-27  |             2158 |      2158 | 1      |                  0.9991 |

Understat appearances that found an FPL partner:

| season   |   understat rows matched |
|:---------|-------------------------:|
| 2016-17  |                   1      |
| 2017-18  |                   1      |
| 2018-19  |                   1      |
| 2019-20  |                   1      |
| 2020-21  |                   1      |
| 2021-22  |                   1      |
| 2022-23  |                   1      |
| 2023-24  |                   1      |
| 2024-25  |                   0.9999 |
| 2025-26  |                   1      |
| 2026-27  |                   1      |

## Players: agreement of independent attributes on matched pairs

| season   |   minutes within 3 |   goals equal |   assists equal |
|:---------|-------------------:|--------------:|----------------:|
| 2016-17  |             0.9334 |        1      |          0.9796 |
| 2017-18  |             0.9565 |        0.9998 |          0.9829 |
| 2018-19  |             0.9477 |        0.9998 |          0.9823 |
| 2019-20  |             0.8927 |        0.9998 |          0.9817 |
| 2020-21  |             0.9431 |        1      |          0.9791 |
| 2021-22  |             0.9053 |        1      |          0.9825 |
| 2022-23  |             0.8412 |        1      |          0.9824 |
| 2023-24  |             0.7309 |        1      |          0.9819 |
| 2024-25  |             0.7935 |        1      |          0.9854 |
| 2025-26  |             0.7713 |        0.9999 |          0.978  |
| 2026-27  |             0.8573 |        1      |          0.9801 |

## Evidence that does not depend on goals/assists (which are inputs to the match cost)

- pairs where the NAMES alone agree (name cost <= 0.25): 99.97% of 110,836; <= 0.45: 100.00%
- one-to-one persistent map: Understat ids per FPL code > 1: 0; FPL codes per Understat id > 1: 0
- pairs that contradict the persistent map: 0

Weak pairs left after pass 2 (name or minutes disagree; review manually):

| season   |   fixture | fpl_name            | us_name             |   fpl_minutes |   us_minutes |   name_cost |
|:---------|----------:|:--------------------|:--------------------|--------------:|-------------:|------------:|
| 2021-22  |       300 | kristoffer klaesson | Kristoffer Klaesson |            35 |           24 |           0 |
| 2023-24  |       113 | ben parkinson       | Ben Parkinson       |            24 |           18 |           0 |
| 2025-26  |       345 | jesse derry         | Jesse Derry         |            45 |           53 |           0 |
| 2025-26  |       372 | jerome abbey        | Jerome Abbey        |            16 |            6 |           0 |
| 2025-26  |       374 | sean neave          | Sean Neave          |            13 |            7 |           0 |
| 2026-27  |        61 | jules ahoka         | Jules Ahoka         |            25 |           19 |           0 |
| 2026-27  |        61 | tymur tutierov      | Tymur Tutierov      |            58 |           65 |           0 |

## Persistent map

2,005 FPL codes mapped; 0 flagged as conflicts (see player_crosswalk.csv).
