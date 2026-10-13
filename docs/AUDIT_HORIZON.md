# Truncation test of the horizon rows (2026-10-13)

Rows predicted h >= 2 gameweeks ahead; every feature recomputed from records dated before the earlier deadline (`analysis/23_audit_horizon.py`).

| check                                                                             | result   | detail                                    |
|:----------------------------------------------------------------------------------|:---------|:------------------------------------------|
| H0 sample covers horizons 2-8 and every season                                    | PASS     | horizons [2, 3, 4, 5, 6, 7, 8], seasons 7 |
| H0 every row is predicted from a deadline before its kickoff, h gameweeks earlier | PASS     | 413,139 rows                              |
| H1a Understat form recomputed from raw rows before the earlier deadline           | PASS     | 4000 comparisons, max rel err 3.25e-15    |
| H1b FPL minutes features recomputed from rows before the earlier deadline         | PASS     | 399 rows x 8 features, mismatches 0       |
| H1c FPL event lags (L365) recomputed from rows before the earlier deadline        | PASS     | 400 rows x 4 features, mismatches 0       |
| H1d team lambda reproduced by a replay truncated at the earlier deadline          | PASS     | 10 fixtures, max |d lambda| = 1.11e-16    |
