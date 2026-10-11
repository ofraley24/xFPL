# Data QC report (2026-10-11)

102/102 checks passed.

## Freshness (C8)

| league     |   season_2026_fixtures |   played | last_played         |
|:-----------|-----------------------:|---------:|:--------------------|
| EPL        |                    380 |       56 | 2026-10-10 16:30:00 |
| La_liga    |                    380 |       73 | 2026-10-10 19:00:00 |
| Bundesliga |                    306 |       43 | 2026-10-10 16:30:00 |
| Serie_A    |                    380 |       54 | 2026-10-11 10:30:00 |
| Ligue_1    |                    306 |       51 | 2026-10-10 18:45:00 |
| RFPL       |                    240 |       78 | 2026-10-11 11:00:00 |

## Checks

| check | result | detail |
|---|---|---|
| C1 EPL matches: pre-2026 history identical | PASS | 4,560 rows |
| C1 EPL team_match_stats: pre-2026 history identical | PASS | 9,120 rows |
| C1 EPL players: pre-2026 history identical | PASS | 6,424 rows |
| C1 EPL player_match_stats: pre-2026 history identical | PASS | 129,576 rows |
| C1 EPL shots: pre-2026 history identical | PASS | 116,448 rows |
| C2 EPL match_id unique | PASS |  |
| C2 EPL (match_id,player_id) unique in roster | PASS | 0 dups |
| C2 EPL shot_id unique | PASS | 0 dups |
| C2 EPL team+date unique | PASS |  |
| C3 EPL 2026 roster match_ids all played | PASS |  |
| C3 EPL 2026 shot match_ids all played | PASS |  |
| C4 EPL 2026 every played match has >=11 roster rows per side | PASS | 56/56 matches with rosters |
| C5 EPL 2026 total goals: matches == team rows | PASS | 161 vs 161 |
| C5 EPL 2026 shot goals+own goals == match goals | PASS | 161 vs 161 |
| C6 EPL 2026 team names: matches vs team table | PASS | diff=[] |
| C7 EPL shot xG in [0,1] | PASS |  |
| C7 EPL minutes in [0,130] | PASS |  |
| C1 La_liga matches: pre-2026 history identical | PASS | 4,560 rows |
| C1 La_liga team_match_stats: pre-2026 history identical | PASS | 9,120 rows |
| C1 La_liga players: pre-2026 history identical | PASS | 6,792 rows |
| C1 La_liga player_match_stats: pre-2026 history identical | PASS | 134,827 rows |
| C1 La_liga shots: pre-2026 history identical | PASS | 108,714 rows |
| C2 La_liga match_id unique | PASS |  |
| C2 La_liga (match_id,player_id) unique in roster | PASS | 0 dups |
| C2 La_liga shot_id unique | PASS | 0 dups |
| C2 La_liga team+date unique | PASS |  |
| C3 La_liga 2026 roster match_ids all played | PASS |  |
| C3 La_liga 2026 shot match_ids all played | PASS |  |
| C4 La_liga 2026 every played match has >=11 roster rows per side | PASS | 73/73 matches with rosters |
| C5 La_liga 2026 total goals: matches == team rows | PASS | 219 vs 219 |
| C5 La_liga 2026 shot goals+own goals == match goals | PASS | 219 vs 219 |
| C6 La_liga 2026 team names: matches vs team table | PASS | diff=[] |
| C7 La_liga shot xG in [0,1] | PASS |  |
| C7 La_liga minutes in [0,130] | PASS |  |
| C1 Bundesliga matches: pre-2026 history identical | PASS | 3,672 rows |
| C1 Bundesliga team_match_stats: pre-2026 history identical | PASS | 7,344 rows |
| C1 Bundesliga players: pre-2026 history identical | PASS | 5,813 rows |
| C1 Bundesliga player_match_stats: pre-2026 history identical | PASS | 107,938 rows |
| C1 Bundesliga shots: pre-2026 history identical | PASS | 95,651 rows |
| C2 Bundesliga match_id unique | PASS |  |
| C2 Bundesliga (match_id,player_id) unique in roster | PASS | 0 dups |
| C2 Bundesliga shot_id unique | PASS | 0 dups |
| C2 Bundesliga team+date unique | PASS |  |
| C3 Bundesliga 2026 roster match_ids all played | PASS |  |
| C3 Bundesliga 2026 shot match_ids all played | PASS |  |
| C4 Bundesliga 2026 every played match has >=11 roster rows per side | PASS | 43/43 matches with rosters |
| C5 Bundesliga 2026 total goals: matches == team rows | PASS | 166 vs 166 |
| C5 Bundesliga 2026 shot goals+own goals == match goals | PASS | 166 vs 166 |
| C6 Bundesliga 2026 team names: matches vs team table | PASS | diff=[] |
| C7 Bundesliga shot xG in [0,1] | PASS |  |
| C7 Bundesliga minutes in [0,130] | PASS |  |
| C1 Serie_A matches: pre-2026 history identical | PASS | 4,560 rows |
| C1 Serie_A team_match_stats: pre-2026 history identical | PASS | 9,120 rows |
| C1 Serie_A players: pre-2026 history identical | PASS | 6,887 rows |
| C1 Serie_A player_match_stats: pre-2026 history identical | PASS | 134,865 rows |
| C1 Serie_A shots: pre-2026 history identical | PASS | 118,820 rows |
| C2 Serie_A match_id unique | PASS |  |
| C2 Serie_A (match_id,player_id) unique in roster | PASS | 0 dups |
| C2 Serie_A shot_id unique | PASS | 0 dups |
| C2 Serie_A team+date unique | PASS |  |
| C3 Serie_A 2026 roster match_ids all played | PASS |  |
| C3 Serie_A 2026 shot match_ids all played | PASS |  |
| C4 Serie_A 2026 every played match has >=11 roster rows per side | PASS | 54/54 matches with rosters |
| C5 Serie_A 2026 total goals: matches == team rows | PASS | 154 vs 154 |
| C5 Serie_A 2026 shot goals+own goals == match goals | PASS | 154 vs 154 |
| C6 Serie_A 2026 team names: matches vs team table | PASS | diff=[] |
| C7 Serie_A shot xG in [0,1] | PASS |  |
| C7 Serie_A minutes in [0,130] | PASS |  |
| C1 Ligue_1 matches: pre-2026 history identical | PASS | 4,338 rows |
| C1 Ligue_1 team_match_stats: pre-2026 history identical | PASS | 8,474 rows |
| C1 Ligue_1 players: pre-2026 history identical | PASS | 6,658 rows |
| C1 Ligue_1 player_match_stats: pre-2026 history identical | PASS | 123,085 rows |
| C1 Ligue_1 shots: pre-2026 history identical | PASS | 102,852 rows |
| C2 Ligue_1 match_id unique | PASS |  |
| C2 Ligue_1 (match_id,player_id) unique in roster | PASS | 0 dups |
| C2 Ligue_1 shot_id unique | PASS | 0 dups |
| C2 Ligue_1 team+date unique | PASS |  |
| C3 Ligue_1 2026 roster match_ids all played | PASS |  |
| C3 Ligue_1 2026 shot match_ids all played | PASS |  |
| C4 Ligue_1 2026 every played match has >=11 roster rows per side | PASS | 51/51 matches with rosters |
| C5 Ligue_1 2026 total goals: matches == team rows | PASS | 143 vs 143 |
| C5 Ligue_1 2026 shot goals+own goals == match goals | PASS | 143 vs 143 |
| C6 Ligue_1 2026 team names: matches vs team table | PASS | diff=[] |
| C7 Ligue_1 shot xG in [0,1] | PASS |  |
| C7 Ligue_1 minutes in [0,130] | PASS |  |
| C1 RFPL matches: pre-2026 history identical | PASS | 2,880 rows |
| C1 RFPL team_match_stats: pre-2026 history identical | PASS | 5,760 rows |
| C1 RFPL players: pre-2026 history identical | PASS | 5,486 rows |
| C1 RFPL player_match_stats: pre-2026 history identical | PASS | 83,143 rows |
| C1 RFPL shots: pre-2026 history identical | PASS | 72,740 rows |
| C2 RFPL match_id unique | PASS |  |
| C2 RFPL (match_id,player_id) unique in roster | PASS | 0 dups |
| C2 RFPL shot_id unique | PASS | 0 dups |
| C2 RFPL team+date unique | PASS |  |
| C3 RFPL 2026 roster match_ids all played | PASS |  |
| C3 RFPL 2026 shot match_ids all played | PASS |  |
| C4 RFPL 2026 every played match has >=11 roster rows per side | PASS | 78/78 matches with rosters |
| C5 RFPL 2026 total goals: matches == team rows | PASS | 214 vs 214 |
| C5 RFPL 2026 shot goals+own goals == match goals | PASS | 214 vs 214 |
| C6 RFPL 2026 team names: matches vs team table | PASS | diff=[] |
| C7 RFPL shot xG in [0,1] | PASS |  |
| C7 RFPL minutes in [0,130] | PASS |  |
