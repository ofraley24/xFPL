# xFPL: expected-points model for Fantasy Premier League

## Objective
Predict the Fantasy Premier League (FPL) points each player will score in each
gameweek, treated as a research project that must withstand academic scrutiny.
The approach has three stages:

1. **Fixture strength.** Estimate team strength separately for attack and
   defence, so that each fixture yields an expected-goals-for and
   expected-goals-against for both sides.
2. **Stratified scoring-mechanism models.** Model each way a player earns FPL
   points separately (minutes/appearance, goals, assists, clean sheets and
   goals conceded, saves, bonus, cards, penalties, defensive contributions).
3. **Expected points (xP).** Combine the components into xP per player per
   gameweek, validated out-of-sample against actual FPL points and against
   baselines.

## Current status
Data layer only (Understat and FPL). No modelling is included in this repository yet.

## Repository contents
| path | purpose |
|---|---|
| `pipeline/01_fetch_understat.py` | fetch fixtures, team history, player totals, and per-match shots and rosters from Understat (EPL, La Liga, Bundesliga, Serie A, Ligue 1, RFPL) into a local JSON cache; logs every fetch |
| `pipeline/02_merge_understat.py` | splice freshly fetched seasons into the historical snapshot, keeping the legacy CSV schema |
| `pipeline/02b_clean_understat.py` | remove duplicate shot/roster records that Understat itself serves; every removal is logged |
| `pipeline/05_validate_data.py` | 100+ automated data-quality checks, written to `docs/DATA_QC.md` |
| `pipeline/06_fetch_fpl_live.py` | timestamped snapshot of the official FPL API: players (injury status/news/chance of playing, prices, ownership, set-piece orders), teams, fixtures, scoring rules, and per-player per-match history (ICT, BPS, xG, xA, xGC, defensive contribution) |
| `pipeline/07_fetch_fpl_history.py` | FPL history 2016-17 onward from the vaastav archive |
| `pipeline/08_build_fpl_tables.py` | unified tables in `data/fpl/` (player-match, player ids, snapshot history, teams, events, fixtures, rules) |
| `pipeline/09_validate_fpl.py` | FPL QC, plus recovery of the scoring rules per season from the data (`docs/FPL_QC.md`, `docs/FPL_RULES.md`) |
| `pipeline/10_crosswalk_fpl_understat.py` | FPL <-> Understat crosswalk: fixtures (kickoff + score), teams (vote, one-to-one), players (name + minutes assignment, consolidated into a persistent FPL `code` -> Understat `player_id` map); report in `docs/CROSSWALK_QC.md` |
| `pipeline/config.py` | paths, league list, column schemas |
| `docs/data_fetch_log.jsonl` | provenance: when each league-season was fetched, with package version |
| `docs/dedup_log.csv` | audit trail of rows removed by the cleaning step |
| `docs/DATA_QC.md` | latest QC report |
| `understat_MANIFEST.sha256` | checksums of the frozen historical snapshot (2014 to 2026-09-01) |

## Running
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python pipeline/run_all.py
```
The raw and derived data (`understat/`, `data/`) are not stored in git.

## Data notes
- Source: Understat via `understatapi`. Check Understat's terms of use before
  redistributing any data or derived work.
- The historical snapshot (2014 to 2026-09-01) predates this repository; its
  integrity is tracked by checksum. Rows from season 2026 onward are replaced
  by freshly fetched data on each refresh.
- Known upstream defect: some matches are served with every shot and roster row
  duplicated. The cleaning step removes these and logs them.
