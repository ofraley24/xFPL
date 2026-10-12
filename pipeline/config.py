"""Shared paths and constants. Every pipeline script imports from here so that
no script carries an absolute or sandbox path."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGACY_UNDERSTAT = ROOT / "understat"          # immutable snapshot, 2014 - 2026-09-01
DATA = ROOT / "data"                           # git-ignored, regenerable
RAW_API = DATA / "raw" / "understat_api"       # per-match JSON cache from the API
MERGED = DATA / "understat_merged"             # legacy snapshot + refreshed seasons
INTERIM = DATA / "interim"                     # derived tables
DOCS = ROOT / "docs"

SEED = 20261011
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]

# Legacy column order, so downstream legacy code reads merged files unchanged.
MATCH_COLS = ["league", "season", "match_id", "isResult", "date", "home_team", "away_team",
              "home_goals", "away_goals", "home_xG", "away_xG",
              "forecast_w", "forecast_d", "forecast_l"]
TEAM_COLS = ["league", "season", "team_id", "team_title", "date", "h_a", "xG", "xGA", "npxG",
             "npxGA", "ppda_att", "ppda_def", "ppda_allowed_att", "ppda_allowed_def", "deep",
             "deep_allowed", "scored", "missed", "xpts", "result", "wins", "draws", "loses",
             "pts", "npxGD"]
PLAYER_COLS = ["league", "season", "player_id", "player_name", "team_title", "games", "time",
               "goals", "xG", "assists", "xA", "shots", "key_passes", "yellow_cards",
               "red_cards", "position", "npg", "npxG", "xGChain", "xGBuildup"]
PMS_COLS = ["league", "season", "match_id", "roster_id", "player_id", "player", "team_id", "h_a",
            "position", "time", "goals", "own_goals", "shots", "xG", "assists", "xA", "xGChain",
            "xGBuildup", "yellow_card", "red_card", "roster_in", "roster_out", "positionOrder"]
SHOT_COLS = ["league", "season", "match_id", "shot_id", "minute", "result", "X", "Y", "xG",
             "player", "player_id", "h_a", "situation", "shotType", "h_team", "a_team",
             "h_goals", "a_goals", "date", "player_assisted", "lastAction"]

# Matches played behind closed doors (COVID-19), by league. Approximate public dates; partial-capacity
# spells are classified as noted. RFPL is treated as having crowds (capacity-limited, never fully closed).
#   EPL: restart 2020-06-17 .. last closed round before limited fans returned 2021-05-18 (the ~2k-fan
#        matches of Dec 2020 are treated as closed). La Liga: restart 2020-06-11 .. end of 2020-21.
#   Serie A: restart 2020-06-20 .. end of 2020-21. Bundesliga: restart 2020-05-16 .. end of 2020-21
#        (small capped crowds Sep-Oct 2020 treated as closed). Ligue 1: 2019-20 abandoned; 2020-21 capped
#        crowds until the late-Oct 2020 lockdown, closed from 2020-10-30 .. end of 2020-21.
BCD_PERIODS = {
    "EPL": [("2020-06-17", "2021-05-17")],
    "La_liga": [("2020-06-11", "2021-05-23")],
    "Serie_A": [("2020-06-20", "2021-05-23")],
    "Bundesliga": [("2020-05-16", "2021-05-22")],
    "Ligue_1": [("2020-10-30", "2021-05-23")],
    "RFPL": [],
}
# Maximum substitutes per match in the EPL: 5 from the June 2020 restart to the end of 2019-20 and from
# 2022-23 onward; 3 otherwise.
FIVE_SUB_PERIODS_EPL = [("2020-06-17", "2020-07-26"), ("2022-08-01", "2099-12-31")]


def in_periods(dates, periods):
    import pandas as _pd
    d = _pd.to_datetime(dates)
    out = _pd.Series(False, index=getattr(dates, "index", None))
    for a, b in periods:
        out |= (d >= _pd.Timestamp(a)) & (d <= _pd.Timestamp(b) + _pd.Timedelta(days=1))
    return out.to_numpy()
