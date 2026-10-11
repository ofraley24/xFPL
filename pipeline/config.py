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
