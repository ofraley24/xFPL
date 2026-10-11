#!/usr/bin/env python3
"""Merge freshly fetched API seasons into the legacy-schema Understat files.

For every (league, season) present in the API cache, ALL legacy rows of that
season are replaced by the freshly fetched ones (replacement, not key-dedupe, so
corrections to earlier matches in the season are picked up and stale rows can
never survive). Seasons not in the cache are copied through untouched. Output
goes to data/understat_merged/ with the legacy column order, so legacy-reading
code works unchanged. The original understat/ snapshot is never modified.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (LEAGUES, LEGACY_UNDERSTAT, MATCH_COLS, MERGED, PLAYER_COLS,  # noqa: E402
                    PMS_COLS, RAW_API, SHOT_COLS, TEAM_COLS)


def load(p):
    return json.loads(Path(p).read_text())


def blank(v):
    return "" if v is None else v


def matches_frame(league, season, rows):
    out = []
    for m in rows:
        played = bool(m.get("isResult"))
        g, x, f = m.get("goals") or {}, m.get("xG") or {}, m.get("forecast") or {}
        out.append({
            "league": league, "season": season, "match_id": m["id"],
            "isResult": "true" if played else "false", "date": m["datetime"],
            "home_team": m["h"]["title"], "away_team": m["a"]["title"],
            "home_goals": blank(g.get("h")) if played else "",
            "away_goals": blank(g.get("a")) if played else "",
            "home_xG": blank(x.get("h")) if played else "",
            "away_xG": blank(x.get("a")) if played else "",
            "forecast_w": blank(f.get("w")) if played else "",
            "forecast_d": blank(f.get("d")) if played else "",
            "forecast_l": blank(f.get("l")) if played else ""})
    return pd.DataFrame(out, columns=MATCH_COLS)


def teams_frame(league, season, teams):
    out = []
    for t in teams.values():
        for h in t["history"]:
            out.append({
                "league": league, "season": season, "team_id": t["id"],
                "team_title": t["title"], "date": h["date"], "h_a": h["h_a"],
                "xG": h["xG"], "xGA": h["xGA"], "npxG": h["npxG"], "npxGA": h["npxGA"],
                "ppda_att": h["ppda"]["att"], "ppda_def": h["ppda"]["def"],
                "ppda_allowed_att": h["ppda_allowed"]["att"],
                "ppda_allowed_def": h["ppda_allowed"]["def"],
                "deep": h["deep"], "deep_allowed": h["deep_allowed"],
                "scored": h["scored"], "missed": h["missed"], "xpts": h["xpts"],
                "result": h["result"], "wins": h["wins"], "draws": h["draws"],
                "loses": h["loses"], "pts": h["pts"], "npxGD": h["npxGD"]})
    return pd.DataFrame(out, columns=TEAM_COLS)


def players_frame(league, season, players):
    df = pd.DataFrame(players).rename(columns={"id": "player_id"})
    df["league"], df["season"] = league, season
    return df.reindex(columns=PLAYER_COLS)


def per_match_frames(league, season, base):
    shots, roster = [], []
    for f in sorted((base / "shots").glob("*.json")):
        d = load(f)
        for side in ("h", "a"):
            for s in d[side]:
                shots.append({**s, "league": league, "season": season,
                              "shot_id": s["id"]})
    for f in sorted((base / "rosters").glob("*.json")):
        mid = f.stem
        d = load(f)
        for side in ("h", "a"):
            for r in d[side].values():
                roster.append({**r, "league": league, "season": season, "match_id": mid,
                               "roster_id": r["id"]})
    return (pd.DataFrame(shots).reindex(columns=SHOT_COLS),
            pd.DataFrame(roster).reindex(columns=PMS_COLS))


def splice(legacy, new_by_season, name, league):
    """Drop legacy rows for refreshed seasons, append the fresh rows."""
    old = pd.read_csv(LEGACY_UNDERSTAT / f"{league}_{name}.csv", dtype=str,
                      keep_default_na=False)
    seasons = set(new_by_season)
    kept = old[~old["season"].isin(seasons)]
    fresh = [df.astype(str).replace("nan", "").replace("None", "") for df in new_by_season.values()]
    out = pd.concat([kept] + fresh, ignore_index=True)[list(old.columns)]
    print(f"  {league:10s} {name:18s} legacy {len(old):>7,} -> merged {len(out):>7,} "
          f"(refreshed seasons {sorted(seasons)})")
    return out


def main():
    MERGED.mkdir(parents=True, exist_ok=True)
    for league in LEAGUES:
        cached = sorted(p.name for p in (RAW_API / league).glob("*") if p.is_dir()) \
            if (RAW_API / league).exists() else []
        fresh = {k: {} for k in ("matches", "team_match_stats", "players",
                                 "player_match_stats", "shots")}
        for season in cached:
            base = RAW_API / league / season
            fresh["matches"][season] = matches_frame(league, season, load(base / "matches.json"))
            fresh["team_match_stats"][season] = teams_frame(league, season, load(base / "teams.json"))
            fresh["players"][season] = players_frame(league, season, load(base / "players.json"))
            sh, pms = per_match_frames(league, season, base)
            fresh["shots"][season], fresh["player_match_stats"][season] = sh, pms
        for name, by_season in fresh.items():
            splice(None, by_season, name, league).to_csv(MERGED / f"{league}_{name}.csv",
                                                         index=False)


if __name__ == "__main__":
    main()
