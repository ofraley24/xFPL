#!/usr/bin/env python3
"""Fetch Understat data for one or more seasons through `understatapi`.

Caching: league-level tables (fixtures, team history, player totals) are
re-fetched every run because they change weekly. Per-match shot and roster JSON
is fetched once for each PLAYED match and cached, so reruns are cheap and
resumable. Every run appends a record to docs/data_fetch_log.jsonl.

Usage:
    python pipeline/01_fetch_understat.py --seasons 2026
    python pipeline/01_fetch_understat.py --seasons 2025 --leagues EPL   # parity check
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DOCS, LEAGUES, RAW_API  # noqa: E402

from importlib.metadata import version as _pkg_version  # noqa: E402
from understatapi import UnderstatClient  # noqa: E402

API_NAME = {"La_liga": "La_Liga"}  # our file prefix -> understatapi league name
PAUSE = 0.5 # seconds between requests; be polite to the site


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def fetch_league_season(client, league, season):
    base = RAW_API / league / season
    lg = client.league(league=API_NAME.get(league, league))
    matches = lg.get_match_data(season=season)
    dump(base / "matches.json", matches)
    time.sleep(PAUSE)
    dump(base / "teams.json", lg.get_team_data(season=season))
    time.sleep(PAUSE)
    dump(base / "players.json", lg.get_player_data(season=season))
    time.sleep(PAUSE)
    played = [m["id"] for m in matches if m.get("isResult")]
    new = failed = 0
    for mid in played:
        f_shots, f_roster = base / "shots" / f"{mid}.json", base / "rosters" / f"{mid}.json"
        if f_shots.exists() and f_roster.exists():
            continue
        try:
            m = client.match(match=mid)
            dump(f_shots, m.get_shot_data())
            time.sleep(PAUSE)
            dump(f_roster, m.get_roster_data())
            time.sleep(PAUSE)
            new += 1
        except Exception as e:  # keep going; the log records what is missing
            failed += 1
            print(f"  !! {league} {season} match {mid}: {e}", file=sys.stderr)
    return {"league": league, "season": season, "fixtures": len(matches),
            "played": len(played), "newly_fetched": new, "failed": failed,
            "last_played": max((m["datetime"] for m in matches if m.get("isResult")),
                               default=None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2026"])
    ap.add_argument("--leagues", nargs="+", default=LEAGUES)
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    with UnderstatClient() as client, open(DOCS / "data_fetch_log.jsonl", "a") as log:
        for season in args.seasons:
            for league in args.leagues:
                rec = fetch_league_season(client, league, season)
                rec["fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                rec["understatapi_version"] = _pkg_version("understatapi")
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec)


if __name__ == "__main__":
    main()
