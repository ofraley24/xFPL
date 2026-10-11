#!/usr/bin/env python3
"""Download historical FPL data from the vaastav/Fantasy-Premier-League archive.

Per season it fetches, unmodified, to data/raw/vaastav/<season>/ :
  gws/merged_gw.csv   one row per player per fixture (GW), all stats the season has
  players_raw.csv     end-of-season player table: maps element id -> persistent `code`,
                      position, team, and the news/status at scrape time
  fixtures.csv        fixtures with FPL difficulty ratings
  teams.csv           teams with FPL strength ratings
Each download is logged with its URL, SHA-256 and time in docs/vaastav_fetch_log.jsonl.

Column availability differs by season (expected_* from 2022-23; defensive
contribution from 2025-26); that is handled in 08_build_fpl_tables.py, not here.
Existing files are skipped unless --refresh is given.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402

RAW_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
           "2023-24", "2024-25", "2025-26", "2026-27"]
FILES = ["gws/merged_gw.csv", "players_raw.csv", "fixtures.csv", "teams.csv"]
OUT = DATA / "raw" / "vaastav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    with open(DOCS / "vaastav_fetch_log.jsonl", "a") as log:
        for season in a.seasons:
            for f in FILES:
                dest = OUT / season / f
                if dest.exists() and not a.refresh:
                    continue
                url = f"{RAW_URL}/{season}/{f}"
                r = requests.get(url, timeout=60)
                rec = {"url": url, "status": r.status_code,
                       "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                if r.status_code == 200:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(r.content)
                    rec.update(bytes=len(r.content), sha256=hashlib.sha256(r.content).hexdigest())
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(season, f, r.status_code, rec.get("bytes"))
                time.sleep(0.3)


if __name__ == "__main__":
    main()
