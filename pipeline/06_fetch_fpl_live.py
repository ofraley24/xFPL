#!/usr/bin/env python3
"""Snapshot the official FPL API (current season).

Saves, under data/raw/fpl_api/<UTC timestamp>/ :
  bootstrap.json          players (incl. injury/news/price/ownership/ICT/xG/xA/DefCon),
                          teams (incl. FPL strength ratings), events (gameweeks, deadlines),
                          scoring rules (game_config.scoring), element_types
  fixtures.json           every fixture with FPL difficulty ratings and per-fixture stats
  element_summaries.json  for EVERY player: `history` (one row per match played or
                          scheduled-with-zero-minutes: minutes, goals, assists, clean sheets,
                          bonus, BPS, ICT components, xG, xA, xGI, xGC, defensive
                          contribution, price, ownership, transfers, ...), plus upcoming
                          `fixtures` and `history_past` (season totals of earlier seasons)

WHY A TIMESTAMPED SNAPSHOT EVERY TIME. Injury/availability news, chance of
playing, prices and ownership are live values; the API keeps no history of
them. A prediction made at a deadline may only use what was known at that
moment, so the only leak-free way to build a training set of such features is
to record snapshots as the season unfolds. Run this before every deadline (and
ideally daily). Snapshots are never overwritten.
"""
import argparse
import gzip
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; xFPL project)"}
RAW = DATA / "raw" / "fpl_api"
PAUSE = 0.4


def get(path, retries=4):
    for i in range(retries):
        r = requests.get(f"{BASE}/{path}", headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 503):
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"{path}: gave up after {retries} tries (last status {r.status_code})")


def season_label(bootstrap):
    m = re.search(r"/(\d{4})_(\d{2})/", bootstrap["game_config"]["settings"]["static_content_url"])
    return f"{m.group(1)}-{m.group(2)}" if m else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-players", action="store_true",
                    help="only bootstrap+fixtures (fast; for frequent injury snapshots)")
    a = ap.parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RAW / ts
    out.mkdir(parents=True)

    boot = get("bootstrap-static/")
    (out / "bootstrap.json").write_text(json.dumps(boot))
    time.sleep(PAUSE)
    (out / "fixtures.json").write_text(json.dumps(get("fixtures/")))
    ids = [e["id"] for e in boot["elements"]]
    rec = {"snapshot_utc": ts, "season": season_label(boot), "players_in_bootstrap": len(ids),
           "current_event": next((e["id"] for e in boot["events"] if e["is_current"]), None),
           "element_summaries": 0, "failed": []}
    if not a.skip_players:
        summaries = {}
        for i, pid in enumerate(ids, 1):
            try:
                summaries[pid] = get(f"element-summary/{pid}/")
            except Exception as e:  # keep going; failures are logged
                rec["failed"].append({"id": pid, "error": str(e)[:120]})
            time.sleep(PAUSE)
            if i % 100 == 0:
                print(f"  {i}/{len(ids)} players", flush=True)
        with gzip.open(out / "element_summaries.json.gz", "wt") as f:
            json.dump(summaries, f)
        rec["element_summaries"] = len(summaries)
    DOCS.mkdir(exist_ok=True)
    with open(DOCS / "fpl_fetch_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec)


if __name__ == "__main__":
    main()
