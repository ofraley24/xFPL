#!/usr/bin/env python3
"""Remove duplicated records that Understat itself serves, writing data/understat_clean/.

Found in the 2026-10-11 audit: for some matches Understat returns every shot and
every roster row twice, under new consecutive ids (legacy snapshot: La Liga 2023
and 2025, RFPL 2025; fresh data: La Liga match 30804). Duplicated shots inflate
goals/xG per match (30804: 10 shot-goals for a 5-goal match) and duplicated roster
rows double a player's minutes and shots.

Rules (deterministic, keep first occurrence):
  shots   duplicate if (match_id, minute, player_id, result, X, Y, xG) repeat
          -- identical coordinates AND xG to full float precision cannot be two real shots
  roster  duplicate if (match_id, player_id) repeats
Every removed row is written to docs/dedup_log.csv so the cleaning is auditable.
data/understat_merged/ (and the understat/ snapshot) are never modified.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS, LEAGUES, MERGED  # noqa: E402

CLEAN = DATA / "understat_clean"
SHOT_KEY = ["match_id", "minute", "player_id", "result", "X", "Y", "xG"]
PMS_KEY = ["match_id", "player_id"]


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    log = []
    for lg in LEAGUES:
        for kind in ["matches", "team_match_stats", "players", "player_match_stats", "shots"]:
            df = pd.read_csv(MERGED / f"{lg}_{kind}.csv", dtype=str, keep_default_na=False)
            key = {"shots": SHOT_KEY, "player_match_stats": PMS_KEY}.get(kind)
            if key:
                dup = df.duplicated(key, keep="first")
                for (mid, season), n in df[dup].groupby(["match_id", "season"]).size().items():
                    log.append({"league": lg, "table": kind, "season": season,
                                "match_id": mid, "rows_removed": n})
                df = df[~dup]
            df.to_csv(CLEAN / f"{lg}_{kind}.csv", index=False)
    out = pd.DataFrame(log, columns=["league", "table", "season", "match_id", "rows_removed"])
    out.to_csv(DOCS / "dedup_log.csv", index=False)
    print(out.groupby(["league", "table", "season"]).agg(matches=("match_id", "nunique"),
                                                         rows=("rows_removed", "sum")))


if __name__ == "__main__":
    main()
