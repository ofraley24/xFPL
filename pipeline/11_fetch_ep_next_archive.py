#!/usr/bin/env python3
"""Pre-deadline FPL ep_next for a past season, from the git history of olbauday/FPL-Core-Insights.

That repository commits a copy of the FPL API about twice a day into data/<season>/By Gameweek/GW<k>/playerstats.csv
(early in the season the GW<n> file appears when gameweek n-1 ends; later it appears only once gameweek n is under
way, so the state before deadline n sits in the GW<n-1> file). Any copy committed between the deadlines of
gameweeks n-1 and n has FPL's `is_next` = n, so its ep_next is FPL's own projection for gameweek n (gameweek 1:
any pre-season copy). For each gameweek the LATEST commit inside that window, of either file, is used; a copy from
before the window (stale) or after deadline n (post-deadline information) is never used.

The history is read from a blob-less clone in data/raw/fpl_core_insights/repo (created on first use, fetched on
later runs; no GitHub API rate limit). Each chosen copy is saved as
data/raw/fpl_core_insights/<season>/gw<nn>_<commit time>_<sha8>.csv (never overwritten) and logged to
docs/core_insights_fetch_log.jsonl. Deadlines come from master.parquet.
The repository carries no licence: the copies are for private research and are not redistributed.
Options: --season 2025-26 (default).
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
URL = "https://github.com/olbauday/FPL-Core-Insights.git"
BASE = ROOT / "data" / "raw" / "fpl_core_insights"
REPO = BASE / "repo"
LOG = ROOT / "docs" / "core_insights_fetch_log.jsonl"


def git(*args, binary=False):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=not binary).stdout


def main():
    season = sys.argv[sys.argv.index("--season") + 1] if "--season" in sys.argv else "2025-26"
    folder = f"{season[:4]}-20{season[5:]}"                     # 2025-26 -> 2025-2026
    out = BASE / season
    out.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        git("fetch", "-q", "--filter=blob:none", "origin")
    else:
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", URL, str(REPO)], check=True)
    m = pd.read_parquet(ROOT / "data" / "analysis" / "master.parquet", columns=["season", "round", "deadline"])
    deadlines = m[m.season == season].groupby("round").deadline.first()
    for gw, dl in deadlines.items():
        if list(out.glob(f"gw{gw:02d}_*.csv")):
            continue
        start = deadlines.get(gw - 1, pd.Timestamp(f"{season[:4]}-06-01"))
        best = None
        for k in (gw, gw - 1):
            path = f"data/{folder}/By Gameweek/GW{k}/playerstats.csv"
            line = git("log", "-1", "--format=%H %cI", f"--before={dl:%Y-%m-%dT%H:%M:%SZ}", "origin/HEAD", "--", path).strip()
            if line:
                sha, when = line.split()
                t = datetime.fromisoformat(when).astimezone(timezone.utc).replace(tzinfo=None)
                if best is None or t > best[1]:
                    best = (sha, t, path)
        if best is None or best[1] <= start:
            print(f"GW{gw}: no copy committed between {start} and the deadline {dl}; skipped")
            continue
        sha, t, path = best
        content = git("show", f"{sha}:{path}", binary=True)
        dest = out / f"gw{gw:02d}_{t:%Y%m%dT%H%M%SZ}_{sha[:8]}.csv"
        dest.write_bytes(content)
        with open(LOG, "a") as fh:
            fh.write(json.dumps({"repo": URL, "commit": sha, "path": path, "gw": int(gw), "deadline": str(dl),
                                 "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                                 "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
        print(f"GW{gw}: {path.split('/')[-2]} at commit {sha[:8]} {t} ({(dl - t).total_seconds() / 3600:.1f} h before the deadline)")


if __name__ == "__main__":
    main()
