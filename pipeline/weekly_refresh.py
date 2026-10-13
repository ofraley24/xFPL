#!/usr/bin/env python3
"""Weekly refresh before a deadline: new results in, features rebuilt, forecast out. No model is refitted
(each season's model is fitted once, from earlier seasons; analysis/13_xp_model.py).

    python pipeline/weekly_refresh.py              # incremental (default), ~10 min
    python pipeline/weekly_refresh.py --full       # full crosswalk and ratings rebuild (reference check)
    python pipeline/weekly_refresh.py --no-fetch   # reuse the local Understat cache and FPL snapshots

Steps (each timed; log in data/logs/weekly_refresh_<UTC>.log):
  1  FPL snapshot: reuse the latest full snapshot if it is under 12 h old, else take a new one (~5 min)
  2  Understat: fetch the current season (only matches not yet cached), merge, clean, validate.
     A failed check stops the run only if it concerns the EPL (other leagues feed player history only)
  3  FPL tables from the latest full snapshot + validation (any failure stops the run)
  4  match table, crosswalk (--incremental: only fixtures not yet matched), team ratings for the current
     season only (17 --current-only; earlier seasons are rated from their own deadline snapshots)
  5  features: scorer dataset (01), minutes (06), master (12)
  6  forecast (analysis/26_forecast.py): writes nothing unless every gate passes
Incremental steps were checked against the full rebuild on 2026-10-14 (identical files); run --full now and
then to keep that true.
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PY = sys.executable
ENV = {"XFPL_RATINGS": str(DATA / "ratings_eval")}


class Runner:
    def __init__(self):
        (DATA / "logs").mkdir(parents=True, exist_ok=True)
        self.log = DATA / "logs" / f"weekly_refresh_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
        self.times = []

    def run(self, script, *args, ok_codes=(0,)):
        import os
        cmd = [PY, str(ROOT / script), *map(str, args)]
        t0 = time.time()
        with open(self.log, "a") as fh:
            fh.write(f"\n=== {' '.join(cmd[1:])}\n")
            fh.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **ENV}).returncode
        dt = time.time() - t0
        self.times.append((script + (" " + " ".join(map(str, args)) if args else ""), dt, rc))
        print(f"{dt:6.0f}s  rc={rc}  {script} {' '.join(map(str, args))}", flush=True)
        if rc not in ok_codes:
            self.fail(f"{script} exited with {rc}")
        return rc

    def fail(self, why):
        print(f"\nSTOPPED: {why}\nlog: {self.log}")
        sys.exit(1)


def latest_full_snapshot_age_h():
    snaps = sorted(p for p in (DATA / "raw" / "fpl_api").iterdir() if (p / "element_summaries.json.gz").exists())
    if not snaps:
        return None
    t = datetime.strptime(snaps[-1].name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 3600


def current_season_year():
    snaps = sorted(p for p in (DATA / "raw" / "fpl_api").iterdir() if (p / "bootstrap.json").exists())
    url = json.load(open(snaps[-1] / "bootstrap.json"))["game_config"]["settings"]["static_content_url"]
    return re.search(r"/(\d{4})_\d{2}/", url).group(1)


def main():
    full = "--full" in sys.argv
    fetch = "--no-fetch" not in sys.argv
    r = Runner()
    t0 = time.time()
    print(f"weekly refresh ({'full' if full else 'incremental'}); log {r.log}")

    age = latest_full_snapshot_age_h()
    if fetch and (age is None or age > 12):
        r.run("pipeline/06_fetch_fpl_live.py")
    else:
        print(f"       reusing latest full FPL snapshot ({age:.1f} h old)")

    if fetch:
        r.run("pipeline/01_fetch_understat.py", "--seasons", current_season_year())
    r.run("pipeline/02_merge_understat.py")
    r.run("pipeline/02b_clean_understat.py")
    if r.run("pipeline/05_validate_data.py", ok_codes=(0, 1)):
        fails = [ln for ln in (ROOT / "docs" / "DATA_QC.md").read_text().splitlines() if "**FAIL**" in ln]
        epl = [ln for ln in fails if "EPL" in ln]
        for ln in fails:
            print("       Understat QC:", ln.strip("| ")[:110])
        if epl:
            r.fail("Understat QC failed for the EPL")

    r.run("pipeline/08_build_fpl_tables.py")
    r.run("pipeline/09_validate_fpl.py")

    r.run("pipeline/03_build_matches.py", "--source", "merged")
    r.run("pipeline/10_crosswalk_fpl_understat.py", *([] if full else ["--incremental"]))
    r.run("analysis/17_build_asof_ratings.py", "--reuse-selection", "--horizon", 8, *([] if full else ["--current-only"]))

    r.run("analysis/01_build_scorer_assist_dataset.py")
    r.run("analysis/06_build_minutes_dataset.py")
    r.run("analysis/12_build_master.py")

    r.run("analysis/26_forecast.py")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min; log {r.log}")


if __name__ == "__main__":
    main()
