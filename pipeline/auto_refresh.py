#!/usr/bin/env python3
"""Run the weekly refresh as soon as the sources have the finished gameweek (launchd, every 2 hours).

Each check reads FPL's gameweek list (one request, nothing saved). The refresh runs when
  * the gameweek before the next deadline is finished AND data_checked by FPL (bonus and stats final),
  * no successful refresh has been recorded for that deadline yet, and
  * the deadline is at least an hour away (decision time is the deadline; the forecast gates also refuse it).
It then runs pipeline/weekly_refresh.py --fresh-snapshot. If a source lags (e.g. Understat has not posted the
last match), the forecast gates fail, nothing is published, and the next check tries again. On success the
deadline is recorded in data/logs/auto_refresh_state.json, the newest forecast is copied into the website
(../xfpl-web, if present) and deployed to Cloudflare Pages (`npm run deploy`; the site is public; a failed deploy
is logged and does not fail the refresh), and a macOS notification is shown.
Options: --dry-run (report the decision only), --force (run even if already done for this deadline).
"""
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "data" / "logs"
STATE = LOGS / "auto_refresh_state.json"
WEB = ROOT.parent / "xfpl-web"
NODE = "/usr/local/bin/node"


def log(msg):
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}"
    print(line, flush=True)
    with open(LOGS / "auto_refresh.log", "a") as fh:
        fh.write(line + "\n")


def notify(title, text):
    subprocess.run(["osascript", "-e", f'display notification "{text}" with title "{title}"'], check=False)


def main():
    dry, force = "--dry-run" in sys.argv, "--force" in sys.argv
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = open(LOGS / "auto_refresh.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another refresh is running; skipping")
        return
    boot = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/",
                        headers={"User-Agent": "Mozilla/5.0 (research; xFPL project)"}, timeout=30).json()
    now = datetime.now(timezone.utc)
    events = sorted(boot["events"], key=lambda e: e["id"])
    upcoming = [e for e in events if datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")) > now]
    if not upcoming:
        log("no upcoming deadline (season over)")
        return
    nxt = upcoming[0]
    deadline = datetime.fromisoformat(nxt["deadline_time"].replace("Z", "+00:00"))
    prev = next((e for e in events if e["id"] == nxt["id"] - 1), None)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    done = state.get("done_deadline") == nxt["deadline_time"]
    ready = prev is None or (prev["finished"] and prev["data_checked"])
    time_ok = deadline - now > timedelta(hours=1)
    why = (f"next GW{nxt['id']} deadline {nxt['deadline_time']}; previous GW finished={prev and prev['finished']} "
           f"data_checked={prev and prev['data_checked']}; already done={done}")
    if not ready or not time_ok or (done and not force):
        log(f"waiting: {why}")
        return
    log(f"RUN: {why}")
    if dry:
        return
    rc = subprocess.run([sys.executable, str(ROOT / "pipeline" / "weekly_refresh.py"), "--fresh-snapshot"], cwd=ROOT).returncode
    state.update({"last_attempt_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "last_rc": rc})
    if rc == 0:
        state["done_deadline"] = nxt["deadline_time"]
        if WEB.exists():
            subprocess.run([NODE, "scripts/sync-data.mjs"], cwd=WEB, check=False)
            # launchd's PATH lacks /usr/local/bin, where node, npm and npx live
            env = {**os.environ, "PATH": f"{Path(NODE).parent}:{os.environ.get('PATH', '/usr/bin:/bin')}"}
            with open(LOGS / "deploy.log", "a") as fh:
                deploy_rc = subprocess.run(["npm", "run", "deploy"], cwd=WEB, env=env, stdout=fh,
                                           stderr=subprocess.STDOUT).returncode
            log(f"deploy rc={deploy_rc} (log data/logs/deploy.log)")
        log(f"done: forecast for GW{nxt['id']} written")
        notify("xFPL", f"GW{nxt['id']} forecast is ready" + ("" if not WEB.exists() or deploy_rc == 0
                                                              else "; site deploy failed, see deploy.log"))
    else:
        log(f"refresh failed (rc={rc}); will retry at the next check")
        if state.get("notified_fail") != nxt["deadline_time"]:
            notify("xFPL", f"GW{nxt['id']} refresh not ready yet (rc={rc}); retrying every 2 h")
            state["notified_fail"] = nxt["deadline_time"]
    STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
