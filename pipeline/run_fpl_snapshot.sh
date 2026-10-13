#!/bin/bash
# Scheduled FPL API snapshot (launchd: ~/Library/LaunchAgents/com.xfpl.fpl_snapshot.plist).
# Morning run: full snapshot (bootstrap, fixtures, every player's element summary, ~5 min).
# Evening run: bootstrap + fixtures only (injury flags / news after press conferences).
set -u
ROOT="/Users/oscarfraley/Documents/Betting model data"
LOG="$ROOT/data/logs"
mkdir -p "$LOG"
cd "$ROOT" || exit 1
if [ "$(date +%H)" -lt 14 ]; then MODE=""; else MODE="--skip-players"; fi
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) start ${MODE:-full}" >> "$LOG/fpl_snapshot.log"
"$ROOT/.venv/bin/python" pipeline/06_fetch_fpl_live.py $MODE >> "$LOG/fpl_snapshot.log" 2>&1
STATUS=$?
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $STATUS" >> "$LOG/fpl_snapshot.log"
exit $STATUS
