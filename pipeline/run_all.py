#!/usr/bin/env python3
"""Data layer end to end: Understat (fetch, merge, clean, validate) and FPL (fetch, build, validate).

    python pipeline/run_all.py                 # fetch season 2026 and rebuild
    python pipeline/run_all.py --skip-fetch    # reuse the local API cache
    python pipeline/run_all.py --seasons 2025 2026
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args):
    print("\n$", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-fpl-players", action="store_true", help="FPL snapshot without the ~11 min per-player pull")
    ap.add_argument("--seasons", nargs="+", default=["2026"])
    a = ap.parse_args()
    if not a.skip_fetch:
        run(HERE / "01_fetch_understat.py", "--seasons", *a.seasons)
        run(HERE / "06_fetch_fpl_live.py", *(["--skip-players"] if a.skip_fpl_players else []))
        run(HERE / "07_fetch_fpl_history.py")
    run(HERE / "02_merge_understat.py")
    run(HERE / "02b_clean_understat.py")
    run(HERE / "05_validate_data.py")
    run(HERE / "08_build_fpl_tables.py")
    run(HERE / "09_validate_fpl.py")


if __name__ == "__main__":
    main()
