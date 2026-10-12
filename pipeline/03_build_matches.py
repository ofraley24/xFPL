#!/usr/bin/env python3
"""Build matches_unified.csv from a chosen Understat source.

--source legacy   the frozen 2014..2026-09-01 snapshot in understat/
                  (used to verify that the rebuilt pipeline reproduces the
                  numbers documented in model_v1/STAGE4_RESULTS.md)
--source merged   snapshot + freshly fetched seasons (data/understat_merged/)

Reuses model_v1/01_build_dataset.build_league unchanged so that the logic under
test is exactly the logic that produced the original results.
"""
import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import INTERIM, LEAGUES, LEGACY_UNDERSTAT, MERGED, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "model_v1"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["legacy", "merged"], required=True)
    args = ap.parse_args()
    mod = importlib.import_module("01_build_dataset")
    mod.DATA_DIR = LEGACY_UNDERSTAT if args.source == "legacy" else MERGED
    out_dir = INTERIM / args.source / "model_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [mod.build_league(lg) for lg in LEAGUES]
    # original used an unstable sort on date alone; ties (simultaneous kick-offs,
    # always different teams) were ordered arbitrarily. Break ties explicitly.
    full = (pd.concat(frames, ignore_index=True)
            .sort_values(["date", "league", "home_team"], kind="mergesort")
            .reset_index(drop=True))
    full.to_csv(out_dir / "matches_unified.csv", index=False)
    print(f"[{args.source}] {len(full):,} played matches, "
          f"{full.date.min().date()} .. {full.date.max().date()} -> {out_dir}")


if __name__ == "__main__":
    main()
