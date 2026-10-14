#!/usr/bin/env python3
"""Regression check: the xP outputs of the current code must equal a frozen baseline.

Used when restructuring analysis/13_xp_model.py (fit / predict split) without changing the model: every
prediction column and every component coefficient must match the baseline written by the earlier code under
the same seeds. Usage: 21_regression_check.py [tag ...] (default: dev sealed live);
baseline folder: data/analysis/baseline_pre_split (override with --baseline <dir>).
Exit status 1 if any tag differs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"


def compare(a, b, what, atol=0.0):
    """Exact (atol=0) or near-exact comparison of two frames with the same rows; returns a list of problems."""
    probs = []
    if len(a) != len(b):
        return [f"{what}: {len(a)} rows vs {len(b)} in the baseline"]
    if set(a.columns) != set(b.columns):
        probs.append(f"{what}: columns differ: new {sorted(set(a.columns) - set(b.columns))}, "
                     f"missing {sorted(set(b.columns) - set(a.columns))}")
    for c in sorted(set(a.columns) & set(b.columns) - {"row_id"}):   # row_id: a running number that shifts when rows are added
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            x, y = x.astype(float), y.astype(float)
            same = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= atol)
        else:
            same = (pd.isna(x) & pd.isna(y)) | (x == y)
        if not same.all():
            i = np.flatnonzero(~same)[0]
            probs.append(f"{what}.{c}: {(~same).sum()} rows differ (first: {x[i]!r} vs {y[i]!r})")
    return probs


def main():
    args = sys.argv[1:]
    base = OUT / "baseline_pre_split"
    if "--baseline" in args:
        i = args.index("--baseline")
        base, args = Path(args[i + 1]), args[:i] + args[i + 2:]
    tags = args or ["dev", "sealed", "live"]
    bad = False
    for t in tags:
        probs = compare(pd.read_parquet(OUT / f"xp_predictions_{t}.parquet"),
                        pd.read_parquet(base / f"xp_predictions_{t}.parquet"), f"predictions[{t}]")
        probs += compare(pd.read_csv(OUT / f"xp_component_coefficients_{t}.csv"),
                         pd.read_csv(base / f"xp_component_coefficients_{t}.csv"), f"coefficients[{t}]")
        print(f"{t}: {'IDENTICAL' if not probs else 'DIFFERENT'}")
        for p in probs[:20]:
            print("   ", p)
        bad |= bool(probs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
