#!/usr/bin/env python3
"""Predict the horizon rows (22_build_horizon_rows.py) with the saved fold models (13_xp_model.py).

Each season uses the model fitted for it in the walk-forward evaluation (dev folds 2019-20..2024-25,
sealed 2025-26), unchanged: components trained on earlier seasons, calibration fitted on earlier
out-of-sample h = 1 rows. Rows are simulated one horizon at a time, so each fixture appears once per call
(bonus is ranked within the match and team noise is shared within it).

CHECK: h = 1 is predicted with the fold's own simulation seed and row order, so it must reproduce
xp_predictions_dev / xp_predictions_sealed exactly; the script stops otherwise.
Scoring is NOT done here (docs/HORIZON_PLAN.md: the decision rule is fixed before results are looked at).
Output: data/analysis/xp_horizon.parquet. Options: --seasons a,b (default all in the horizon rows),
--horizons 1,2,... (default 1..8).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
sp = importlib.util.spec_from_file_location("x13", ROOT / "analysis" / "13_xp_model.py")
X13 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(X13)
KEEP = ["season", "round", "asof_round", "asof_deadline", "horizon", "element", "code", "position", "fixture",
        "was_home", "kickoff_time", "minutes", "started", "total_points", "tgt_np_goals", "assists"]
RES = ["xp", "var_total", "var_epi", "q10", "q25", "q50", "q75", "q90", "p_ge2", "p_ge5", "p_ge10", "cdf_lo",
       "cdf_hi", "e_app", "e_goals", "e_assists", "e_cs", "e_gc_pts", "e_saves_pts", "e_bonus", "e_minutes", "e_dc"]


def main():
    args = sys.argv[1:]
    hz = pd.read_parquet(OUT / "horizon_rows.parquet")
    seasons = args[args.index("--seasons") + 1].split(",") if "--seasons" in args else sorted(hz.season.unique())
    horizons = [int(h) for h in args[args.index("--horizons") + 1].split(",")] if "--horizons" in args else list(range(1, 9))
    full = X13.load("2025-26")   # master rows with model columns, in the order the fold models predicted them
    lg_yel = full.drop_duplicates("deadline").set_index("deadline").log_lg_yel
    order = pd.Series(np.arange(len(full)), index=pd.MultiIndex.from_frame(full[["season", "element", "fixture"]]))
    out, bad = [], False
    for s in seasons:
        tag = "sealed" if s >= "2025-26" else "dev"
        M = X13.load_model(tag, s)
        ref = pd.read_parquet(OUT / f"xp_predictions_{tag}.parquet", columns=["season", "element", "fixture"] + RES)
        ref = ref[ref.season == s]
        for h in horizons:
            te = hz[(hz.season == s) & (hz.horizon == h)].copy()
            # master-table order (h = 1 is then exactly the fold's test set)
            te["_o"] = order.reindex(pd.MultiIndex.from_frame(te[["season", "element", "fixture"]])).to_numpy()
            te = X13.prepare(te.sort_values("_o").reset_index(drop=True), lg_yel)
            seed = X13.SEED + 7 * int(s[:4]) + 1000 + (0 if h == 1 else 100 * h)
            res, _, _ = X13.predict_rows(M, te, np.random.default_rng(seed))
            r = pd.concat([te[KEEP], res[RES]], axis=1)
            if h == 1:
                j = r.merge(ref, on=["season", "element", "fixture"], suffixes=("", "_ref"))
                diff = max(float(np.nanmax(np.abs(j[c] - j[c + "_ref"]))) for c in RES)
                ok = len(j) == len(ref) == len(r) and diff == 0.0
                print(f"  {s} h=1 vs {tag} predictions: {len(j):,}/{len(ref):,} rows, max |diff| {diff:.2e} "
                      f"-> {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
                bad |= not ok
            print(f"{s} h={h}: {len(r):,} rows, mean xP {r.xp.mean():.3f}", flush=True)
            out.append(r)
        if bad:
            sys.exit("h = 1 does not reproduce the walk-forward predictions; stopping")
    pd.concat(out, ignore_index=True).to_parquet(OUT / "xp_horizon.parquet", index=False)
    print("saved", sum(len(o) for o in out))


if __name__ == "__main__":
    main()
