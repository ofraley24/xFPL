#!/usr/bin/env python3
"""Evaluation ratings with COVID / drift adjustments whose settings are chosen AS OF each season.

Hyperparameters: fitted before 2019-08-01 (10_fit_pi_params_pre2019.py). Closed-doors home advantage:
online (n0 = 50), always on. The two drift rates (league scoring level, crowd home advantage) are chosen
for each FPL season S by walk-forward Poisson NLL on the two seasons BEFORE S (replay over all data
before S). Seasons up to 2018-19 use the pre-2019 choice. 2025-26 and 2026-27 use the choice made
before 2025-26. Each season's FPL fixture ratings are then taken from a replay with that season's
settings, so every rating is one that could have been computed at the time.
Output: data/ratings_eval/fpl_team_fixture_ratings.csv (+ selection table in docs/).
Options: --reuse-selection (take the drift settings from docs/ratings_asof_selection.csv instead of re-running
the search), --horizon N (also fpl_team_fixture_ratings_asof.csv: played fixtures rated as of the deadlines
1..N gameweeks earlier, for the horizon backtest), --current-only (weekly refresh: replay only the current
season's setting and replace that season's rows in the existing files; earlier seasons are rated from
their own deadline snapshots and cannot change).
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sp = importlib.util.spec_from_file_location("r11", ROOT / "pipeline" / "11_pi_ratings_to_fpl.py")
R11 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(R11)
PARAMS = ROOT / "data/interim/pre2019/fitted_params_pre2019.json"
DRIFT = (0.001, 0.002, 0.004)
HA = (0.0, 0.0005, 0.001, 0.002, 0.004)


def main():
    params = json.load(open(PARAMS))
    horizon = int(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else 0
    if "--reuse-selection" in sys.argv:
        prev = pd.read_csv(ROOT / "docs" / "ratings_asof_selection.csv")
        sel = {int(r.season[:4]): (r.drift_lr, r.ha_lr) for r in prev.itertuples()}
    else:
        sel = select_drift(params)
    build(sel, horizon, current_only="--current-only" in sys.argv)


def select_drift(params):
    m = R11.load_matches_with_clock()   # FPL kickoff clock for EPL; same ordering as the ratings script
    sel = {}
    for y in range(2019, 2026):
        start = pd.Timestamp(f"{y}-08-01")
        hist_d = m[m.t < start].reset_index(drop=True)
        score = hist_d.t >= pd.Timestamp(f"{y-2}-08-01")
        best = None
        for dl in DRIFT:
            for ha in HA:
                _, h = R11.replay(hist_d, params, bcd=True, bcd_prior_n=50, drift_lr=dl, ha_lr=ha)
                nll = -(poisson.logpmf(hist_d.home_goals[score], h.lam[score]).sum()
                        + poisson.logpmf(hist_d.away_goals[score], h.mu[score]).sum())
                if best is None or nll < best[0]:
                    best = (nll, dl, ha)
        sel[y] = (best[1], best[2])
        print(f"season {y}-{str(y+1)[2:]}: chosen drift_lr={best[1]} ha_lr={best[2]} (NLL on {score.sum()} matches)", flush=True)
    sel_tab = pd.DataFrame([{"season": f"{y}-{str(y+1)[2:]}", "drift_lr": v[0], "ha_lr": v[1]} for y, v in sel.items()])
    sel_tab.to_csv(ROOT / "docs" / "ratings_asof_selection.csv", index=False)
    return sel


def build(sel, horizon=0, current_only=False):
    # seasons <= 2018-19: pre-2019 choice (drift chosen pre-2019 in 16_covid_team_eval.py = 0.002, ha = 0)
    key = lambda s: (0.002, 0.0) if int(s[:4]) < 2019 else sel[min(int(s[:4]), 2025)]
    dst = ROOT / "data" / "ratings_eval"
    cur_season = pd.read_csv(ROOT / "data" / "fpl" / "fpl_events.csv").season.iloc[0]
    combos = {key(cur_season)} if current_only else {(0.002, 0.0)} | set(sel.values())
    outs = {}
    for dl, ha in combos:
        tag = f"d{dl}_h{ha}"
        out = ROOT / "data" / "ratings_tmp" / tag
        subprocess.run([sys.executable, str(ROOT / "pipeline/11_pi_ratings_to_fpl.py"), "--params", str(PARAMS),
                        "--label", f"pre2019+covid({tag})", "--out", str(out), "--bcd", "--drift-lr", str(dl),
                        "--ha-lr", str(ha)] + (["--horizon", str(horizon)] if horizon else []),
                       check=True, capture_output=True)
        outs[(dl, ha)] = pd.read_csv(out / "fpl_team_fixture_ratings.csv")
        if horizon:
            outs[(dl, ha, "asof")] = pd.read_csv(out / "fpl_team_fixture_ratings_asof.csv")
    dst.mkdir(exist_ok=True)
    if current_only:
        k = key(cur_season)
        for name, o in (("fpl_team_fixture_ratings.csv", outs[k]),
                        ("fpl_team_fixture_ratings_asof.csv", outs.get(k + ("asof",)))):
            if o is None:
                continue
            old = pd.read_csv(dst / name)
            pd.concat([old[old.season != cur_season], o[o.season == cur_season]], ignore_index=True).to_csv(
                dst / name, index=False)
        print(f"current season {cur_season} replaced with setting d{k[0]}_h{k[1]}")
        return
    seasons = sorted(outs[(0.002, 0.0)].season.unique())
    allr = pd.concat([outs[key(s)][outs[key(s)].season == s] for s in seasons], ignore_index=True)
    allr.to_csv(dst / "fpl_team_fixture_ratings.csv", index=False)
    if horizon:
        t_ = lambda s: outs[key(s) + ("asof",)]
        pd.concat([t_(s)[t_(s).season == s] for s in seasons], ignore_index=True).to_csv(
            dst / "fpl_team_fixture_ratings_asof.csv", index=False)
    print(allr.groupby("season").params_label.first().to_string())


if __name__ == "__main__":
    main()
