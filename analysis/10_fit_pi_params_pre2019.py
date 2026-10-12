#!/usr/bin/env python3
"""Fit the pi-rating hyperparameters on matches BEFORE 2019-08-01 only.

The first player-model test season is 2019-20, so ratings produced with these parameters carry
no look-ahead at all into any test fold (team ratings themselves are always walk-forward).
Same objective, bounds, starting values and optimiser budget as model_v1/fit_v1.py; rho is then
fitted on the training-period walk-forward predictions (vectorised grid, same grid as the original).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model_v1"))
from pi_rating_model import PiRatingDixonColesModel, walk_forward_nll  # noqa: E402

SRC = ROOT / "data" / "interim" / "merged" / "model_v1" / "matches_unified.csv"
OUT = ROOT / "data" / "interim" / "pre2019"
CUTOFF = pd.Timestamp("2019-08-01")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC, parse_dates=["date"])
    df = df.sort_values(["date", "league", "home_team"], kind="mergesort").reset_index(drop=True)
    tr = df[df.date < CUTOFF].reset_index(drop=True)
    leagues = sorted(df.league.unique())
    lm = {lg: float(np.log(tr.loc[tr.league == lg, ["home_goals", "away_goals"]].mean().mean())) for lg in leagues}
    x0 = {"home_adv": 0.25, "scale": 3.0, "alpha": 0.6, "lr_attack": 0.06, "lr_defence": 0.06}
    for lg in leagues:
        x0[f"c_{lg}"] = lm[lg] - 0.15
    order = ["home_adv", "scale", "alpha", "lr_attack", "lr_defence"] + [f"c_{lg}" for lg in leagues]
    bounds = [(-0.5, 1.0) if p == "home_adv" else (0.5, 20.0) if p == "scale" else (0.0, 1.0) if p == "alpha"
              else (0.001, 0.5) if p.startswith("lr_") else (-2.0, 2.0) for p in order]
    t0 = time.time()
    res = minimize(lambda v: walk_forward_nll(tr, dict(zip(order, v)), leagues)[0], np.array([x0[p] for p in order]),
                   method="Nelder-Mead", bounds=bounds,
                   options={"maxfev": 250, "xatol": 1e-3, "fatol": 5.0, "adaptive": True})
    params = dict(zip(order, map(float, res.x)))
    print(f"fit done in {time.time()-t0:.0f}s, nfev={res.nfev}, NLL={res.fun:.1f}", flush=True)
    # rho on training-period walk-forward lambdas (pre-update), vectorised
    m = PiRatingDixonColesModel(home_adv=params["home_adv"], scale=params["scale"], alpha=params["alpha"],
                                lr_attack=params["lr_attack"], lr_defence=params["lr_defence"],
                                league_intercepts={lg: params[f"c_{lg}"] for lg in leagues})
    lam, mu = np.empty(len(tr)), np.empty(len(tr))
    for i, r in enumerate(tr.itertuples(index=False)):
        lam[i], mu[i] = m.update(r.league, r.home_team, r.away_team, r.home_goals, r.away_goals,
                                 r.home_npxG, r.away_npxG)
    gh, ga = tr.home_goals.to_numpy(), tr.away_goals.to_numpy()
    best, best_ll = None, -np.inf
    for rho in np.linspace(-0.30, 0.10, 81):
        t = np.ones(len(tr))
        t[(gh == 0) & (ga == 0)] = (1 - lam * mu * rho)[(gh == 0) & (ga == 0)]
        t[(gh == 0) & (ga == 1)] = (1 + lam * rho)[(gh == 0) & (ga == 1)]
        t[(gh == 1) & (ga == 0)] = (1 + mu * rho)[(gh == 1) & (ga == 0)]
        t[(gh == 1) & (ga == 1)] = 1 - rho
        if (t <= 0).any():
            continue
        ll = np.log(t).sum()
        if ll > best_ll:
            best, best_ll = rho, ll
    params["rho"] = float(best)
    params["_meta"] = {"train_cutoff": str(CUTOFF.date()), "n_train_matches": int(len(tr)),
                       "optimiser": "Nelder-Mead maxfev=250 (as fit_v1.py)", "nll": float(res.fun)}
    json.dump(params, open(OUT / "fitted_params_pre2019.json", "w"), indent=2)
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
