#!/usr/bin/env python3
"""Do the COVID adjustments improve TEAM-level predictions out of sample?

All variants use hyperparameters fitted before 2019-08-01. The drift learning rate is chosen on
pre-2019 matches only (walk-forward NLL). Scored on every match from 2019-08-01 (all leagues, and EPL):
Poisson deviance of home and away goals, 1X2 RPS, and the home/away bias inside and outside the
closed-doors period. Paired match-bootstrap intervals for the difference from the base model.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("r11", ROOT / "pipeline" / "11_pi_ratings_to_fpl.py")
R11 = importlib.util.module_from_spec(sp)
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "pipeline"))
sp.loader.exec_module(R11)
SEED = 20261011
CUT = pd.Timestamp("2019-08-01")


def dev(y, mu):
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (t - (y - mu))


def rps_1x2(lam, mu, gh, ga, max_goals=10):
    """Ranked probability score of home/draw/away from independent Poisson goals."""
    xs = np.arange(max_goals + 1)
    out = np.empty(len(lam))
    for i in range(len(lam)):
        mat = np.outer(poisson.pmf(xs, lam[i]), poisson.pmf(xs, mu[i]))
        mat /= mat.sum()
        ph, pd_, pa = np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()
        o = 0 if gh[i] > ga[i] else 1 if gh[i] == ga[i] else 2
        cp, co = np.cumsum([ph, pd_, pa]), np.cumsum(np.eye(3)[o])
        out[i] = ((cp - co) ** 2).sum() / 2
    return out


def main():
    params = json.load(open(ROOT / "data/interim/pre2019/fitted_params_pre2019.json"))
    m = pd.read_csv(ROOT / "data/interim/merged/model_v1/matches_unified.csv", parse_dates=["date"])
    m = m.sort_values(["date", "league", "home_team"], kind="mergesort").reset_index(drop=True)
    pre = m[m.date < CUT].reset_index(drop=True)
    # choose drift_lr on pre-2019 only (Poisson NLL of goals, walk-forward)
    nll = {}
    for lr in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008):
        _, h = R11.replay(pre, params, drift_lr=lr)
        nll[lr] = -(poisson.logpmf(pre.home_goals, h.lam).sum() + poisson.logpmf(pre.away_goals, h.mu).sum())
    best_lr = min(nll, key=nll.get)
    print("pre-2019 NLL by drift_lr:", {k: round(v, 1) for k, v in nll.items()}, "-> chosen", best_lr)
    nll_h = {}
    for lr in (0.0, 0.0002, 0.0005, 0.001, 0.002, 0.004):
        _, h = R11.replay(pre, params, drift_lr=best_lr, ha_lr=lr)
        nll_h[lr] = -(poisson.logpmf(pre.home_goals, h.lam).sum() + poisson.logpmf(pre.away_goals, h.mu).sum())
    best_ha = min(nll_h, key=nll_h.get)
    print("pre-2019 NLL by ha_lr (with chosen drift):", {k: round(v, 1) for k, v in nll_h.items()}, "-> chosen", best_ha)
    variants = {"base (fixed home adv, fixed intercept)": dict(),
                "+ closed-doors home adv (online, n0=50)": dict(bcd=True, bcd_prior_n=50),
                "+ closed-doors home adv (online, n0=20)": dict(bcd=True, bcd_prior_n=20),
                f"+ intercept drift (lr={best_lr})": dict(drift_lr=best_lr),
                f"+ both (n0=50, lr={best_lr})": dict(bcd=True, bcd_prior_n=50, drift_lr=best_lr),
                f"+ all three (ha_lr={best_ha})": dict(bcd=True, bcd_prior_n=50, drift_lr=best_lr, ha_lr=best_ha),
                f"+ all three, ha_lr x4 ({best_ha*4})": dict(bcd=True, bcd_prior_n=50, drift_lr=best_lr, ha_lr=best_ha * 4)}
    res, hist = {}, {}
    for name, kw in variants.items():
        _, h = R11.replay(m, params, **kw)
        h = h.merge(m[["league", "date", "home_team", "away_team", "home_goals", "away_goals"]],
                    on=["league", "date", "home_team", "away_team"])
        hist[name] = h[h.date >= CUT].reset_index(drop=True)
    base = list(variants)[0]
    rng = np.random.default_rng(SEED)
    rows = []
    for name, h in hist.items():
        d = dev(h.home_goals.to_numpy(float), h.lam.to_numpy()) + dev(h.away_goals.to_numpy(float), h.mu.to_numpy())
        r = rps_1x2(h.lam.to_numpy(), h.mu.to_numpy(), h.home_goals.to_numpy(), h.away_goals.to_numpy())
        h["d"], h["r"] = d, r
        hist[name] = h
    for scope, mask_fn in (("all leagues", lambda h: np.ones(len(h), bool)), ("EPL", lambda h: (h.league == "EPL").to_numpy())):
        for name, h in hist.items():
            mk = mask_fn(h)
            b = hist[base]
            diff = (h.d - b.d).to_numpy()[mk]
            boot = rng.integers(0, mk.sum(), size=(2000, mk.sum()))
            bd = diff[boot].mean(1)
            bc = h.bcd.to_numpy()[mk]
            hh = h[mk]
            rows.append({"scope": scope, "model": name, "dev/match": hh.d.mean(), "d_vs_base": diff.mean(),
                         "lo": np.percentile(bd, 2.5), "hi": np.percentile(bd, 97.5), "RPS": hh.r.mean(),
                         "closed-doors home bias (obs-pred)": (hh.home_goals - hh.lam)[bc].mean(),
                         "closed-doors away bias": (hh.away_goals - hh.mu)[bc].mean(),
                         "crowd home bias": (hh.home_goals - hh.lam)[~bc].mean(),
                         "crowd away bias": (hh.away_goals - hh.mu)[~bc].mean(),
                         "season 23-25 total-goals bias": ((hh.home_goals + hh.away_goals) - (hh.lam + hh.mu))[hh.date >= "2023-08-01"].mean()})
    pd.set_option("display.width", 250)
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    t.to_csv(ROOT / "data/analysis/covid_team_eval.csv", index=False)
    h = hist[list(variants)[1]]
    tr = h[h.bcd].groupby(h[h.bcd].date.dt.to_period("M")).home_adv_used.mean()
    print("\nonline closed-doors home advantage (log scale; normal value", round(params["home_adv"], 3), ")\n",
          tr.round(3).to_string())


if __name__ == "__main__":
    main()
