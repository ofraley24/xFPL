#!/usr/bin/env python3
"""Round 3: combine the ideas that helped in round 2 into one structural model and compare it
with the best free-coefficient models.

  C0  STRICT two-factor fraction:  lambda_i = lambda_team * (share of team shots) * (npxG per shot)
      x exp(position), coefficients fixed at 1, weaker shrinkage (x0.25)
  C1  C0 + minutes elasticity (free)
  C2  C1 + price and ownership (free)
  C3  C2 + home and opponent lambda (free)
  C4  free-coefficient version of C3 (fraction terms get fitted weights)
  R1  reference: B6 from round 2 (absolute rates + roles + elasticity + price + 1080d + fraction)
  R2  reference: G9-style (absolute rates + opponent lambda + price/ownership)
Assists use the xA fraction (no shot-volume analogue is available in Understat).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("r2", ROOT / "analysis" / "03_fractions_and_engineering.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
OUT = M.OUT


def specs(target):
    g = target == "goals_scored"
    base = ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]
    fac = ["log_lamf", "log_fr_shots_540", "log_quality_540"] if g else ["log_lamf", "log_fr_xa_540"]
    d = 0.25
    S = {}
    S["C0 strict fraction"] = dict(cols=base, offset=fac, d0mult=d)
    S["C1 +minutes elasticity"] = dict(cols=base + ["log_minutes"], offset=fac, d0mult=d)
    S["C2 +price, ownership"] = dict(cols=base + ["log_minutes", "log_price", "log_sel"], offset=fac, d0mult=d)
    S["C3 +home, opponent lambda"] = dict(cols=base + ["log_minutes", "log_price", "log_sel", "home", "log_lama"],
                                         offset=fac, d0mult=d)
    S["C4 free coefficients (C3)"] = dict(cols=base + fac + ["log_minutes", "log_price", "log_sel", "home", "log_lama"],
                                         d0mult=d)
    ab = "npxg" if g else "xa"
    ref = ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD", "home", "log_lamf", f"log_{ab}90_540", f"log_{ab}90_180",
           "log_lama", "log_price", "log_sel"]
    S["R2 absolute rates + price (G9-like)"] = dict(cols=ref)
    S["R1 B6 combined (round 2)"] = M.specs(target)["B6 combined (all of the above)"]
    return S


def elasticity(df, target):
    """Coefficient on log(minutes/90) in an UNSCALED Poisson fit, with rate features + team lambda."""
    pr = M.prior_table(df)
    F = M.build(df, pr)
    ab = "npxg" if target == "goals_scored" else "xa"
    cols = ["pos_DEF", "pos_MID", "pos_FWD", "home", "log_lamf", f"log_{ab}90_540", "log_minutes"]
    # y ~ Poisson(e*lam*(m/90)^b): with offset log e, coefficient on log_minutes is (b); fitted as rate
    m = PoissonRegressor(alpha=0, max_iter=2000).fit(F[cols], df[target] / (df.minutes / 90),
                                                      sample_weight=df.minutes / 90)
    return dict(zip(cols, m.coef_))["log_minutes"]


def main():
    df = pd.read_parquet(OUT / "scorer_assist_dataset.parquet")
    df = df[df.season <= "2024-25"].reset_index(drop=True)
    pd.set_option("display.width", 250)
    for target in ("goals_scored", "assists"):
        print(f"\n== {target}", flush=True)
        test, names = M.run_target(target, df, specs_fn=specs, recal=False)
        ref = "R2 absolute rates + price (G9-like)"
        t = M.compare(target, test, names, ref)
        print(t.round(4).to_string(index=False))
        t.to_csv(OUT / f"round3_{target}.csv", index=False)
        print(f"minutes elasticity b (rate ~ minutes^b), pooled dev set: {elasticity(df, target):+.3f}")
        # calibration of the best structural model by predicted-rate decile
        n = "C3 +home, opponent lambda"
        y = test[target].to_numpy(float)
        mu = test[f"mu::{n}"].to_numpy()
        q = pd.qcut(mu, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"pred": mu, "obs": y}).groupby(q).sum()
        cal["ratio"] = cal.pred / cal.obs
        print(f"\ncalibration by predicted decile ({n})\n", cal.round(2).T.to_string())
        by_pos = pd.DataFrame({"pred": mu, "obs": y}).groupby(test.position.to_numpy()).sum()
        by_pos["ratio"] = by_pos.pred / by_pos.obs
        print("\nby position\n", by_pos.round(2).to_string())


if __name__ == "__main__":
    main()
