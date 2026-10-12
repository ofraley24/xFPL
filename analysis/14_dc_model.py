#!/usr/bin/env python3
"""Defensive-contribution component (points exist from 2025-26: DEF >= 10 CBIT, MID/FWD >= 12 CBIRT -> +2).

The points did not exist before 2025-26, but the underlying counts (tackles, clearances/blocks/
interceptions, recoveries) are recorded for 2016-17..2018-19. The model is built and validated on those
seasons only (walk-forward: 2016-17 -> 2017-18, 2016-18 -> 2018-19), so the sealed 2025-26 holdout is
not used. 2026-27 (live) rows are read ONLY to check that FPL's own `defensive_contribution` equals the
reconstructed CBIT / CBIRT, i.e. that the 2016-19 counts mean what the rule means.

Model: negative binomial GLM for the count (offset log minutes/90, minutes elasticity), with the player's
lagged count rate (shrunk to the position mean), team/opponent lambda and home; P(count >= T) from the NB
distribution. Parameter uncertainty: coefficient covariance (reported as the sd of P(>=T) over 200 draws).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
SEED = 20261011


def prep(d):
    d = d[(d.minutes > 0) & d.position.isin(["DEF", "MID", "FWD"])].copy()
    d["cbit"] = d.clearances_blocks_interceptions + d.tackles
    d["cbirt"] = d.cbit + d.recoveries
    d["y"] = np.where(d.position == "DEF", d.cbit, d.cbirt)
    d["T"] = np.where(d.position == "DEF", 10, 12)
    d["e"] = d.minutes / 90
    lag_num = np.where(d.position == "DEF", d.L365_f_cbit, d.L365_f_cbirt)
    d["lag_num"], d["lag_min"] = lag_num, d.L365_f_min_dc
    d["log_lamf"], d["log_lama"] = np.log(d.xg_for_pi), np.log(d.xg_against_pi)
    d["home"] = d.was_home.astype(float)
    d["pos_MID"], d["pos_FWD"] = (d.position == "MID").astype(float), (d.position == "FWD").astype(float)
    return d


def design(d, prior):
    pr = d.position.map(prior).to_numpy()
    X = pd.DataFrame({"const": 1.0, "pos_MID": d.pos_MID, "pos_FWD": d.pos_FWD,
                      "log_rate": np.log((d.lag_num + pr * 5) / (d.lag_min / 90 + 5)),
                      "log_lamf": d.log_lamf, "log_lama": d.log_lama, "home": d.home,
                      "log_minutes": np.log(d.e)}, index=d.index)
    return X


def main():
    m = pd.read_parquet(OUT / "master.parquet")
    live = m[m.season == "2026-27"]
    lv = prep(live)
    print("definition check on live 2026-27 rows: FPL defensive_contribution == reconstructed count:",
          f"{(lv.defensive_contribution == lv.y).mean():.4f} of {len(lv):,}")
    d = prep(m[m.season <= "2018-19"])
    rng = np.random.default_rng(SEED)
    rows = []
    for ts in ("2017-18", "2018-19"):
        tr, te = d[d.season < ts], d[d.season == ts]
        prior = (tr.groupby("position").y.sum() / tr.groupby("position").e.sum()).to_dict()
        Xtr, Xte = design(tr, prior), design(te, prior)
        res = sm.NegativeBinomial(tr.y, Xtr, offset=np.log(tr.e)).fit(disp=0, maxiter=200)
        beta, cov = res.params.to_numpy(), res.cov_params().to_numpy()
        alpha = beta[-1]
        mu = np.exp(Xte.to_numpy() @ beta[:-1] + np.log(te.e.to_numpy()))
        n_ = 1 / alpha
        p_hit = nbinom.sf(te["T"].to_numpy() - 1, n_, n_ / (n_ + mu))
        draws = rng.multivariate_normal(beta, cov, size=200)
        P = np.array([nbinom.sf(te["T"].to_numpy() - 1, 1 / b[-1], (1 / b[-1]) / (1 / b[-1] + np.exp(
            Xte.to_numpy() @ b[:-1] + np.log(te.e.to_numpy())))) for b in draws])
        base = te.position.map(tr.assign(h=tr.y >= tr["T"]).groupby("position").h.mean()).to_numpy()
        hit = (te.y >= te["T"]).to_numpy().astype(float)
        bl = lambda p: np.mean((p - hit) ** 2)
        ll = lambda p: -np.mean(hit * np.log(np.clip(p, 1e-9, 1)) + (1 - hit) * np.log(np.clip(1 - p, 1e-9, 1)))
        rows.append({"test season": ts, "n": len(te), "observed rate": hit.mean(), "mean predicted": p_hit.mean(),
                     "Brier model": bl(p_hit), "Brier position base": bl(base), "logloss model": ll(p_hit),
                     "logloss base": ll(base), "median param sd of P": np.median(P.std(0))})
        te = te.assign(p_hit=p_hit, hit=hit)
        q = pd.qcut(te.p_hit, 10, labels=False, duplicates="drop")
        cal = te.groupby(q).agg(pred=("p_hit", "mean"), obs=("hit", "mean"), n=("hit", "size"))
        print(f"\ncalibration, test {ts}\n", cal.round(3).T.to_string())
        coef = pd.Series(beta, index=list(Xtr.columns) + ["alpha"])
    print("\n", pd.DataFrame(rows).round(4).to_string(index=False))
    print("\ncoefficients (last fold)\n", coef.round(3).to_string())
    pd.DataFrame(rows).to_csv(OUT / "dc_model_validation.csv", index=False)


if __name__ == "__main__":
    main()
