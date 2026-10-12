#!/usr/bin/env python3
"""Ablation ladder of Poisson models for FPL goals and assists, walk-forward by season.

Model: y ~ Poisson(e * lambda), e = minutes/90, log(lambda) = linear in features
(sklearn PoissonRegressor; fitted as rate y/e with weight e, which is the same likelihood).
HistGradientBoosting (Poisson loss) is the non-linear ceiling check.

Protocol (PROTOCOL.md): development uses seasons 2016-17..2024-25 only. 2025-26 is the sealed
holdout and 2026-27 is live; neither is read here. Test folds are 2019-20..2024-25; each fold
trains on every earlier season. All shrinkage priors are computed from TRAINING rows only.

Each rung adds ONE idea to the previous, so the paired difference isolates its value:
  G0 position mean | G1 +home,+team lambda | G2/G2a/G3 player npxG rate (540d / 180d / both)
  G4 +shot volume | G5 +actual goals (finishing) | G6 +penalty role, xGChain, xGBuildup
  G7 +opponent lambda | G8s team-relative SHARE instead of absolute rates (vs G3)
  G9 +price, ownership | G10 gradient boosting on everything.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
POS = ["GKP", "DEF", "MID", "FWD"]
RATE_STATS = ["npxg", "np_shots", "xa", "xgchain", "xgbuildup", "np_goals", "pen_att", "assists"]
M0 = 900.0          # pseudo-minutes of position-average prior (10 full matches)
SEED = 20261011
TEST_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def priors_from(train):
    pr = {}
    for hl in (180, 540):
        for st in RATE_STATS:
            for p in POS:
                g = train[train.position == p]
                den = g[f"L{hl}_time"].sum()
                pr[(hl, st, p)] = 90 * g[f"L{hl}_{st}"].sum() / den if den > 0 else 0.0
        for p in POS:
            g = train[train.position == p]
            for st in ("npxg", "xa"):
                d = g[f"L{hl}_team_np_on"].sum()
                pr[(hl, f"share_{st}", p)] = g[f"L{hl}_{st}"].sum() / d if d > 0 else 0.0
    return pr


def features(df, pr, m0=M0):
    F = pd.DataFrame(index=df.index)
    for p in POS:
        F[f"pos_{p}"] = (df.position == p).astype(float)
    F["home"] = df.was_home.astype(float)
    F["log_lamf"] = np.log(df.xg_for_pi)
    F["log_lama"] = np.log(df.xg_against_pi)
    F["log_price"] = np.log(df.value.clip(lower=35))
    F["log_sel"] = np.log1p(df.selected)
    F["log_cs"] = np.log(df.p_clean_sheet)
    d0 = m0 / 90 * 1.2
    for hl in (180, 540):
        t = df[f"L{hl}_time"]
        for st in RATE_STATS:
            prior = df.position.map({p: pr[(hl, st, p)] for p in POS}).to_numpy()
            r = (df[f"L{hl}_{st}"] + prior * m0 / 90) / ((t + m0) / 90)
            F[f"log_{st}90_{hl}"] = np.log(np.clip(r, 1e-4, None))
        for st in ("npxg", "xa"):
            prior = df.position.map({p: pr[(hl, f"share_{st}", p)] for p in POS}).to_numpy()
            s = (df[f"L{hl}_{st}"] + prior * d0) / (df[f"L{hl}_team_np_on"] + d0)
            F[f"log_share_{st}_{hl}"] = np.log(np.clip(s, 1e-4, None))
        F[f"log_exposure_{hl}"] = np.log1p(t / 90)
    F["log_apps"] = np.log1p(df.n_prev_apps)
    F["gap"] = np.log1p(df.days_since_last.fillna(3650))
    return F


POSC = [f"pos_{p}" for p in POS]
G = {}
G["G0 position mean"] = POSC
G["G1 +home +team lambda"] = G["G0 position mean"] + ["home", "log_lamf"]
G["G2 +npxG/90 (540d)"] = G["G1 +home +team lambda"] + ["log_npxg90_540"]
G["G2a npxG/90 (180d) instead"] = G["G1 +home +team lambda"] + ["log_npxg90_180"]
G["G3 +npxG/90 both HL"] = G["G1 +home +team lambda"] + ["log_npxg90_540", "log_npxg90_180"]
G["G4 +shot volume"] = G["G3 +npxG/90 both HL"] + ["log_np_shots90_540", "log_np_shots90_180"]
G["G5 +actual goals/90"] = G["G4 +shot volume"] + ["log_np_goals90_540", "log_np_goals90_180"]
G["G6 +penalty, xGChain, xGBuildup"] = G["G5 +actual goals/90"] + ["log_pen_att90_540", "log_xgchain90_540",
                                                                    "log_xgbuildup90_540"]
G["G7 +opponent lambda"] = G["G6 +penalty, xGChain, xGBuildup"] + ["log_lama"]
G["G8s share instead of rates"] = G["G1 +home +team lambda"] + ["log_share_npxg_540", "log_share_npxg_180"]
G["G9 +price, ownership"] = G["G7 +opponent lambda"] + ["log_price", "log_sel"]
G["G11 G9 + share"] = G["G9 +price, ownership"] + ["log_share_npxg_540", "log_share_npxg_180"]
A = {}
A["A0 position mean"] = POSC
A["A1 +home +team lambda"] = A["A0 position mean"] + ["home", "log_lamf"]
A["A2 +xA/90 (540d)"] = A["A1 +home +team lambda"] + ["log_xa90_540"]
A["A3 +xA/90 both HL"] = A["A1 +home +team lambda"] + ["log_xa90_540", "log_xa90_180"]
A["A4 +actual assists/90"] = A["A3 +xA/90 both HL"] + ["log_assists90_540", "log_assists90_180"]
A["A5 +xGChain, xGBuildup"] = A["A4 +actual assists/90"] + ["log_xgchain90_540", "log_xgbuildup90_540"]
A["A6 +opponent lambda"] = A["A5 +xGChain, xGBuildup"] + ["log_lama"]
A["A7s share instead of rates"] = A["A1 +home +team lambda"] + ["log_share_xa_540", "log_share_xa_180"]
A["A8 +price, ownership"] = A["A6 +opponent lambda"] + ["log_price", "log_sel"]
A["A10 A8 + share"] = A["A8 +price, ownership"] + ["log_share_xa_540", "log_share_xa_180"]


def fit_predict(kind, cols, Ftr, ytr, etr, Fte):
    if kind == "glm":
        sc = StandardScaler().fit(Ftr[cols])
        m = PoissonRegressor(alpha=1e-4, max_iter=1000)
        m.fit(sc.transform(Ftr[cols]), ytr / etr, sample_weight=etr)
        return m.predict(sc.transform(Fte[cols]))
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.04, max_depth=4,
                                      min_samples_leaf=200, l2_regularization=2.0, random_state=SEED)
    m.fit(Ftr[cols], ytr / etr, sample_weight=etr)
    return m.predict(Fte[cols])


def deviance(y, mu):
    mu = np.clip(mu, 1e-6, None)  # a stratified fit can predict exactly 0 (e.g. goalkeepers)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def run(target, ladder, df):
    dev = df[df.season.isin(TEST_SEASONS) | (df.season < TEST_SEASONS[0])].copy()
    preds = {}
    for ts in TEST_SEASONS:
        tr, te = dev[dev.season < ts], dev[dev.season == ts]
        pr = priors_from(tr)
        Ftr, Fte = features(tr, pr), features(te, pr)
        etr = (tr.minutes / 90).to_numpy()
        for name, cols in ladder.items():
            preds.setdefault(name, []).append(pd.Series(
                fit_predict("glm", cols, Ftr, tr[target].to_numpy(float), etr, Fte), index=te.index))
        allcols = [c for c in Ftr.columns]
        preds.setdefault("GBM all features", []).append(pd.Series(
            fit_predict("gbm", allcols, Ftr, tr[target].to_numpy(float), etr, Fte), index=te.index))
        print(f"  {target}: fold {ts} done", flush=True)
    test = dev[dev.season.isin(TEST_SEASONS)].copy()
    for k in preds:
        test[f"mu::{k}"] = pd.concat(preds[k]).reindex(test.index) * (test.minutes / 90)
    return test


def summarise(target, test, names):
    y = test[target].to_numpy(float)
    e = (test.minutes / 90).to_numpy()
    fx = test.season + "_" + test.fixture.astype(str)
    codes, uniq = pd.factorize(fx)
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, len(uniq), size=(2000, len(uniq)))
    per_fx = {}
    for n in names:
        d = deviance(y, test[f"mu::{n}"].to_numpy())
        per_fx[n] = np.bincount(codes, weights=d, minlength=len(uniq))
    exp_fx = np.bincount(codes, weights=e, minlength=len(uniq))
    base = names[0]
    rows = []
    for i, n in enumerate(names):
        mu = test[f"mu::{n}"].to_numpy()
        D = per_fx[n].sum() / e.sum()
        def ci(a, b):  # mean-deviance-per-90 difference a-b with fixture bootstrap
            diff = per_fx[a] - per_fx[b]
            samp = diff[boot].sum(1) / exp_fx[boot].sum(1)
            return diff.sum() / e.sum(), np.percentile(samp, 2.5), np.percentile(samp, 97.5)
        d0 = ci(n, base)
        dp = ci(n, names[i - 1]) if i > 0 else (np.nan, np.nan, np.nan)
        p1 = 1 - np.exp(-mu)
        yb = (y > 0).astype(float)
        ll = -np.mean(yb * np.log(np.clip(p1, 1e-9, 1)) + (1 - yb) * np.log(np.clip(1 - p1, 1e-9, 1)))
        top = np.argsort(-mu)[: len(mu) // 10]
        rows.append({"model": n, "dev_per90": D, "vs_G0_%": 100 * d0[0] / (per_fx[base].sum() / e.sum()),
                     "vs_prev": dp[0], "prev_lo": dp[1], "prev_hi": dp[2],
                     "logloss_P(y>=1)": ll, "top10%_share": y[top].sum() / y.sum(),
                     "sum_pred/sum_obs": mu.sum() / y.sum()})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(OUT / "scorer_assist_dataset.parquet")
    df = df[df.season <= "2024-25"].reset_index(drop=True)   # sealed + live seasons excluded
    assert df.season.max() == "2024-25"
    res = {}
    for target, ladder in (("goals_scored", G), ("assists", A)):
        print(f"== {target}", flush=True)
        test = run(target, ladder, df)
        names = list(ladder.keys()) + ["GBM all features"]
        res[target] = (test, summarise(target, test, names))
        res[target][1].to_csv(OUT / f"ladder_{target}.csv", index=False)
        test.to_parquet(OUT / f"ladder_{target}_predictions.parquet", index=False)
        pd.set_option("display.width", 250)
        print(res[target][1].round(4).to_string(index=False), flush=True)
        # per-season deviance for the key rungs
        y = test[target].to_numpy(float); e = test.minutes / 90
        rows = []
        for n in names:
            d = pd.Series(deviance(y, test[f"mu::{n}"].to_numpy()), index=test.index)
            rows.append(d.groupby(test.season).sum() / e.groupby(test.season).sum())
        print(pd.DataFrame(rows, index=names).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
