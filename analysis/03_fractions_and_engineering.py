#!/usr/bin/env python3
"""Round 2: (A) does separating team xG from the player's xG FRACTION help, and (B) other engineering.

Same protocol as 02 (walk-forward, test 2019-20..2024-25, development seasons only, priors from
training rows only, Poisson with minutes exposure).

BLOCK A  the fraction idea.  The player's fraction f = decayed player npxG / decayed team npxG
  while he was on the pitch (team xG scaled by his minutes share), shrunk to the position mean.
  A1  STRICT structural:  lambda_i = lambda_team * f_i, coefficients FIXED at 1 (position intercepts only)
  A2  free coefficients on log lambda_team and log f
  A3  A2 + the player's absolute rate (does the absolute level add to the fraction?)
  A4  two-factor fraction: share of team SHOTS x shot quality (npxG per shot), both shrunk
  A5  shrinkage strength of f varied (pseudo-count x0.25 .. x16)
  A0  reference: absolute rate model (team lambda + player npxG/90, both half-lives)
BLOCK B  engineering on a common base (absolute rates + opponent lambda): role shares, minutes
  elasticity, 1080-day half-life, per-position fits, prior strength, rolling recalibration.
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
spec = importlib.util.spec_from_file_location("ladder", ROOT / "analysis" / "02_fit_poisson_ladder.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
POS, SEED, TEST = L.POS, L.SEED, L.TEST_SEASONS
HLS = (180, 540, 1080)
RS = ["npxg", "np_shots", "xa", "xgchain", "xgbuildup", "np_goals", "pen_att", "assists"]
ROLES = ["r_gk", "r_def", "r_dm", "r_mid", "r_am", "r_fw", "r_sub"]


def prior_table(tr):
    pr = {}
    for hl in HLS:
        for p in POS:
            g = tr[tr.position == p]
            t = g[f"L{hl}_time"].sum()
            for st in RS:
                pr[(hl, st, p)] = 90 * g[f"L{hl}_{st}"].sum() / t if t > 0 else 0.0
            for st in ("npxg", "xa"):
                d = g[f"L{hl}_team_np_on"].sum()
                pr[(hl, f"fr_{st}", p)] = g[f"L{hl}_{st}"].sum() / d if d > 0 else 0.0
            d = g[f"L{hl}_team_shots_on"].sum()
            pr[(hl, "fr_shots", p)] = g[f"L{hl}_np_shots"].sum() / d if d > 0 else 0.0
            s = g[f"L{hl}_np_shots"].sum()
            pr[(hl, "quality", p)] = g[f"L{hl}_npxg"].sum() / s if s > 0 else 0.1
    return pr


def build(df, pr, m0=900.0, d0mult=1.0, kq=10.0):
    F = pd.DataFrame(index=df.index)
    for p in POS:
        F[f"pos_{p}"] = (df.position == p).astype(float)
    F["home"] = df.was_home.astype(float)
    F["log_lamf"] = np.log(df.xg_for_pi)
    F["log_lama"] = np.log(df.xg_against_pi)
    F["log_price"] = np.log(df.value.clip(lower=35))
    F["log_sel"] = np.log1p(df.selected)
    F["log_minutes"] = np.log(df.minutes / 90)
    d0 = m0 / 90 * 1.2 * d0mult       # pseudo xG-while-on-pitch for the fraction (~team xG/90 = 1.2)
    ds = m0 / 90 * 11.0 * d0mult      # pseudo team-shots-while-on-pitch (~11 non-pen shots/90)
    pm = lambda key, hl: df.position.map({p: pr[(hl, key, p)] for p in POS}).to_numpy()
    for hl in HLS:
        t = df[f"L{hl}_time"]
        for st in RS:
            r = (df[f"L{hl}_{st}"] + pm(st, hl) * m0 / 90) / ((t + m0) / 90)
            F[f"log_{st}90_{hl}"] = np.log(np.clip(r, 1e-4, None))
        for st in ("npxg", "xa"):
            f = (df[f"L{hl}_{st}"] + pm(f"fr_{st}", hl) * d0) / (df[f"L{hl}_team_np_on"] + d0)
            F[f"log_fr_{st}_{hl}"] = np.log(np.clip(f, 1e-4, None))
        fs = (df[f"L{hl}_np_shots"] + pm("fr_shots", hl) * ds) / (df[f"L{hl}_team_shots_on"] + ds)
        F[f"log_fr_shots_{hl}"] = np.log(np.clip(fs, 1e-4, None))
        q = (df[f"L{hl}_npxg"] + pm("quality", hl) * kq) / (df[f"L{hl}_np_shots"] + kq)
        F[f"log_quality_{hl}"] = np.log(np.clip(q, 1e-3, None))
    n = df["L540_n_app"] + 1.0
    for r in ROLES:
        F[f"role_{r}"] = df[f"L540_{r}"] / n
    return F


def fit_rate(cols, Ftr, ytr, etr, Fte, offset=None, alpha=1e-4):
    """Returns predicted RATE per 90 (lambda). `offset`: list of log-columns with coefficient fixed at 1."""
    if offset:
        otr = np.exp(Ftr[offset].sum(1).to_numpy()); ote = np.exp(Fte[offset].sum(1).to_numpy())
        sc = StandardScaler().fit(Ftr[cols])
        m = PoissonRegressor(alpha=alpha, max_iter=1000)
        m.fit(sc.transform(Ftr[cols]), ytr / (etr * otr), sample_weight=etr * otr)
        return m.predict(sc.transform(Fte[cols])) * ote
    sc = StandardScaler().fit(Ftr[cols])
    m = PoissonRegressor(alpha=alpha, max_iter=1000)
    m.fit(sc.transform(Ftr[cols]), ytr / etr, sample_weight=etr)
    return m.predict(sc.transform(Fte[cols]))


def fit_stratified(cols, Ftr, ytr, etr, Fte, tr_pos, te_pos):
    out = np.zeros(len(Fte))
    for p in POS:
        a, b = (tr_pos == p).to_numpy(), (te_pos == p).to_numpy()
        if b.sum() == 0:
            continue
        c = [x for x in cols if not x.startswith("pos_")]
        out[b] = fit_rate(c, Ftr[a], ytr[a], etr[a], Fte[b], alpha=1e-2 if p == "GKP" else 1e-4)
    return out


def specs(target):
    g = target == "goals_scored"
    ab = "npxg" if g else "xa"
    base = ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]
    s1 = base + ["home", "log_lamf"]
    S = {}
    ref = s1 + [f"log_{ab}90_540", f"log_{ab}90_180"]
    S["A0 REF absolute rates (540+180)"] = dict(cols=ref)
    S["A1 STRICT fraction (coef fixed)"] = dict(cols=base, offset=["log_lamf", f"log_fr_{ab}_540"])
    S["A1b STRICT fraction, 1080d"] = dict(cols=base, offset=["log_lamf", f"log_fr_{ab}_1080"])
    S["A2 fraction, free coefficients"] = dict(cols=s1 + [f"log_fr_{ab}_540"])
    S["A2b fraction 540+180"] = dict(cols=s1 + [f"log_fr_{ab}_540", f"log_fr_{ab}_180"])
    S["A3 fraction + absolute rate"] = dict(cols=s1 + [f"log_fr_{ab}_540", f"log_{ab}90_540"])
    if g:
        S["A4 two-factor: shot share x quality"] = dict(cols=s1 + ["log_fr_shots_540", "log_quality_540"])
        S["A4b two-factor + absolute npxG rate"] = dict(cols=s1 + ["log_fr_shots_540", "log_quality_540", "log_npxg90_540"])
    for mult in (0.25, 4.0, 16.0):
        S[f"A5 fraction, shrinkage x{mult}"] = dict(cols=s1 + [f"log_fr_{ab}_540"], d0mult=mult)
    # block B: common base = absolute rates + opponent lambda (+ the target-specific extras)
    extra = ["log_np_shots90_540", "log_np_goals90_540", "log_pen_att90_540", "log_xgchain90_540",
             "log_xgbuildup90_540"] if g else ["log_assists90_540", "log_xgchain90_540", "log_xgbuildup90_540"]
    bb = ref + extra + ["log_lama"]
    S["B0 base (rates + opp lambda)"] = dict(cols=bb)
    S["B1 +role shares"] = dict(cols=bb + [f"role_{r}" for r in ROLES])
    S["B2 +minutes elasticity"] = dict(cols=bb + ["log_minutes"])
    S["B3 +1080d half-life rate"] = dict(cols=bb + [f"log_{ab}90_1080"])
    S["B4 per-position fits"] = dict(cols=bb, strat=True)
    S["B5a prior strength 300 min"] = dict(cols=bb, m0=300.0)
    S["B5b prior strength 2700 min"] = dict(cols=bb, m0=2700.0)
    comb = bb + [f"role_{r}" for r in ROLES] + ["log_minutes", f"log_{ab}90_1080", "log_price", "log_sel",
                                                f"log_fr_{ab}_540"]
    S["B6 combined (all of the above)"] = dict(cols=comb)
    S["B7 combined, per-position"] = dict(cols=comb, strat=True)
    return S


def run_target(target, df, specs_fn=None, recal=True):
    S = (specs_fn or specs)(target)
    dev = df[df.season <= "2024-25"]
    preds = {k: [] for k in S}
    for ts in TEST:
        tr, te = dev[dev.season < ts], dev[dev.season == ts]
        pr = prior_table(tr)
        cache = {}
        etr = (tr.minutes / 90).to_numpy()
        ytr = tr[target].to_numpy(float)
        for name, sp in S.items():
            key = (sp.get("m0", 900.0), sp.get("d0mult", 1.0))
            if key not in cache:
                cache[key] = (build(tr, pr, *key), build(te, pr, *key))
            Ftr, Fte = cache[key]
            if sp.get("strat"):
                lam = fit_stratified(sp["cols"], Ftr, ytr, etr, Fte, tr.position, te.position)
            else:
                lam = fit_rate(sp["cols"], Ftr, ytr, etr, Fte, offset=sp.get("offset"))
            preds[name].append(pd.Series(lam, index=te.index))
        print(f"  {target}: fold {ts}", flush=True)
    test = dev[dev.season.isin(TEST)].copy()
    for k, v in preds.items():
        test[f"mu::{k}"] = pd.concat(v).reindex(test.index) * (test.minutes / 90)
    if not recal:
        return test, list(S.keys())
    # B8: rolling recalibration of B0 (factor from the previous 8 gameweeks of realised/predicted)
    base = test["mu::B0 base (rates + opp lambda)"]
    grp = test.assign(_mu=base).groupby(["season", "round"]).agg(y=(target, "sum"), mu=("_mu", "sum")).reset_index()
    grp = grp.sort_values(["season", "round"]).reset_index(drop=True)
    num = grp.y.shift(1).rolling(8, min_periods=3).sum().fillna(0)
    den = grp.mu.shift(1).rolling(8, min_periods=3).sum().fillna(0)
    grp["f"] = (num + 20) / (den + 20)
    test = test.merge(grp[["season", "round", "f"]], on=["season", "round"], how="left")
    test["mu::B8 base + rolling recalibration"] = base.to_numpy() * test.f.to_numpy()
    return test, list(S.keys()) + ["B8 base + rolling recalibration"]


def compare(target, test, names, ref):
    y = test[target].to_numpy(float)
    e = (test.minutes / 90).to_numpy()
    codes, uniq = pd.factorize(test.season + "_" + test.fixture.astype(str))
    boot = np.random.default_rng(SEED).integers(0, len(uniq), size=(2000, len(uniq)))
    fx = lambda n: np.bincount(codes, weights=L.deviance(y, test[f"mu::{n}"].to_numpy()), minlength=len(uniq))
    efx = np.bincount(codes, weights=e, minlength=len(uniq))
    r = fx(ref)
    rows = []
    for n in names:
        d = fx(n)
        diff = d - r
        samp = diff[boot].sum(1) / efx[boot].sum(1)
        mu = test[f"mu::{n}"].to_numpy()
        rows.append({"model": n, "dev/90": d.sum() / e.sum(), "d_vs_ref": diff.sum() / e.sum(),
                     "lo": np.percentile(samp, 2.5), "hi": np.percentile(samp, 97.5),
                     "sig": "*" if (np.percentile(samp, 2.5) > 0 or np.percentile(samp, 97.5) < 0) else "",
                     "pred/obs": mu.sum() / y.sum()})
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(OUT / "scorer_assist_dataset.parquet")
    df = df[df.season <= "2024-25"].reset_index(drop=True)
    pd.set_option("display.width", 250)
    for target in ("goals_scored", "assists"):
        print(f"\n== {target}", flush=True)
        test, names = run_target(target, df)
        a = [n for n in names if n[0] == "A"]
        b = [n for n in names if n[0] == "B"]
        ta = compare(target, test, a, a[0])
        tb = compare(target, test, b, "B0 base (rates + opp lambda)")
        print("\nBLOCK A  (reference = absolute-rate model; negative = better)\n", ta.round(4).to_string(index=False))
        print("\nBLOCK B  (reference = B0)\n", tb.round(4).to_string(index=False))
        pd.concat([ta, tb]).to_csv(OUT / f"round2_{target}.csv", index=False)
        # subgroup: by amount of history, base vs combined
        y = test[target].to_numpy(float); e = test.minutes / 90
        bucket = pd.cut(test.L540_time / 90, [-1, 5, 15, 30, 1e9], labels=["<5 full games", "5-15", "15-30", "30+"])
        rows = {}
        for n in ("A0 REF absolute rates (540+180)", "A2 fraction, free coefficients",
                  "B0 base (rates + opp lambda)", "B6 combined (all of the above)"):
            d = pd.Series(L.deviance(y, test[f"mu::{n}"].to_numpy()), index=test.index)
            rows[n] = d.groupby(bucket).sum() / e.groupby(bucket).sum()
        sub = pd.DataFrame(rows)
        sub["n_rows"] = bucket.value_counts().sort_index()
        print("\nBY PLAYER HISTORY (dev/90)\n", sub.round(4).to_string())


if __name__ == "__main__":
    main()
