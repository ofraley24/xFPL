#!/usr/bin/env python3
"""Minutes component: predict each player's minutes distribution BEFORE the deadline.

Output per player-fixture: probabilities over 8 minute buckets
  0 | 1-14 | 15-29 | 30-44 | 45-59 | 60-74 | 75-89 | 90
and derived quantities xP needs: P(plays), P(60+) [appearance points 1 vs 2], E[minutes], and
E[(m/90)^0.8] [the scoring-rate scale; goals/assists per appearance scale ~ m^0.8, see round 2].

Models (walk-forward, test 2019-20..2024-25, dev seasons only; sealed 2025-26 untouched):
  B0 position climatology        B1 'as last match' (bucket persistence, smoothed)
  B2 multinomial logistic         B3 gradient-boosted classifier
UNIVERSE: players with any minutes in their previous <=10 fixtures (recall of true appearances is
reported). Players with no recent usage are mostly irrelevant to FPL decisions and dominate the
zero class; they would flatter every model.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
SEED = 20261011
TEST = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
EDGES = [0.5, 14.5, 29.5, 44.5, 59.5, 74.5, 89.5]
K = 8
POS = ["GKP", "DEF", "MID", "FWD"]
B_EL = 0.8

FEATS = ["m1", "m3", "m5", "m10", "f_play5", "f_long5", "f_full5", "ewm3", "absent_streak", "fix_since_app",
         "days_since_app", "std_mpf", "prev_season_mpf", "career_apps", "n_prior_rows",
         "price_rank", "price_gap_top", "own_share_team", "m5_rank_pos", "n_active_pos", "value", "selected_log",
         "home", "rest_days", "opp_rest_days", "next_gap_days", "fix_in_event", "supremacy", "total_goals",
         "season_progress", "pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]


def prep(d):
    d = d.copy()
    d["bucket"] = np.digitize(d.minutes.to_numpy(), EDGES)
    d["selected_log"] = np.log1p(d.selected)
    d["home"] = d.was_home.astype(float)
    for p in POS:
        d[f"pos_{p}"] = (d.position == p).astype(float)
    return d


def bucket_stats(tr):
    mean_m = tr.groupby("bucket").minutes.mean().reindex(range(K)).fillna(0).to_numpy()
    mean_pow = tr.assign(pw=(tr.minutes / 90) ** B_EL).groupby("bucket").pw.mean().reindex(range(K)).fillna(0).to_numpy()
    return mean_m, mean_pow


def b0(tr, te):
    tab = tr.groupby("position").bucket.apply(lambda s: np.bincount(s, minlength=K) / len(s))
    return np.vstack([tab[p] for p in te.position])


def b1(tr, te):
    def last(d):
        return np.where(d.m1.isna(), K, np.digitize(d.m1.fillna(0).to_numpy(), EDGES))
    ltr, lte = last(tr), last(te)
    base = np.bincount(tr.bucket, minlength=K) / len(tr)
    P = np.zeros((K + 1, K))
    for c in range(K + 1):
        cnt = np.bincount(tr.bucket[ltr == c], minlength=K).astype(float)
        P[c] = (cnt + 5 * base) / (cnt.sum() + 5)   # Dirichlet smoothing towards the marginal
    return P[lte]


def b2(tr, te):
    def mat(d):
        X = d[FEATS].copy()
        for c in ("m1", "m3", "m5", "m10", "f_play5", "f_long5", "f_full5", "ewm3", "std_mpf", "prev_season_mpf",
                  "days_since_app", "fix_since_app", "rest_days", "opp_rest_days", "next_gap_days"):
            X[f"{c}_na"] = X[c].isna().astype(float)
        X["days_since_app"] = np.log1p(X.days_since_app.fillna(400))
        X["fix_since_app"] = np.log1p(X.fix_since_app.fillna(50))
        X["career_apps"] = np.log1p(X.career_apps)
        return X.fillna(X.median(numeric_only=True)).fillna(0)
    Xtr, Xte = mat(tr), mat(te)
    med = Xtr.median()
    Xte = Xte.fillna(med)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=600, C=0.5)
    m.fit(sc.transform(Xtr), tr.bucket)
    P = np.zeros((len(te), K))
    P[:, m.classes_] = m.predict_proba(sc.transform(Xte))
    return P


def b3(tr, te, return_model=False):
    m = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.06, max_depth=5, min_samples_leaf=100,
                                       l2_regularization=1.0, random_state=SEED)
    m.fit(tr[FEATS], tr.bucket)
    P = np.zeros((len(te), K))
    P[:, m.classes_] = m.predict_proba(te[FEATS])
    return (P, m) if return_model else P


def metrics(P, y, mean_m, m_actual):
    P = np.clip(P, 1e-9, 1)
    P = P / P.sum(1, keepdims=True)
    ll = -np.log(P[np.arange(len(y)), y])
    cum = np.cumsum(P, 1)
    obs = (np.arange(K)[None, :] >= y[:, None]).astype(float)
    rps = ((cum - obs) ** 2).sum(1) / (K - 1)
    p_play = 1 - P[:, 0]
    p60 = P[:, 5:].sum(1)
    em = P @ mean_m
    return dict(ll=ll, rps=rps, brier_play=(p_play - (m_actual > 0)) ** 2, brier_60=(p60 - (m_actual >= 60)) ** 2,
                abs_err=np.abs(em - m_actual), sq_err=(em - m_actual) ** 2, em=em, p_play=p_play, p60=p60)


def main():
    d = prep(pd.read_parquet(OUT / "minutes_dataset.parquet"))
    d = d[d.season <= "2024-25"].reset_index(drop=True)
    uni = d.m10.fillna(0) > 0
    rec = {s: ((d.minutes > 0) & uni)[d.season == s].sum() / (d.minutes > 0)[d.season == s].sum() for s in TEST}
    print("universe recall of true appearances by season:", {k: round(v, 3) for k, v in rec.items()})
    print(f"universe share of rows: {uni.mean():.3f}; zero-minute share inside universe: {(d.minutes[uni] == 0).mean():.3f}")
    u = d[uni].reset_index(drop=True)

    models = {"B0 position climatology": b0, "B1 as last match": b1, "B2 multinomial logistic": b2, "B3 gradient boosting": b3}
    store = {k: [] for k in models}  # DataFrames indexed like the test rows
    rows = []
    last_fold = None
    for ts in TEST:
        tr, te = u[u.season < ts], u[u.season == ts]
        mean_m, mean_pow = bucket_stats(tr)
        y, ma = te.bucket.to_numpy(), te.minutes.to_numpy(float)
        for name, fn in models.items():
            P = fn(tr, te)
            store[name].append(pd.DataFrame(P, index=te.index))
            mt = metrics(P, y, mean_m, ma)
            rows.append({"season": ts, "model": name, **{k: v.mean() for k, v in mt.items() if k not in ("em", "p_play", "p60")}})
        if ts == TEST[-1]:
            last_fold = (tr, te)
        print(f"  fold {ts}", flush=True)
    test = u[u.season.isin(TEST)]   # keep u's index: predictions are aligned on it
    mean_m_all = None
    # pooled metrics use each fold's own bucket means for E[m]
    test = test.copy()
    y = test.bucket.to_numpy(); ma = test.minutes.to_numpy(float)
    em_cols = {}
    per_row = {}
    off = 0
    for i, ts in enumerate(TEST):
        pass
    cum = {name: pd.concat(store[name]).reindex(test.index).to_numpy() for name in models}
    assert all(not np.isnan(v).any() for v in cum.values()), 'prediction alignment failed'
    # E[m] with fold-specific bucket means
    fold_id = test.season.map({s: i for i, s in enumerate(TEST)}).to_numpy()
    mm = np.vstack([bucket_stats(u[u.season < s])[0] for s in TEST])
    mp = np.vstack([bucket_stats(u[u.season < s])[1] for s in TEST])
    codes, uniq = pd.factorize(test.season + "_" + test.fixture.astype(str))
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, len(uniq), size=(1500, len(uniq)))
    res, perfx = [], {}
    for name in models:
        P = np.clip(cum[name], 1e-9, 1); P = P / P.sum(1, keepdims=True)
        ll = -np.log(P[np.arange(len(y)), y])
        rps = (((np.cumsum(P, 1) - (np.arange(K)[None, :] >= y[:, None])) ** 2).sum(1)) / (K - 1)
        em = np.einsum("ij,ij->i", P, mm[fold_id])
        p60 = P[:, 5:].sum(1)
        perfx[name] = (np.bincount(codes, weights=ll, minlength=len(uniq)), np.bincount(codes, weights=rps, minlength=len(uniq)))
        res.append({"model": name, "logloss": ll.mean(), "RPS": rps.mean(), "brier P(play)": ((1 - P[:, 0] - (ma > 0)) ** 2).mean(),
                    "brier P(60+)": ((p60 - (ma >= 60)) ** 2).mean(), "MAE E[min]": np.abs(em - ma).mean(),
                    "bias E[min]": (em - ma).mean()})
        test[f"em::{name}"] = em
        test[f"pplay::{name}"] = 1 - P[:, 0]
        test[f"p60::{name}"] = p60
    n_fx = np.bincount(codes, minlength=len(uniq))
    base = "B0 position climatology"
    for r in res:
        for tag, idx in (("logloss", 0), ("RPS", 1)):
            diff = perfx[r["model"]][idx] - perfx[base][idx]
            samp = diff[boot].sum(1) / n_fx[boot].sum(1)
            r[f"d_{tag}_vs_B0"] = diff.sum() / n_fx.sum()
            r[f"ci_{tag}"] = f"[{np.percentile(samp, 2.5):+.4f}, {np.percentile(samp, 97.5):+.4f}]"
    # naive minute rules for MAE context
    naive = pd.DataFrame({"rule": ["last match minutes", "mean of last 5", "mean of last 10"],
                          "MAE": [np.abs(test.m1.fillna(0) - ma).mean(), np.abs(test.m5.fillna(0) - ma).mean(),
                                  np.abs(test.m10.fillna(0) - ma).mean()]})
    pd.set_option("display.width", 250)
    print("\nPOOLED (universe rows, test 2019-20..2024-25)\n", pd.DataFrame(res).round(4).to_string(index=False))
    print("\nNaive minute rules (MAE of the point forecast)\n", naive.round(3).to_string(index=False))
    pd.DataFrame(res).to_csv(OUT / "minutes_results.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "minutes_results_by_fold.csv", index=False)

    best = "B3 gradient boosting"
    # calibration of P(plays) and P(60+) by decile, best model
    for col, truth, nm in ((f"pplay::{best}", ma > 0, "P(plays)"), (f"p60::{best}", ma >= 60, "P(60+)")):
        q = pd.qcut(test[col], 10, labels=False, duplicates="drop")
        c = pd.DataFrame({"pred": test[col], "obs": truth.astype(float)}).groupby(q).mean()
        print(f"\ncalibration {nm} ({best}) by decile of prediction\n", c.round(3).T.to_string())
    # where it fails: subgroups
    sub = {}
    for nm, mask in (("absent >=3 fixtures", test.absent_streak >= 3), ("returning after >=30 days", test.days_since_app >= 30),
                     ("new to FPL (<5 prior rows)", test.n_prior_rows < 5), ("double gameweek", test.fix_in_event >= 2),
                     ("regulars (m5>=75)", test.m5 >= 75), ("all", pd.Series(True, index=test.index))):
        mk = mask.to_numpy()
        sub[nm] = {"n": int(mk.sum()), "actual P(plays)": (ma[mk] > 0).mean(),
                   "pred P(plays)": test[f"pplay::{best}"][mk].mean(), "actual mean min": ma[mk].mean(),
                   "pred mean min": test[f"em::{best}"][mk].mean(), "MAE B3": np.abs(test[f"em::{best}"][mk] - ma[mk]).mean(),
                   "MAE last-match": np.abs(test.m1.fillna(0)[mk] - ma[mk]).mean()}
    print("\nSUBGROUPS (best model)\n", pd.DataFrame(sub).T.round(3).to_string())
    # permutation importance on the last fold
    tr, te = last_fold
    P, mdl = b3(tr, te, return_model=True)
    samp = te.sample(8000, random_state=SEED)
    imp = permutation_importance(mdl, samp[FEATS], samp.bucket, n_repeats=3, random_state=SEED, scoring="neg_log_loss")
    s = pd.Series(imp.importances_mean, index=FEATS).sort_values(ascending=False)
    print("\nTop permutation importances (increase in log-loss when shuffled), fold 2024-25\n", s.head(12).round(4).to_string())
    keep = test[["season", "round", "fixture", "element", "code", "position", "minutes", "bucket"]].copy()
    for j in range(K):
        keep[f"p{j}"] = cum[best][:, j]
    keep["p_play"], keep["p_60plus"] = test[f"pplay::{best}"], test[f"p60::{best}"]
    keep["e_minutes"] = test[f"em::{best}"]
    keep["e_scale_m08"] = np.einsum("ij,ij->i", cum[best], mp[fold_id])
    keep.to_parquet(OUT / "minutes_predictions_B3.parquet", index=False)


if __name__ == "__main__":
    main()
