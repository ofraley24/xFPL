#!/usr/bin/env python3
"""Availability model: P(start), P(sub), P(no appearance) for every registered player-fixture, as of the
gameweek deadline (docs/AVAILABILITY_PLAN.md, written before any run).

MODEL. Multinomial logistic regression (scikit-learn), standardised inputs, cubic splines (4 knots) on six key
features, linear terms for the rest, L2 penalty C = 1. Features are the master table's deadline-as-of minutes
and start-history features (06, 12); counts, days, price and ownership log-transformed; missing values 0 with
an indicator. Classes: start (started = 1), sub (minutes > 0, not started), none (0 minutes).
WALK-FORWARD. Development seasons 2019-20..2024-25, each fitted on every earlier season from 2016-17; 2025-26
and 2026-27 scored by models fitted on the seasons before them (reported only). The live model is fitted on
every season before the current one and saved to data/models/availability_live_<season>.pkl.
BASELINE. The player's recent shares (start = starts in the last 10 / fixtures available, appearance = share
of the last 5 played), smoothed towards the position's training rates with the weight of one fixture.
HORIZONS. With --horizons, the same fold models score data/analysis/horizon_rows_all.parquet (features frozen
at the deadline, h = 1..8; 'days since last appearance' reset to its deadline value, see freeze_clock).
Output: data/analysis/availability_predictions.parquet, docs/AVAILABILITY_RESULTS.md.
"""
import hashlib
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT, DOCS, MODELS = ROOT / "data" / "analysis", ROOT / "docs", ROOT / "data" / "models"
DEV = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
REPORT = ["2025-26", "2026-27"]
CLASSES = ["none", "start", "sub"]
POS = ["GKP", "DEF", "MID", "FWD"]
LOG = ["st_n10", "st_since", "absent_streak", "fix_since_app", "days_since_app", "career_apps", "n_prior_rows",
       "price_rank", "m5_rank_pos", "n_active_pos", "value", "selected", "rest_days", "fix_in_event"]
RAW = ["st_full5", "st_mean5", "m1", "m3", "m5", "m10", "f_play5", "f_long5", "f_full5", "ewm3", "std_mpf",
       "prev_season_mpf", "price_gap_top", "own_share_team", "five_subs", "season_progress", "home"]
MISS = ["st_full5", "st_since", "fix_since_app", "prev_season_mpf", "m5", "std_mpf"]   # with indicators
SPLINE = ["st_n10", "m5", "absent_streak", "days_since_app", "price_rank", "ewm3"]


def classes(d):
    return np.where(d.started == 1, 1, np.where(d.minutes > 0, 2, 0))       # index into CLASSES


def design(d):
    """Model inputs from master-table rows (or horizon rows: same columns)."""
    X = pd.DataFrame(index=d.index)
    home = d.was_home.astype(float) if "home" not in d else d.home
    src = d.assign(home=home)
    for c in MISS:
        X[f"miss_{c}"] = src[c].isna().astype(float)
    for c in LOG:
        X[c] = np.log1p(src[c].astype(float).clip(lower=0)).fillna(0.0)
    for c in RAW:
        X[c] = src[c].astype(float).fillna(0.0)
    for p in POS[1:]:
        X[f"pos_{p}"] = (src.position == p).astype(float)
    return X


def make_model():
    cols_lin = [c for c in design_columns() if c not in SPLINE]
    pre = ColumnTransformer([("spl", Pipeline([("s", SplineTransformer(n_knots=4, degree=3)),
                                               ("z", StandardScaler())]), SPLINE),
                             ("lin", StandardScaler(), cols_lin)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=3000))])


def design_columns():
    return [f"miss_{c}" for c in MISS] + LOG + RAW + [f"pos_{p}" for p in POS[1:]]


def fit(tr):
    m = make_model()
    m.fit(design(tr)[design_columns()], classes(tr))
    assert list(m.classes_) == [0, 1, 2]
    return m


def predict(m, d):
    return m.predict_proba(design(d)[design_columns()])       # columns: none, start, sub


def freeze_clock(hz):
    """Horizon rows roll 'days since last appearance' forward to the target kickoff (22: anchor value + days
    from the anchor's kickoff), which assumes no appearance in between. For availability every feature must be
    the state at the deadline, so it is reset to the anchor's (earliest horizon's) value for each player."""
    key = ["season", "asof_round", "element"]
    first = hz.sort_values("horizon").drop_duplicates(key)[key + ["days_since_app"]]
    return hz.drop(columns="days_since_app").merge(first, on=key, how="left")


def recal_x(g):
    p = np.clip(g[["p_none", "p_start", "p_sub"]].to_numpy(float), 1e-9, 1)
    return np.column_stack([np.log(p[:, 1] / p[:, 0]), np.log(p[:, 2] / p[:, 0])])


def fit_recal(pool):
    """Vector scaling: multinomial logistic regression on the two log-odds (plan E)."""
    return LogisticRegression(C=1e6, max_iter=2000).fit(recal_x(pool), pool.y.to_numpy())


def apply_recal(recal, P, h):
    """Recalibrated probabilities (columns none, start, sub) for horizon h; h = 1 or no recalibrator: unchanged."""
    if h < 2 or h not in recal:
        return P
    return recal[h].predict_proba(recal_x(pd.DataFrame(P, columns=["p_none", "p_start", "p_sub"])))


def load_live(season):
    with open(MODELS / f"availability_live_{season}.pkl", "rb") as f:
        return pickle.load(f)


def predict_live(A, d):
    """P(none), P(start), P(sub) for forecast rows (features as of the deadline; 'horizon' column), with the
    per-horizon recalibration of plan E."""
    dz = freeze_clock(d) if "asof_round" in d else d
    P = predict(A["model"], dz)
    for h in np.unique(dz.horizon):
        m = (dz.horizon == h).to_numpy()
        P[m] = apply_recal(A["recal"], P[m], int(h))
    return P


def live_season(master):
    return master.season.max()


def position_rates(tr):
    y = classes(tr)
    return {p: np.bincount(y[(tr.position == p).to_numpy()], minlength=3) / max((tr.position == p).sum(), 1)
            for p in POS}


def baseline(d, rates):
    n = np.minimum(d.n_prior_rows.fillna(0).to_numpy(float), 10)
    s = np.where(n > 0, d.st_n10.fillna(0).to_numpy(float) / np.maximum(n, 1), 0.0)
    a = np.maximum(d.f_play5.fillna(0).to_numpy(float), s)
    own = np.column_stack([1 - a, s, a - s])
    base = np.stack([rates[p] for p in d.position])
    w = n[:, None]
    return (w * own + base) / (w + 1)


def calib(y, p):
    """Calibration in the large and slope (logistic regression of the outcome on logit p)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    r = sm.GLM(y, sm.add_constant(np.log(p / (1 - p))), family=sm.families.Binomial()).fit()
    return p.mean(), y.mean(), float(r.params[1])


def frame_hash(df):
    h = hashlib.sha256(pd.util.hash_pandas_object(df[["season", "element", "fixture", "minutes", "started"]],
                                                  index=False).values.tobytes())
    return h.hexdigest()[:16]


def main():
    args = sys.argv[1:]
    master = pd.read_parquet(OUT / "master.parquet")
    master = master[master.position.isin(POS)].reset_index(drop=True)
    preds, models, lines = [], {}, []
    for ts in DEV + REPORT:
        tr, te = master[master.season < ts], master[master.season == ts]
        m = fit(tr)
        models[ts] = m
        P, B = predict(m, te), baseline(te, position_rates(tr))
        preds.append(te[["season", "round", "element", "code", "position", "fixture", "was_home", "minutes",
                         "started", "total_points"]].assign(
            cls=classes(te), p_none=P[:, 0], p_start=P[:, 1], p_sub=P[:, 2],
            b_none=B[:, 0], b_start=B[:, 1], b_sub=B[:, 2]))
        print(f"{ts}: {len(te):,} rows, log loss model {log_loss(classes(te), P, labels=[0, 1, 2]):.4f} "
              f"baseline {log_loss(classes(te), B, labels=[0, 1, 2]):.4f}", flush=True)
    R = pd.concat(preds, ignore_index=True)
    R.to_parquet(OUT / "availability_predictions.parquet", index=False)

    # ---- evaluation and the pre-registered rule
    rows = []
    for ts, g in R.groupby("season"):
        y = g.cls.to_numpy()
        rows.append({"season": ts, "rows": len(g),
                     "log loss model": log_loss(y, g[["p_none", "p_start", "p_sub"]], labels=[0, 1, 2]),
                     "log loss baseline": log_loss(y, g[["b_none", "b_start", "b_sub"]], labels=[0, 1, 2]),
                     "obs start": (y == 1).mean(), "pred start": g.p_start.mean(),
                     "obs sub": (y == 2).mean(), "pred sub": g.p_sub.mean()})
    T = pd.DataFrame(rows)
    D = R[R.season.isin(DEV)]
    yd = D.cls.to_numpy()
    cal = {k: calib((yd == i).astype(float), D[f"p_{k}"].to_numpy()) for k, i in (("start", 1), ("sub", 2))}
    beats = (T[T.season.isin(DEV)]["log loss model"] < T[T.season.isin(DEV)]["log loss baseline"]).all()
    cal_ok = all(abs(v[0] - v[1]) <= 0.01 and 0.9 <= v[2] <= 1.1 for v in cal.values())
    lines += ["# Availability model: results\n",
              "Plan and rule (written before any run): docs/AVAILABILITY_PLAN.md. Script: analysis/30_availability.py.\n",
              f"**Rule: {'ADOPTED' if beats and cal_ok else 'NOT ADOPTED'}** (beats the baseline in every development "
              f"season: {beats}; calibration within tolerance: {cal_ok}).\n",
              T.round(4).to_markdown(index=False), "",
              "Calibration on the pooled development rows (mean predicted, observed, slope of outcome on logit p):\n",
              pd.DataFrame(cal, index=["mean predicted", "observed", "slope"]).T.round(4).to_markdown(), ""]
    # reliability by decile of P(start) and P(sub), development
    for k, i in (("start", 1), ("sub", 2)):
        q = pd.qcut(D[f"p_{k}"].rank(method="first"), 10, labels=False)
        rel = D.groupby(q).agg(pred=(f"p_{k}", "mean"), obs=("cls", lambda s, i=i: (s == i).mean()), n=("cls", "size"))
        lines += [f"Reliability of P({k}) by decile, development:\n", rel.round(3).T.to_markdown(), ""]
    # by position
    bp = D.groupby("position").apply(lambda g: pd.Series({
        "obs start": (g.cls == 1).mean(), "pred start": g.p_start.mean(),
        "obs sub": (g.cls == 2).mean(), "pred sub": g.p_sub.mean()}))
    lines += ["By position, development:\n", bp.round(4).to_markdown(), ""]

    # ---- horizons (report only)
    recal = {}
    if "--horizons" in args and (OUT / "horizon_rows_all.parquet").exists():
        hz = freeze_clock(pd.read_parquet(OUT / "horizon_rows_all.parquet"))
        hz = hz[hz.position.isin(POS)]
        parts = []
        for (ts, h), g in hz.groupby(["season", "horizon"]):
            if ts in models:
                P = predict(models[ts], g)
                parts.append(pd.DataFrame({"season": ts, "horizon": h, "y": classes(g),
                                           "p_none": P[:, 0], "p_start": P[:, 1], "p_sub": P[:, 2]}))
        HZ = pd.concat(parts, ignore_index=True)
        # E: per-horizon recalibration, fitted on earlier seasons' out-of-sample horizon predictions only
        cols = ["r_none", "r_start", "r_sub"]
        HZ[cols] = HZ[["p_none", "p_start", "p_sub"]].to_numpy()
        for (ts, h), idx in HZ.groupby(["season", "horizon"]).groups.items():
            pool = HZ[(HZ.season < ts) & (HZ.horizon == h)]
            if h >= 2 and len(pool):
                HZ.loc[idx, cols] = fit_recal(pool).predict_proba(recal_x(HZ.loc[idx]))
        for h in sorted(HZ.horizon.unique()):
            if h >= 2:
                recal[int(h)] = fit_recal(HZ[(HZ.horizon == h) & (HZ.season < live_season(master))])
        tab = []
        dev2 = HZ[HZ.season.isin(DEV[1:])]
        for h, g in dev2.groupby("horizon"):
            y = g.y.to_numpy()
            row = {"horizon": h, "rows": len(g), "obs start": (y == 1).mean(), "obs sub": (y == 2).mean()}
            for tag, c in (("raw", ["p_none", "p_start", "p_sub"]), ("recal", cols)):
                P = g[c].to_numpy()
                row |= {f"log loss {tag}": log_loss(y, P, labels=[0, 1, 2]), f"pred start {tag}": P[:, 1].mean(),
                        f"slope start {tag}": calib((y == 1).astype(float), P[:, 1])[2],
                        f"slope sub {tag}": calib((y == 2).astype(float), P[:, 2])[2]}
            tab.append(row)
        TH = pd.DataFrame(tab)
        g2 = TH[TH.horizon >= 2]
        ok_e = bool((g2["log loss recal"] < g2["log loss raw"]).all() and
                    g2[["slope start recal", "slope sub recal"]].apply(lambda c: c.between(0.9, 1.1)).all().all())
        if not ok_e:
            recal = {}
        lines += ["## Horizons: features frozen at the deadline, development seasons 2020-21..2024-25\n",
                  f"**Recalibration (plan E): {'ADOPTED' if ok_e else 'NOT ADOPTED'}** (log loss lower at every "
                  "h >= 2 and slopes within [0.9, 1.1]).\n",
                  TH.round(4).to_markdown(index=False), ""]
        HZ.to_parquet(OUT / "availability_horizon_predictions.parquet", index=False)

    # ---- live model: every season before the current one
    live = live_season(master)
    tr = master[master.season < live]
    MODELS.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    obj = {"model": models[live] if live in models else fit(tr), "season": live, "rates": position_rates(tr),
           "train_hash": frame_hash(tr), "recal": recal, "code_commit": git.stdout.strip(), "columns": design_columns(),
           "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    with open(MODELS / f"availability_live_{live}.pkl", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    lines += [f"Live model: fitted on {len(tr):,} rows (seasons before {live}), saved to "
              f"data/models/availability_live_{live}.pkl.\n"]
    (DOCS / "AVAILABILITY_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
