#!/usr/bin/env python3
"""Round 4.

PART H  Where does home advantage enter?  The pi-rating lambda is built as
        log(lambda_home) = c + home_adv + (att_h - def_a)/scale,  home_adv = 0.23 (x1.26),
        so the venue effect is already inside log_lamf. We test that directly: the coefficient on a
        `home` dummy with and without lambda in the model, and the deviance value of the dummy.
PART P  Penalties as a separate component.  FPL goals = open-play goals + penalty goals (99.99%
        identical in the data).  T0 unified model on FPL goals (round-3 C4 spec)
        T1 split: open-play Poisson model + (taker-fraction penalty-attempt model x conversion)
        T2 split, but penalty part WITHOUT the taker information (to price the taker signal)
PART E  Empirical-Bayes / hierarchical shrinkage vs the tuned fixed shrinkage.
        H0 fixed: shrunk = (S + prior*M0) / (T + M0), one M0 for everyone (tuned, x0.25)
        H1 hierarchical (gamma-Poisson): S ~ Poisson(T*theta), theta ~ Gamma(k_g, k_g/m); the
           strength k_g is ESTIMATED by marginal likelihood separately for each position g and
           each statistic, instead of one hand-chosen M0
        H2 as H1, and the prior MEAN m is regressed on covariates (position, price, team lambda),
           so a newcomer is shrunk towards 'what players like him do', not the position average
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.linear_model import PoissonRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("r2", ROOT / "analysis" / "03_fractions_and_engineering.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
L, OUT, POS, TEST = M.L, M.OUT, M.POS, M.TEST
BASE = ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]
TAIL = ["log_minutes", "log_price", "log_sel", "home", "log_lama"]
FAC_G = ["log_lamf", "log_fr_shots_540", "log_quality_540"]
FAC_A = ["log_lamf", "log_fr_xa_540"]


# ------------------------------------------------------------------ PART H
def part_home(df):
    dev = df[df.season <= "2024-25"]
    pr = M.prior_table(dev)
    F = M.build(dev, pr, 900.0, 0.25)
    e = dev.minutes / 90
    print("PART H: home advantage (unscaled Poisson fits, dev seasons 2016-17..2024-25)")
    for target in ("goals_scored", "assists"):
        y = dev[target] / e
        rows = {}
        for label, cols in (("home only", ["home"]),
                            ("home + position", BASE[1:] + ["home"]),
                            ("home + position + team lambda", BASE[1:] + ["home", "log_lamf"]),
                            ("home + position + team lambda + player fraction",
                             BASE[1:] + ["home", "log_lamf"] + (FAC_G[1:] if target == "goals_scored" else FAC_A[1:]))):
            m = PoissonRegressor(alpha=0, max_iter=3000).fit(F[cols], y, sample_weight=e)
            rows[label] = np.exp(m.coef_[cols.index("home")])
        print(f"  {target}: multiplier on a player's rate when playing at home")
        for k, v in rows.items():
            print(f"     {k:52s} x{v:.3f}")
    ha = np.exp(0.2301)
    print(f"  pi-rating home advantage built into team lambda: x{ha:.3f}")
    hl = dev.groupby("was_home").agg(lam=("xg_for_pi", "mean"))
    print(f"  mean team lambda home/away: {hl.loc[True,'lam']:.3f} / {hl.loc[False,'lam']:.3f}  "
          f"(ratio {hl.loc[True,'lam']/hl.loc[False,'lam']:.3f})")


# ------------------------------------------------------------------ PART P
def pen_frame(df, tr_ref):
    prior = {p: tr_ref[tr_ref.position == p].L540_pen_att.sum() / max(tr_ref[tr_ref.position == p].L540_team_pen_on.sum(), 1e-9)
             for p in POS}
    return prior


def pen_features(F, df, prior, dpen):
    G = F.copy()
    pm = df.position.map(prior).to_numpy()
    G["log_fr_pen_540"] = np.log(np.clip((df.L540_pen_att + pm * dpen) / (df.L540_team_pen_on + dpen), 1e-4, None))
    return G


def run_penalty(df):
    dev = df[df.season <= "2024-25"]
    preds = {k: [] for k in ("T0 unified (FPL goals, C4)", "T1 split: open-play + penalty(taker)",
                             "T2 split: penalty WITHOUT taker info")}
    pen_rows = []
    for ts in TEST:
        tr, te = dev[dev.season < ts], dev[dev.season == ts]
        pr = M.prior_table(tr)
        Ftr, Fte = M.build(tr, pr, 900.0, 0.25), M.build(te, pr, 900.0, 0.25)
        etr = (tr.minutes / 90).to_numpy()
        c4 = BASE + FAC_G + TAIL
        preds["T0 unified (FPL goals, C4)"].append(pd.Series(
            M.fit_rate(c4, Ftr, tr.goals_scored.to_numpy(float), etr, Fte), index=te.index))
        lam_open = M.fit_rate(c4, Ftr, tr.tgt_np_goals.to_numpy(float), etr, Fte)
        ppr = pen_frame(tr, tr)
        Ptr, Pte = pen_features(Ftr, tr, ppr, 1.5), pen_features(Fte, te, ppr, 1.5)
        pcols = BASE[1:] + ["log_lamf", "log_lama", "log_minutes", "log_fr_pen_540"]
        lam_pen = M.fit_rate(pcols, Ptr, tr.tgt_pen_att.to_numpy(float), etr, Pte)
        lam_pen0 = M.fit_rate([c for c in pcols if c != "log_fr_pen_540"], Ptr, tr.tgt_pen_att.to_numpy(float), etr, Pte)
        conv = tr.tgt_pen_goals.sum() / tr.tgt_pen_att.sum()
        preds["T1 split: open-play + penalty(taker)"].append(pd.Series(lam_open + conv * lam_pen, index=te.index))
        preds["T2 split: penalty WITHOUT taker info"].append(pd.Series(lam_open + conv * lam_pen0, index=te.index))
        # penalty-attempt diagnostics
        e = (te.minutes / 90).to_numpy()
        for nm, lam in (("with taker fraction", lam_pen), ("without", lam_pen0)):
            pen_rows.append({"season": ts, "model": nm,
                             "dev/90": L.deviance(te.tgt_pen_att.to_numpy(float), lam * e).sum() / e.sum(),
                             "pred": (lam * e).sum(), "obs": te.tgt_pen_att.sum()})
        print(f"  penalty: fold {ts}", flush=True)
    test = dev[dev.season.isin(TEST)].copy()
    for k, v in preds.items():
        test[f"mu::{k}"] = pd.concat(v).reindex(test.index) * (test.minutes / 90)
    names = list(preds)
    print("\nPART P: total FPL goals (reference = unified model; negative = better)")
    print(M.compare("goals_scored", test, names, names[0]).round(4).to_string(index=False))
    pr_df = pd.DataFrame(pen_rows).groupby("model").agg(dev=("dev/90", "mean"), pred=("pred", "sum"), obs=("obs", "sum"))
    print("\npenalty ATTEMPTS only (avg of fold deviances):\n", pr_df.round(3).to_string())
    # penalties among players the model flags as takers
    return test


# ------------------------------------------------------------------ PART E
def nb_fit(S, T, grp, Z, init_logk=2.0):
    """Gamma-Poisson marginal likelihood: S ~ Poisson(T*theta), theta ~ Gamma(k, k/m), m = exp(Z g),
    k per group. S and T are decayed (non-integer) sums, handled through gammaln."""
    ok = T > 0.05
    S, T, grp, Z = S[ok], T[ok], grp[ok], Z[ok]
    nz = Z.shape[1]

    def nll(p):
        m = np.exp(Z @ p[:nz])
        k = np.exp(p[nz:])[grp]
        mu = T * m
        ll = gammaln(S + k) - gammaln(k) - gammaln(S + 1) + k * np.log(k / (k + mu)) + S * np.log(mu / (k + mu) + 1e-300)
        return -ll.sum()

    p0 = np.concatenate([np.zeros(nz), np.full(grp.max() + 1, init_logk)])
    r = minimize(nll, p0, method="L-BFGS-B")
    return r.x[:nz], np.exp(r.x[nz:])


def eb_posterior(S, T, grp, Z, g, k):
    m = np.exp(Z @ g)
    kk = k[grp]
    return (S + kk) / (T + kk / m), m, kk


def hier_features(tr, te, Ftr, Fte, covariates):
    """Replace the fixed-shrinkage columns by hierarchical (empirical-Bayes) posterior means."""
    gtr = tr.position.map({p: i for i, p in enumerate(POS)}).to_numpy()
    gte = te.position.map({p: i for i, p in enumerate(POS)}).to_numpy()

    def Zmat(d, F):
        cols = [np.eye(4)[d.position.map({p: i for i, p in enumerate(POS)}).to_numpy()]]
        if covariates:
            cols.append(np.c_[F.log_price - 4.2, F.log_lamf - 0.3])
        return np.hstack(cols)

    Ztr, Zte = Zmat(tr, Ftr), Zmat(te, Fte)
    Ftr, Fte = Ftr.copy(), Fte.copy()
    out = {}
    specs = [("log_fr_shots_540", "L540_np_shots", "L540_team_shots_on"),
             ("log_fr_xa_540", "L540_xa", "L540_team_np_on"),
             ("log_fr_npxg_540", "L540_npxg", "L540_team_np_on")]
    for name, num, den in specs:
        g, k = nb_fit(tr[num].to_numpy(float), tr[den].to_numpy(float), gtr, Ztr)
        for F, d, Z, gg in ((Ftr, tr, Ztr, gtr), (Fte, te, Zte, gte)):
            th, _, _ = eb_posterior(d[num].to_numpy(float), d[den].to_numpy(float), gg, Z, g, k)
            F[name] = np.log(np.clip(th, 1e-4, None))
        out[name] = k
    # shot quality: npxG per shot, as xG in units of the position-average quality
    q0 = tr.L540_npxg.sum() / tr.L540_np_shots.sum()
    g, k = nb_fit(tr.L540_npxg.to_numpy() / q0, tr.L540_np_shots.to_numpy(float), gtr, Ztr)
    for F, d, Z, gg in ((Ftr, tr, Ztr, gtr), (Fte, te, Zte, gte)):
        th, _, _ = eb_posterior(d.L540_npxg.to_numpy() / q0, d.L540_np_shots.to_numpy(float), gg, Z, g, k)
        F["log_quality_540"] = np.log(np.clip(th * q0, 1e-3, None))
    out["log_quality_540"] = k
    return Ftr, Fte, out


def run_hier(df):
    dev = df[df.season <= "2024-25"]
    res = {}
    kstore = {}
    for target, fac in (("goals_scored", FAC_G), ("assists", FAC_A)):
        names = ["H0 fixed shrinkage (tuned x0.25)", "H1 hierarchical, strength per position",
                 "H2 hierarchical + covariate prior mean"]
        preds = {n: [] for n in names}
        for ts in TEST:
            tr, te = dev[dev.season < ts], dev[dev.season == ts]
            pr = M.prior_table(tr)
            Ftr0, Fte0 = M.build(tr, pr, 900.0, 0.25), M.build(te, pr, 900.0, 0.25)
            etr = (tr.minutes / 90).to_numpy()
            cols = BASE + fac + TAIL
            y = tr[target].to_numpy(float)
            preds[names[0]].append(pd.Series(M.fit_rate(cols, Ftr0, y, etr, Fte0), index=te.index))
            for n, cov in ((names[1], False), (names[2], True)):
                Ftr, Fte, ks = hier_features(tr, te, Ftr0, Fte0, cov)
                preds[n].append(pd.Series(M.fit_rate(cols, Ftr, y, etr, Fte), index=te.index))
                if ts == TEST[-1]:
                    kstore[(target, n)] = ks
            print(f"  hier {target}: fold {ts}", flush=True)
        test = dev[dev.season.isin(TEST)].copy()
        for k, v in preds.items():
            test[f"mu::{k}"] = pd.concat(v).reindex(test.index) * (test.minutes / 90)
        print(f"\nPART E: {target} (reference = fixed shrinkage; negative = better)")
        print(M.compare(target, test, names, names[0]).round(4).to_string(index=False))
        res[target] = test
    print("\nestimated prior strengths k (pseudo-exposure) per position GKP/DEF/MID/FWD, last fold, H1:")
    for (t, n), ks in kstore.items():
        if n.startswith("H1"):
            for stat, k in ks.items():
                print(f"   {t:13s} {stat:20s} " + "  ".join(f"{x:8.2f}" for x in k))
    return res


def main():
    pd.set_option("display.width", 250)
    df = pd.read_parquet(OUT / "scorer_assist_dataset.parquet")
    df = df[df.season <= "2024-25"].reset_index(drop=True)
    part_home(df)
    print()
    run_penalty(df)
    print()
    run_hier(df)


if __name__ == "__main__":
    main()
