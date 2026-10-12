#!/usr/bin/env python3
"""Trustworthy empirical-Bayes shrinkage via SPLIT-HALF estimation of talent vs noise.

Model for a ratio statistic (numerator S, exposure T):  r_h = S_h/T_h = theta + e_h,
Var(e_h) = sigma2 / T_h,  theta ~ (m_p, tau2_p)  [m_p, tau2_p per position p].
Each player-season is split into odd/even matches -> two NON-OVERLAPPING measurements of the same
player. Then, with no distributional assumption (no Poisson, works for xG sums and ratios):
    sigma2 = sum (r_o - r_e)^2 / sum (1/T_o + 1/T_e)          [noise, from the difference]
    tau2   = weighted mean (r_o - m)(r_e - m)                  [talent, from the covariance]
    k      = sigma2 / tau2                                     [prior strength in exposure units]
Posterior mean from accumulated exposure T:   (S + k m) / (T + k).
Estimated on seasons BEFORE each test season only. Replaces the failed Poisson-likelihood version.

Compared (walk-forward, same GLM, same folds as rounds 3-4):
  H0   fixed shrinkage, one tuned pseudo-count for everyone (round-3 spec)
  EB   split-half k per statistic and position, multiplied by 0.5 / 1 / 2 to test whether the
       estimate sits near the optimum (a trustworthy estimate should)
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("r2", ROOT / "analysis" / "03_fractions_and_engineering.py")
M = importlib.util.module_from_spec(sp)
sp.loader.exec_module(M)
sp1 = importlib.util.spec_from_file_location("b1", ROOT / "analysis" / "01_build_scorer_assist_dataset.py")
B1 = importlib.util.module_from_spec(sp1)
sp1.loader.exec_module(B1)
POS, TEST, OUT = M.POS, M.TEST, M.OUT
BASE, TAIL = ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"], ["log_minutes", "log_price", "log_sel", "home", "log_lama"]
FAC_G, FAC_A = ["log_lamf", "log_fr_shots_540", "log_quality_540"], ["log_lamf", "log_fr_xa_540"]
COARSE = {"GK": "GKP", "DC": "DEF", "DL": "DEF", "DR": "DEF", "DMC": "MID", "DML": "MID", "DMR": "MID", "MC": "MID",
          "ML": "MID", "MR": "MID", "AMC": "MID", "AML": "MID", "AMR": "MID", "FW": "FWD", "FWL": "FWD", "FWR": "FWD"}
# statistic -> (numerator column, exposure column, dataset columns of the decayed sums, feature name)
STATS = {"shots_fr": ("np_shots", "team_shots_on", "L540_np_shots", "L540_team_shots_on", "log_fr_shots_540", 40.0),
         "quality": ("npxg", "np_shots", "L540_npxg", "L540_np_shots", "log_quality_540", 6.0),
         "xa_fr": ("xa", "team_np_on", "L540_xa", "L540_team_np_on", "log_fr_xa_540", 6.0)}


def player_seasons():
    pms, shots, matches = B1.load_understat()
    matches = matches[matches.isResult == True].drop_duplicates("match_id")  # noqa: E712
    s = shots[shots.result != "OwnGoal"].copy()
    s["pen"] = s.situation == "Penalty"
    s["npxg"] = np.where(s.pen, 0.0, s.xG)
    pl = s.groupby(["match_id", "player_id"]).agg(np_shots=("pen", lambda x: (~x).sum()), npxg=("npxg", "sum")).reset_index()
    tm = s.groupby(["match_id", "h_a"]).agg(team_npxg=("npxg", "sum"), team_shots=("pen", lambda x: (~x).sum())).reset_index()
    d = pms.merge(matches[["match_id", "date"]], on="match_id").merge(pl, on=["match_id", "player_id"], how="left") \
        .merge(tm, on=["match_id", "h_a"], how="left")
    for c in ("np_shots", "npxg", "team_npxg", "team_shots"):
        d[c] = d[c].fillna(0.0)
    d = d.rename(columns={"xA": "xa"})
    d["team_np_on"] = d.team_npxg * d.time / 90.0
    d["team_shots_on"] = d.team_shots * d.time / 90.0
    d["pos_c"] = d.position.map(COARSE)
    modal = d[d.pos_c.notna()].groupby(["player_id", "season"]).pos_c.agg(lambda x: x.value_counts().index[0])
    d = d.join(modal.rename("grp"), on=["player_id", "season"])
    d = d[d.grp.notna()].sort_values(["player_id", "season", "date"])
    d["half"] = d.groupby(["player_id", "season"]).cumcount() % 2
    return d


def split_half_table(d):
    keys = ["player_id", "season", "grp", "half"]
    cols = sorted({c for v in STATS.values() for c in v[:2]})
    g = d.groupby(keys)[cols].sum().reset_index()
    o, e = g[g.half == 0], g[g.half == 1]
    return o.merge(e, on=["player_id", "season", "grp"], suffixes=("_o", "_e"))


def estimate_k(H, seasons_max):
    """k, m per (statistic, position) from player-seasons with season <= seasons_max."""
    H = H[H.season <= seasons_max]
    out = {}
    for name, (num, den, *_rest, tmin) in STATS.items():
        for p in POS:
            h = H[H.grp == p]
            To, Te = h[f"{den}_o"].to_numpy(), h[f"{den}_e"].to_numpy()
            ok = (To >= tmin) & (Te >= tmin)
            if ok.sum() < 30:
                out[(name, p)] = (1e6, h[f"{num}_o"].sum() / max(h[f"{den}_o"].sum(), 1e-9), int(ok.sum()))
                continue
            To, Te = To[ok], Te[ok]
            ro, re = h[f"{num}_o"].to_numpy()[ok] / To, h[f"{num}_e"].to_numpy()[ok] / Te
            m = (h[f"{num}_o"].sum() + h[f"{num}_e"].sum()) / (h[f"{den}_o"].sum() + h[f"{den}_e"].sum())
            sigma2 = ((ro - re) ** 2).sum() / (1 / To + 1 / Te).sum()
            W = 1 / (1 / To + 1 / Te)
            tau2 = (W * (ro - m) * (re - m)).sum() / W.sum()
            out[(name, p)] = ((sigma2 / tau2) if tau2 > 0 else 1e6, m, int(ok.sum()))
    return out


def apply_eb(F, d, kmap, mult):
    F = F.copy()
    for name, (_, _, nc, dc, feat, _) in STATS.items():
        k = d.position.map({p: kmap[(name, p)][0] * mult for p in POS}).to_numpy()
        m = d.position.map({p: kmap[(name, p)][1] for p in POS}).to_numpy()
        post = (d[nc] + k * m) / (d[dc] + k)
        F[feat] = np.log(np.clip(post, 1e-4, None))
    return F


def main():
    pd.set_option("display.width", 250)
    d = player_seasons()
    H = split_half_table(d)
    print(f"player-season split-half pairs: {len(H):,}")
    df = pd.read_parquet(OUT / "scorer_assist_dataset.parquet")
    df = df[df.season <= "2024-25"].reset_index(drop=True)
    print("\nESTIMATED PRIOR STRENGTH k (exposure units), by fold (training seasons only)")
    kfold = {}
    for ts in TEST:
        kfold[ts] = estimate_k(H, int(ts[:4]) - 1)
    tab = pd.DataFrame({ts: {f"{n} {p}": v[0] for (n, p), v in kfold[ts].items()} for ts in TEST})
    print(tab.round(1).to_string())
    print("\nfixed-shrinkage pseudo-counts used by H0 for comparison: shots_fr 27.5 team shots, quality 10 shots, "
          "xa_fr 3.0 xG")
    mults = (0.5, 1.0, 2.0)
    for target, fac in (("goals_scored", FAC_G), ("assists", FAC_A)):
        names = ["H0 fixed shrinkage (tuned)"] + [f"EB split-half k x{m}" for m in mults]
        preds = {n: [] for n in names}
        for ts in TEST:
            tr, te = df[df.season < ts], df[df.season == ts]
            pr = M.prior_table(tr)
            Ftr0, Fte0 = M.build(tr, pr, 900.0, 0.25), M.build(te, pr, 900.0, 0.25)
            etr = (tr.minutes / 90).to_numpy()
            y = tr[target].to_numpy(float)
            cols = BASE + fac + TAIL
            preds[names[0]].append(pd.Series(M.fit_rate(cols, Ftr0, y, etr, Fte0), index=te.index))
            for mult, n in zip(mults, names[1:]):
                Ftr, Fte = apply_eb(Ftr0, tr, kfold[ts], mult), apply_eb(Fte0, te, kfold[ts], mult)
                preds[n].append(pd.Series(M.fit_rate(cols, Ftr, y, etr, Fte), index=te.index))
        test = df[df.season.isin(TEST)].copy()
        for k, v in preds.items():
            test[f"mu::{k}"] = pd.concat(v).reindex(test.index) * (test.minutes / 90)
        print(f"\n{target} (reference = fixed shrinkage; negative = better)")
        print(M.compare(target, test, names, names[0]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
