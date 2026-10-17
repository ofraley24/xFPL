#!/usr/bin/env python3
"""Points given a substitute appearance and the full expectation (docs/AVAILABILITY_PLAN.md, parts B and C).

  E[points] = P(start) x E[points | start] + P(sub) x E[points | sub]        (no appearance scores 0)

ROWS. Development seasons, players with minutes in their previous 10 fixtures (the universe; 97% of
appearances). Each row's fold model (13_xp_model.py, data/models/xp_model_dev_<season>.pkl) is used unchanged.
MATCHES. As in the live forecast (26_forecast.py): each team plays a probable XI ranked by starts in the last
10, recent minutes and price (deadline information only; no availability flags exist for past seasons).
  start run: the XI plays one match; every other player is simulated in his own copy of it, replacing the
             lowest-ranked XI player of his position.
  sub run:   EVERY player gets his own copy (XI players included), and only he has substitute minutes, so the
             21 others play as starters and the bonus competition stays realistic.
SUBSTITUTE MINUTES. Bucket probabilities from training substitute appearances of the same position, from
training seasons of the test season's five-substitute era when one exists, else all training seasons.
Within-bucket minutes are the model's own (start) values.
DRAWS. J = 10 parameter draws x R = 25 (live: 40 x 25); means are unaffected.
Output: data/analysis/full_xp/full_xp_<season>.parquet (one row per universe player-fixture: xP and CDF at
the outcome for start and sub). --report scores B and C into docs/FULL_XP_RESULTS.md.
Options: --seasons a,b (default the development seasons), --report.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT, DOCS = ROOT / "data" / "analysis", ROOT / "docs"
FX = OUT / "full_xp"
DEV = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def load_mod(name, path):
    sp = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


X13 = load_mod("x13", "analysis/13_xp_model.py")
X13.J = 10
RES = ["xp", "cdf_lo", "cdf_hi", "e_minutes", "e_bonus", "e_goals", "e_assists", "e_cs"]


def probable_xi(g):
    """Indices of each side's probable XI (1 GKP + 10 outfield) and the rest, deadline information only."""
    sides = {}
    for home, s in g.groupby("was_home"):
        s = s.assign(_st=s.st_n10.fillna(0), _m5=s.m5.fillna(0)).sort_values(
            ["_st", "_m5", "value", "element"], ascending=[False, False, False, True])
        xi = list(s.index[s.position == "GKP"][:1]) + list(s.index[s.position != "GKP"][:10])
        sides[home] = (xi, [i for i in s.index if i not in xi])
    return sides


def copies(d, every_player):
    """Simulation rows: per fixture the XI match and one copy per focal player (see header)."""
    idx, simfx, focal = [], [], []
    for f, g in d.groupby("fixture", sort=True):
        sides = probable_xi(g)
        xi_all = [i for v in sides.values() for i in v[0]]
        k = 0
        if not every_player:
            idx += xi_all; simfx += [f * 1000] * len(xi_all); focal += xi_all
            k = 1
        for home, (xi, bench) in sides.items():
            other = [i for h, v in sides.items() if h != home for i in v[0]]
            for p in (xi + bench) if every_player else bench:
                if p in xi:
                    rows = xi + other
                else:
                    same = [i for i in xi if d.position[i] == d.position[p]]
                    pool = same or [i for i in xi if d.position[i] != "GKP"]
                    rows = [i for i in xi if i != pool[-1]] + [p] + other
                idx += rows; simfx += [f * 1000 + k] * len(rows); focal += [p] * len(rows)
                k += 1
    S = d.loc[idx].reset_index().rename(columns={"index": "row"})
    S["sim_fixture"], S["focal_row"] = simfx, focal
    S["is_focal"] = S.row == S.focal_row
    return S


def sub_bucket_probs(tr, ts, te_era):
    sub = tr[(tr.started == 0) & (tr.minutes > 0)]
    same = sub[sub.five_subs == te_era]
    sub = same if len(same) else sub
    b = np.digitize(sub.minutes.to_numpy(), X13.EDGES)
    out = {}
    for p in X13.POS:
        bb = b[(sub.position == p).to_numpy()]
        pr = np.bincount(bb, minlength=X13.K).astype(float)
        pr[0] = 0.0
        out[p] = pr / pr.sum() if pr.sum() else np.r_[0, np.ones(X13.K - 1) / (X13.K - 1)]
    return out


def run_season(ts, full):
    M = X13.load_model("dev", ts)
    d = full[(full.season == ts) & full.universe].reset_index(drop=True)
    tr = full[full.season < ts]
    era = d.five_subs.mode().iloc[0]
    sp = sub_bucket_probs(tr, ts, era)
    out = d[["season", "round", "element", "code", "position", "fixture", "was_home", "minutes", "started",
             "total_points"]].copy()
    for run, every in (("start", False), ("sub", True)):
        S = copies(d, every)
        te = S.drop(columns=["fixture"]).rename(columns={"sim_fixture": "fixture"})
        Pmin = X13.predict_minutes(M["members"], te, M["mfeats"])
        if run == "sub":
            foc = S.is_focal.to_numpy()
            Pmin[:, foc, :] = np.stack([sp[p] for p in S.position[foc]])[None, :, :]
        seed = X13.SEED + 7 * int(ts[:4]) + (5000 if run == "start" else 6000)
        res, _, _ = X13.predict_rows(M, te, np.random.default_rng(seed), Pmin)
        r = pd.concat([S[["row", "is_focal"]], res[RES]], axis=1)
        r = r[r.is_focal].set_index("row").sort_index()
        assert r.index.is_unique and len(r) == len(d), f"{run}: {len(r)} focal rows for {len(d)} players"
        out = out.join(r[RES].add_suffix(f"_{run}"))
        print(f"  {ts} {run}: {len(te):,} simulated rows, mean xP {r.xp.mean():.3f}", flush=True)
    FX.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FX / f"full_xp_{ts}.parquet", index=False)


def form_baseline(master):
    """Mean points over the player's previous 5 fixtures (zeros included), as of the row."""
    m = master.sort_values(["code", "kickoff_time", "fixture"])
    f = m.groupby("code").total_points.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return m.assign(form5=f.fillna(0))[["season", "element", "fixture", "form5"]]


def report():
    F = pd.concat([pd.read_parquet(p) for p in sorted(FX.glob("full_xp_*.parquet"))], ignore_index=True)
    A = pd.read_parquet(OUT / "availability_predictions.parquet",
                        columns=["season", "element", "fixture", "p_none", "p_start", "p_sub", "cls"])
    F = F.merge(A, on=["season", "element", "fixture"], how="left", validate="one_to_one")
    master = pd.read_parquet(OUT / "master.parquet")
    F = F.merge(form_baseline(master), on=["season", "element", "fixture"], how="left")
    assert F.p_start.notna().all()
    F["xp_full"] = F.p_start * F.xp_start + F.p_sub * F.xp_sub
    y = F.total_points.to_numpy(float)
    lines = ["# Full xP and points given a substitute appearance: results\n",
             "Plan (written before any run): docs/AVAILABILITY_PLAN.md, parts B and C. Script: analysis/31_full_xp.py.\n"]
    # ---- B: substitute appearances
    S = F[F.cls == 2]
    b = S.groupby("position").apply(lambda g: pd.Series({
        "appearances": len(g), "actual pts": g.total_points.mean(), "xP | sub": g.xp_sub.mean(),
        "actual/predicted": g.total_points.sum() / g.xp_sub.sum(),
        "actual min": g.minutes.mean(), "predicted min": g.e_minutes_sub.mean()}))
    allb = pd.Series({"appearances": len(S), "actual pts": S.total_points.mean(), "xP | sub": S.xp_sub.mean(),
                      "actual/predicted": S.total_points.sum() / S.xp_sub.sum(), "actual min": S.minutes.mean(),
                      "predicted min": S.e_minutes_sub.mean()}, name="all")
    b = pd.concat([b, allb.to_frame().T])
    flag = (b["actual/predicted"] - 1).abs() > 0.15
    lines += ["## B. Points given a substitute appearance (development substitute appearances)\n",
              b.round(3).to_markdown(), "",
              ("Flag: outside +-15% for " + ", ".join(b.index[flag]) + ".\n") if flag.any() else
              "Within +-15% for every position.\n"]
    st = F[F.cls == 1]
    lines += [f"For comparison, starters: actual/predicted {st.total_points.sum() / st.xp_start.sum():.3f} "
              f"({len(st):,} starts).\n"]
    # ---- C: full expectation over all universe rows
    ratio = y.sum() / F.xp_full.sum()
    lines += ["## C. The full expectation (all players with minutes in their previous 10 fixtures)\n",
              f"**Sanity rule (overall actual/predicted within +-5%): {'PASS' if abs(ratio - 1) <= 0.05 else 'FAIL'}** "
              f"({ratio:.3f}; {len(F):,} player-fixtures).\n"]
    q = pd.qcut(F.p_start.rank(method="first"), 10, labels=False)
    dec = F.groupby(q).agg(p_start=("p_start", "mean"), xp_full=("xp_full", "mean"), actual=("total_points", "mean"))
    dec["actual/predicted"] = dec.actual / dec.xp_full
    lines += ["By decile of P(start):\n", dec.round(3).T.to_markdown(), ""]
    pos = F.groupby("position").apply(lambda g: pd.Series({"rows": len(g), "xP full": g.xp_full.mean(),
                                                           "actual": g.total_points.mean(),
                                                           "actual/predicted": g.total_points.sum() / g.xp_full.sum()}))
    lines += ["By position:\n", pos.round(3).to_markdown(), ""]
    acc = []
    for name, col in (("full xP", "xp_full"), ("xP given a start", "xp_start"), ("form (last 5, zeros incl.)", "form5")):
        rho = F.groupby(["season", "round"]).apply(lambda g: spearmanr(g[col], g.total_points).correlation).mean()
        top = F[F.groupby(["season", "round"])[col].rank(ascending=False, method="first") <= 10].total_points.mean()
        acc.append({"predictor": name, "within-GW rank corr": rho,
                    "RMSE": np.sqrt(((F[col] - y) ** 2).mean()) if col != "xp_start" else np.nan,
                    "top-10 by GW: mean actual": top})
    lines += ["Ranking all players (every universe player-fixture, starters or not):\n",
              pd.DataFrame(acc).round(4).to_markdown(index=False), ""]
    # randomised PIT of the mixture
    u = np.random.default_rng(1).random(len(F))
    lo0, hi0 = (y > 0).astype(float), (y >= 0).astype(float)
    lo = F.p_start * F.cdf_lo_start + F.p_sub * F.cdf_lo_sub + F.p_none * lo0
    hi = F.p_start * F.cdf_hi_start + F.p_sub * F.cdf_hi_sub + F.p_none * hi0
    pit = lo + u * (hi - lo)
    lines += [f"Mixture distribution, randomised PIT: central 50% {((pit > .25) & (pit < .75)).mean():.3f}, "
              f"central 80% {((pit > .1) & (pit < .9)).mean():.3f} (nominal 0.50 / 0.80).\n"]
    ms = master[master.season.isin(F.season.unique())]
    outside = ms[~(ms.m10.fillna(0) > 0)]
    lines += [f"Players outside the universe (no minutes in their previous 10 fixtures) are predicted 0: "
              f"{len(outside):,} player-fixtures, {(outside.minutes > 0).sum():,} appearances, "
              f"{outside.total_points.sum() / ms.total_points.sum():.1%} of all points scored.\n"]
    (DOCS / "FULL_XP_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    args = sys.argv[1:]
    if "--report" in args:
        report()
        return
    seasons = args[args.index("--seasons") + 1].split(",") if "--seasons" in args else DEV
    full = X13.load("2024-25")
    for ts in seasons:
        run_season(ts, full)


if __name__ == "__main__":
    main()
