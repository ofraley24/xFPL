#!/usr/bin/env python3
"""Evidence for the website's method pages: the validation results of the xP model as chart data.

Also the availability model (30_availability.py) and the full expectation (31_full_xp.py).
Recomputed from the saved walk-forward predictions with the evaluation's own definitions (15_evaluate_xp.py:
fixture-cluster bootstrap intervals, the points-per-start baseline) and the horizon backtest
(25_score_horizons.py). Sets: development 2019-20..2024-25, 2025-26 (sealed), 2026-27 GW1-6 (live).
Output: data/forecasts/evidence.json. Static: rerun only when the model or its evaluation changes.
"""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
DST = ROOT / "data" / "forecasts"


def load_mod(name, path):
    sp = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(sp)
    sys.argv = [str(path)]
    sp.loader.exec_module(m)
    return m


E15 = load_mod("e15", "analysis/15_evaluate_xp.py")
S25 = load_mod("s25", "analysis/25_score_horizons.py")
SETS = {"dev": "Development 2019-20 to 2024-25", "sealed": "2025-26 (held out)", "live": "2026-27 GW1-6"}
r = lambda x, d=3: None if x is None or not np.isfinite(x) else round(float(x), d)


def with_baselines(d):
    m = pd.read_parquet(OUT / "master.parquet", columns=["season", "fixture", "element", "code", "kickoff_time",
                                                           "total_points", "started"]).sort_values(["code", "kickoff_time"])
    m["form5"] = m.groupby("code").total_points.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    d = d.merge(m[["season", "fixture", "element", "form5"]], on=["season", "fixture", "element"], how="left")
    st = m[m.started == 1].copy()
    st["pps5"] = st.groupby("code").total_points.transform(lambda x: x.rolling(5, min_periods=1).mean())
    q = d[["season", "fixture", "element", "code", "kickoff_time"]].copy()
    q["kickoff_time"] = pd.to_datetime(q.kickoff_time)
    st["kickoff_time"] = pd.to_datetime(st.kickoff_time)
    q = pd.merge_asof(q.sort_values("kickoff_time"), st[["code", "kickoff_time", "pps5"]].sort_values("kickoff_time"),
                      on="kickoff_time", by="code", direction="backward", allow_exact_matches=False)
    d = d.merge(q[["season", "fixture", "element", "pps5"]], on=["season", "fixture", "element"], how="left")
    d["pps5"] = d.pps5.fillna(d.pps5.median())
    d["form5"] = d.form5.fillna(0)
    return d


def acc(sub, col):
    e = sub.total_points - sub[col]
    gw = sub.groupby(["season", "round"]).apply(lambda g: spearmanr(g[col], g.total_points).correlation)
    top = sub.groupby(["season", "round"]).apply(lambda g: g.nlargest(10, col).total_points.mean())
    return {"bias": r(-e.mean()), "rmse": r(np.sqrt((e ** 2).mean())), "mae": r(e.abs().mean()),
            "rank": r(gw.mean()), "top10": r(top.mean(), 2)}


def ratio_ci(g, obs, pred):
    """actual/predicted with a fixture-cluster bootstrap 95% interval."""
    lo, hi = E15.cluster_ci(g.assign(_r=g[obs] / g[pred].mean()), "_r")
    return {"ratio": r(g[obs].mean() / g[pred].mean()), "lo": r(lo), "hi": r(hi), "n": int(len(g))}


def one_set(tag):
    d = pd.read_parquet(OUT / f"xp_predictions_{tag}.parquet")
    d["fx"] = d.season + "_" + d.fixture.astype(str)
    d = with_baselines(d)
    out = {"label": SETS[tag], "starts": int(len(d)), "seasons": sorted(d.season.unique().tolist())}
    out["accuracy"] = {"model": acc(d, "xp"), "pps5": acc(d, "pps5"), "form5": acc(d, "form5")}
    # reliability: 25 quantile bins
    d["_b"] = pd.qcut(d.xp.rank(method="first"), 25, labels=False)
    rel = []
    for _, g in d.groupby("_b"):
        lo, hi = E15.cluster_ci(g, "total_points")
        rel.append({"pred": r(g.xp.mean()), "obs": r(g.total_points.mean()), "lo": r(lo), "hi": r(hi), "n": int(len(g))})
    out["reliability"] = rel
    # randomised PIT
    rng = np.random.default_rng(20261015)
    pit = d.cdf_lo + rng.random(len(d)) * (d.cdf_hi - d.cdf_lo)
    out["pit"] = {"hist": [r(x, 4) for x in np.histogram(pit, bins=10, range=(0, 1))[0] / len(d) * 10],
                  "c50": r(((pit > .25) & (pit < .75)).mean()), "c80": r(((pit > .1) & (pit < .9)).mean())}
    # components: mean observed / mean predicted per start
    dcd = (d.season >= "2025-26") & (d.position != "GKP")
    hit = (d.defensive_contribution.fillna(0) >= np.where(d.position == "DEF", 10, 12)).astype(float)
    comps = [("Minutes", "e_minutes", d.minutes), ("Appearance points", "e_app", (d.minutes > 0).astype(int) + (d.minutes >= 60).astype(int)),
             ("Goals", "e_goals", d.goals_scored), ("Assists", "e_assists", d.assists), ("Clean sheets", "e_cs", d.clean_sheets),
             ("Bonus", "e_bonus", d.bonus), ("Yellow cards", "e_yellow", d.yellow_cards)]
    out["components"] = [{"name": n, "pred": r(d[c].mean(), 4), "obs": r(o.mean(), 4), "ratio": r(o.mean() / d[c].mean())} for n, c, o in comps]
    gk = d.position == "GKP"
    out["components"].append({"name": "Save points (GK)", "pred": r(d.e_saves_pts[gk].mean(), 4), "obs": r((d.saves[gk] // 3).mean(), 4),
                              "ratio": r((d.saves[gk] // 3).mean() / d.e_saves_pts[gk].mean())})
    if dcd.any():
        out["components"].append({"name": "Defensive contribution", "pred": r(d.e_dc[dcd].mean(), 4), "obs": r(hit[dcd].mean(), 4),
                                  "ratio": r(hit[dcd].mean() / d.e_dc[dcd].mean())})
    # the high end: actual/predicted for each gameweek's top k
    gw = d.groupby(["season", "round"])
    tops = {}
    for k in (5, 10, 20, 40):
        t = gw.apply(lambda g: g.nlargest(k, "xp")).reset_index(drop=True)
        tops[f"top{k}"] = ratio_ci(t, "total_points", "xp")
    tops["all"] = ratio_ci(d, "total_points", "xp")
    out["high_end"] = tops
    return out


def horizon():
    d = pd.read_parquet(OUT / "xp_horizon.parquet")
    d = d[d.season.between("2019-20", "2024-25")]
    bal = S25.balanced(d)
    rows = bal.groupby("horizon").apply(S25.row_metrics)
    win = pd.DataFrame({L: S25.windows(d, L) for L in range(1, 9)}).T
    return [{"h": int(h), "bias": r(rows.loc[h, "bias"]), "mae": r(rows.loc[h, "MAE"]), "rank": r(rows.loc[h, "rank_corr_gw"]),
             "cover80": r(win.loc[h, "cover80"])} for h in rows.index]


def log_loss3(cls, P):
    """Mean multinomial log loss; cls 0 none, 1 start, 2 sub; P columns in that order."""
    p = np.clip(P[np.arange(len(cls)), cls], 1e-12, 1)
    return -np.log(p).mean()


def deciles(p, y):
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).to_numpy()
    return [{"pred": r(p[q == k].mean(), 4), "obs": r(y[q == k].mean(), 4), "n": int((q == k).sum())} for k in range(10)]


def availability():
    """Availability model (30_availability.py): walk-forward accuracy against the recent-shares baseline,
    reliability of P(start) and P(sub), and horizons (features frozen at the deadline, recalibrated)."""
    a = pd.read_parquet(OUT / "availability_predictions.parquet")
    seasons = []
    for s, g in a.groupby("season"):
        c = g.cls.to_numpy()
        seasons.append({"season": s, "n": int(len(g)),
                        "set": "dev" if s <= "2024-25" else "sealed" if s == "2025-26" else "live",
                        "ll_model": r(log_loss3(c, g[["p_none", "p_start", "p_sub"]].to_numpy()), 4),
                        "ll_base": r(log_loss3(c, g[["b_none", "b_start", "b_sub"]].to_numpy()), 4),
                        "obs_start": r((c == 1).mean(), 4), "pred_start": r(g.p_start.mean(), 4),
                        "obs_sub": r((c == 2).mean(), 4), "pred_sub": r(g.p_sub.mean(), 4)})
    dv = a[a.season.between("2019-20", "2024-25")]
    h = pd.read_parquet(OUT / "availability_horizon_predictions.parquet")
    h = h[h.season.between("2020-21", "2024-25")]
    hz = []
    for k, g in h.groupby("horizon"):
        y = g.y.to_numpy()
        hz.append({"h": int(k), "n": int(len(g)), "obs_start": r((y == 1).mean(), 4), "pred_start": r(g.r_start.mean(), 4),
                   "ll_raw": r(log_loss3(y, g[["p_none", "p_start", "p_sub"]].to_numpy()), 4),
                   "ll_recal": r(log_loss3(y, g[["r_none", "r_start", "r_sub"]].to_numpy()), 4)})
    return {"rows": int(len(dv)), "seasons": seasons,
            "rel_start": deciles(dv.p_start.to_numpy(), (dv.cls == 1).to_numpy(float)),
            "rel_sub": deciles(dv.p_sub.to_numpy(), (dv.cls == 2).to_numpy(float)), "horizon": hz}


def full_model():
    """The full expectation P(start) x xP|start + P(sub) x xP|sub (31_full_xp.py), development seasons, every
    player with minutes in his previous 10 fixtures. No FPL availability flags exist for past seasons, so the
    backtest has none; the live forecast applies them as a heuristic."""
    F = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "full_xp").glob("full_xp_*.parquet"))], ignore_index=True)
    A = pd.read_parquet(OUT / "availability_predictions.parquet",
                        columns=["season", "element", "fixture", "p_none", "p_start", "p_sub", "cls"])
    F = F.merge(A, on=["season", "element", "fixture"], how="left", validate="one_to_one")
    F["xp_full"] = F.p_start * F.xp_start + F.p_sub * F.xp_sub
    F["fx"] = F.season + "_" + F.fixture.astype(str)
    ko = pd.read_parquet(OUT / "master.parquet", columns=["season", "fixture", "element", "kickoff_time"])
    F = with_baselines(F.merge(ko, on=["season", "fixture", "element"], how="left", validate="one_to_one"))
    y = F.total_points.to_numpy(float)
    by_season = [{"season": s, **ratio_ci(g, "total_points", "xp_full")} for s, g in F.groupby("season")]
    pos = [{"pos": p, **ratio_ci(g, "total_points", "xp_full")} for p, g in F.groupby("position")]
    q = pd.qcut(F.p_start.rank(method="first"), 10, labels=False)
    dec = [{"p_start": r(g.p_start.mean()), "pred": r(g.xp_full.mean()), "obs": r(g.total_points.mean()), "n": int(len(g))}
           for _, g in F.groupby(q)]
    rng = np.random.default_rng(20261018)
    lo = F.p_start * F.cdf_lo_start + F.p_sub * F.cdf_lo_sub + F.p_none * (y > 0)
    hi = F.p_start * F.cdf_hi_start + F.p_sub * F.cdf_hi_sub + F.p_none * (y >= 0)
    pit = (lo + rng.random(len(F)) * (hi - lo)).to_numpy()
    S = F[F.cls == 2]
    sub = {"n": int(len(S)), "ratio": r(S.total_points.sum() / S.xp_sub.sum()),
           "min_pred": r(S.e_minutes_sub.mean(), 1), "min_obs": r(S.minutes.mean(), 1)}
    sub_by_season = [{"season": s, "min_pred": r(g.e_minutes_sub.mean(), 1), "min_obs": r(g.minutes.mean(), 1)}
                     for s, g in S.groupby("season")]
    return {"rows": int(len(F)), "seasons": sorted(F.season.unique().tolist()),
            "overall": ratio_ci(F, "total_points", "xp_full"), "by_season": by_season, "by_position": pos,
            "deciles": dec, "accuracy": {"model": acc(F, "xp_full"), "form5": acc(F, "form5")},
            "pit": {"hist": [r(x, 4) for x in np.histogram(pit, bins=10, range=(0, 1))[0] / len(F) * 10],
                    "c50": r(((pit > .25) & (pit < .75)).mean()), "c80": r(((pit > .1) & (pit < .9)).mean())},
            "sub": sub, "sub_by_season": sub_by_season}


def main():
    doc = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sets": {tag: one_set(tag) for tag in SETS}, "horizon": horizon(),
           "availability": availability(), "full": full_model()}
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "evidence.json").write_text(json.dumps(doc, separators=(",", ":")))
    for tag, s in doc["sets"].items():
        a = s["accuracy"]
        print(f"{tag}: {s['starts']:,} starts; model RMSE {a['model']['rmse']} rank {a['model']['rank']} top10 {a['model']['top10']}; "
              f"pps5 RMSE {a['pps5']['rmse']} rank {a['pps5']['rank']} top10 {a['pps5']['top10']}; all {s['high_end']['all']['ratio']} "
              f"top10 {s['high_end']['top10']['ratio']}")
    f = doc["full"]
    print(f"full: {f['rows']:,} rows; actual/predicted {f['overall']['ratio']} ({f['overall']['lo']}-{f['overall']['hi']}); "
          f"rank {f['accuracy']['model']['rank']} vs form {f['accuracy']['form5']['rank']}; PIT {f['pit']['c50']} / {f['pit']['c80']}")


if __name__ == "__main__":
    main()
