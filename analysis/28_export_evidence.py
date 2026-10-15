#!/usr/bin/env python3
"""Evidence for the website's method pages: the validation results of the xP model as chart data.

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


def main():
    doc = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sets": {tag: one_set(tag) for tag in SETS}, "horizon": horizon()}
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "evidence.json").write_text(json.dumps(doc, separators=(",", ":")))
    for tag, s in doc["sets"].items():
        a = s["accuracy"]
        print(f"{tag}: {s['starts']:,} starts; model RMSE {a['model']['rmse']} rank {a['model']['rank']} top10 {a['model']['top10']}; "
              f"pps5 RMSE {a['pps5']['rmse']} rank {a['pps5']['rank']} top10 {a['pps5']['top10']}; all {s['high_end']['all']['ratio']} "
              f"top10 {s['high_end']['top10']['ratio']}")


if __name__ == "__main__":
    main()
