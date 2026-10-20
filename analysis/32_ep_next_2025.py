#!/usr/bin/env python3
"""Our full model against FPL's own pre-deadline projection (ep_next) over a past season (default 2025-26).

FPL: ep_next from the last copy of the FPL API committed before each deadline (pipeline/11_fetch_ep_next_archive.py;
0.4-10 hours before the deadline in 2025-26). Rounded by FPL to 0.1.
OURS: the full expectation P(start) x xP|start + P(sub) x xP|sub (analysis/31_full_xp.py with the season's sealed
model, availability from analysis/30_availability.py); a player outside the universe (no minutes in his previous
10 fixtures) is predicted 0, as on the site. Double gameweeks summed, blank gameweeks 0.
ROWS: every player registered in FPL's pre-deadline copy, played or not (0 points if not): the like-for-like
comparison, since ep_next also prices availability. The same metrics as the live track record (29_track_record.py).
UNCERTAINTY: the difference between the two methods (ours minus FPL) with a 95% interval from resampling whole
gameweeks (2,000 draws), because errors within a gameweek are correlated.
Output: docs/EP_NEXT_<season>.md (private: not exported to the website).
Options: --season 2025-26.
"""
import glob
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT, DOCS = ROOT / "data" / "analysis", ROOT / "docs"


def load_mod(name, path):
    sp = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


TR = load_mod("tr29", "analysis/29_track_record.py")
METHODS = {"ours (full model)": "ours", "FPL ep_next": "fpl"}


def frame(season):
    snaps = []
    for f in sorted(glob.glob(str(ROOT / "data" / "raw" / "fpl_core_insights" / season / "gw*.csv"))):
        d = pd.read_csv(f, usecols=["id", "ep_next"]).rename(columns={"id": "element", "ep_next": "fpl"})
        snaps.append(d.assign(gw=int(re.search(r"gw(\d+)_", Path(f).name).group(1))))
    E = pd.concat(snaps, ignore_index=True)
    F = pd.read_parquet(OUT / "full_xp" / f"full_xp_{season}.parquet")
    A = pd.read_parquet(OUT / "availability_predictions.parquet", columns=["season", "element", "fixture", "p_start", "p_sub"])
    F = F.merge(A, on=["season", "element", "fixture"], how="left", validate="one_to_one")
    assert F.p_start.notna().all()
    F["ours"] = F.p_start * F.xp_start + F.p_sub * F.xp_sub
    ours = F.groupby(["round", "element"]).ours.sum()
    m = pd.read_parquet(OUT / "master.parquet", columns=["season", "round", "element", "total_points", "minutes", "position"])
    m = m[m.season == season]
    act = m.groupby(["round", "element"]).agg(actual=("total_points", "sum"), minutes=("minutes", "sum"))
    pos = m.groupby("element").position.last()
    E = E.join(ours.rename_axis(["gw", "element"]), on=["gw", "element"]).join(
        act.rename_axis(["gw", "element"]), on=["gw", "element"])
    E["ours"] = E.ours.fillna(0.0)
    E["actual"] = E.actual.fillna(0.0)
    E["minutes"] = E.minutes.fillna(0.0)
    E["position"] = E.element.map(pos)
    return E


def boot_diff(E, n=2000):
    """95% interval of (ours - FPL) for MAE, RMSE and within-gameweek rank correlation, resampling gameweeks."""
    gws = E.gw.unique()
    by = {g: d for g, d in E.groupby("gw")}
    rng = np.random.default_rng(7)
    out = {k: [] for k in ("mae", "rmse", "rank")}
    for _ in range(n):
        S = pd.concat([by[g] for g in rng.choice(gws, len(gws))], ignore_index=True)
        a, b = S.actual - S.ours, S.actual - S.fpl
        out["mae"].append(a.abs().mean() - b.abs().mean())
        out["rmse"].append(np.sqrt((a ** 2).mean()) - np.sqrt((b ** 2).mean()))
    rk = E.groupby("gw").apply(lambda d: pd.Series({
        "o": d.ours.rank().corr(d.actual.rank()), "f": d.fpl.rank().corr(d.actual.rank())}))
    for _ in range(n):
        s = rk.loc[rng.choice(rk.index, len(rk))]
        out["rank"].append(s.o.mean() - s.f.mean())
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}, rk


def table(E):
    return pd.DataFrame({name: TR.metrics(E, col) for name, col in METHODS.items()}).T


def main():
    season = sys.argv[sys.argv.index("--season") + 1] if "--season" in sys.argv else "2025-26"
    E = frame(season)
    ci, rk = boot_diff(E)
    point = {k: TR.metrics(E, "ours")[k] - TR.metrics(E, "fpl")[k] for k in ("mae", "rmse", "rank")}
    L = [f"# Our full model vs FPL's ep_next, {season} (private)\n",
         "Script: analysis/32_ep_next_2025.py. FPL's projection is the pre-deadline ep_next archived by "
         "olbauday/FPL-Core-Insights (pipeline/11_fetch_ep_next_archive.py). Ours is the sealed 2025-26 model "
         "(fitted on 2016-17..2024-25) with P(start) and P(sub): the full expectation, as on the site.\n",
         f"Rows: {len(E):,} player-gameweeks over {E.gw.nunique()} gameweeks, every player registered at the "
         f"deadline (played: {(E.minutes > 0).sum():,}; 0 points if not).\n",
         "Metrics as in the live track record: bias (predicted minus actual), MAE, RMSE, mean within-gameweek "
         "Spearman rank correlation, mean actual points of each method's top 10 per gameweek, actual/predicted.\n",
         "## All registered players\n", table(E).to_markdown(), "",
         "Difference ours minus FPL (negative MAE/RMSE = ours more accurate; positive rank = ours ranks better), "
         "95% interval from resampling gameweeks:\n",
         pd.DataFrame({"difference": point, "95% low": {k: v[0] for k, v in ci.items()},
                       "95% high": {k: v[1] for k, v in ci.items()}}).round(4).to_markdown(), "",
         f"Gameweeks where our rank correlation is higher: {(rk.o > rk.f).sum()} of {len(rk)}.\n",
         "## By position\n"]
    for p, g in E.dropna(subset=["position"]).groupby("position"):
        L += [f"**{p}** ({len(g):,} rows)\n", table(g).to_markdown(), ""]
    L += ["## By part of the season\n"]
    for lab, g in E.groupby(pd.cut(E.gw, [0, 10, 19, 29, 38], labels=["GW1-10", "GW11-19", "GW20-29", "GW30-38"]),
                            observed=True):
        L += [f"**{lab}**\n", table(g).to_markdown(), ""]
    L += ["## Players either method expects to matter (ours or FPL >= 1 point)\n",
          "Removes the many fringe players both methods put near 0, which flatter MAE.\n"]
    S = E[(E.ours >= 1) | (E.fpl >= 1)]
    L += [f"{len(S):,} rows.\n", table(S).to_markdown(), ""]
    L += ["## Caveats\n",
          "- 2025-26 is our sealed hold-out season and has been looked at before (docs/XP_EVALUATION_sealed.md), "
          "though the model was not tuned on it.\n",
          "- FPL's copy is 0.4-10 hours before the deadline; ours uses deadline information but no team news "
          "(no availability flags exist for past seasons in our features), so FPL has an information advantage "
          "on injuries and suspensions.\n",
          "- Players outside our universe are predicted 0 by us; FPL gives some of them a positive ep_next.\n"]
    (DOCS / f"EP_NEXT_{season}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
