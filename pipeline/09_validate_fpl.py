#!/usr/bin/env python3
"""Validate the FPL tables and RECOVER each season's scoring rules from the data.

Why recover rules instead of writing them down: FPL's rules changed over time and
a hand-typed table cannot be audited. total_points is a deterministic function of
the event counts, so a regression of points on event features per season returns
the points-per-event actually applied, with an R^2 / exact-match rate that doubles
as a test of the data's integrity. The recovered 2026-27 coefficients are then
compared with the rules the API publishes (fpl_scoring_rules.csv).

Writes docs/FPL_QC.md and docs/FPL_RULES_RECOVERED.csv.

Features (per player-match, positions GKP/DEF/MID/FWD):
  short (1-59 min), long (>=60 min), goals_scored, assists, clean sheet & >=60 min,
  floor(goals_conceded/2), floor(saves/3), penalties_saved, penalties_missed,
  yellow, red, own goals, bonus, defensive_contribution >= 10 / >= 12.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402

FPL = DATA / "fpl"
POSITIONS = ["GKP", "DEF", "MID", "FWD"]
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))


def design(d):
    m = d.minutes.fillna(0)
    X = {"short_play": ((m >= 1) & (m < 60)).astype(float), "long_play": (m >= 60).astype(float)}
    for p in POSITIONS:
        ip = (d.position == p).astype(float)
        X[f"goal_{p}"] = d.goals_scored.fillna(0) * ip
        X[f"cs_{p}"] = d.clean_sheets.fillna(0) * (m >= 60) * ip
        X[f"gc2_{p}"] = np.floor(d.goals_conceded.fillna(0) / 2) * ip
        X[f"dc10_{p}"] = (d.defensive_contribution.fillna(0) >= 10).astype(float) * ip
        X[f"dc12_{p}"] = (d.defensive_contribution.fillna(0) >= 12).astype(float) * ip
    X["saves3_GKP"] = np.floor(d.saves.fillna(0) / 3) * (d.position == "GKP")
    for c, k in [("assists", "assist"), ("penalties_saved", "pen_save"),
                 ("penalties_missed", "pen_miss"), ("yellow_cards", "yellow"),
                 ("red_cards", "red"), ("own_goals", "own_goal"), ("bonus", "bonus")]:
        X[k] = d[c].fillna(0).astype(float)
    return pd.DataFrame(X)


def main():
    pm = pd.read_csv(FPL / "fpl_player_match.csv", low_memory=False)
    rules_api = pd.read_csv(FPL / "fpl_scoring_rules.csv")
    snap = pd.read_csv(FPL / "fpl_players_snapshot.csv", low_memory=False)

    check("F1 key (season,element,fixture,round) unique",
          not pm.duplicated(["season", "element", "fixture", "round"]).any())
    check("F1 every row has a persistent code", pm.code.notna().mean() > 0.999,
          f"{pm.code.notna().mean():.4%}")
    check("F1 every row has a position (managers labelled MNG)", pm.position.notna().all())
    check("F1 fixture_finished present for every row", pm.fixture_finished.notna().all())
    check("F1 minutes in [0,150]", pm.minutes.between(0, 150).all())
    for s, g in pm.groupby("season"):
        has_xg = g.expected_goals.notna().mean()
        want = s >= "2022-23"
        check(f"F2 {s} expected_goals coverage ({'expected' if want else 'n/a pre-2022-23'})",
              (has_xg > 0.95) if want else True, f"{has_xg:.1%}")
        if s >= "2025-26":
            check(f"F2 {s} defensive_contribution coverage", g.defensive_contribution.notna().mean() > 0.95,
                  f"{g.defensive_contribution.notna().mean():.1%}")
    cur = pm.season.max()
    live_all = pm[(pm.season == cur) & pm.source.str.startswith("fpl_api")]
    in_play = set(live_all[~live_all.fixture_finished].element)  # players in an unfinished fixture
    live = live_all[~live_all.element.isin(in_play)]
    print(f"(excluding {len(in_play)} players whose team has a fixture in play from the totals check)")
    if len(live):
        latest_snap = snap[snap.snapshot_utc == snap.snapshot_utc.max()].set_index("id")
        agg = live.groupby("element")[["total_points", "minutes", "goals_scored", "assists"]].sum()
        j = agg.join(latest_snap[["total_points", "minutes", "goals_scored", "assists"]],
                     rsuffix="_boot", how="inner")
        for c in ["total_points", "minutes", "goals_scored", "assists"]:
            check(f"F3 {cur} summed match history == bootstrap season total ({c})",
                  (j[c] == j[f"{c}_boot"]).mean() > 0.99,
                  f"{(j[c] == j[f'{c}_boot']).mean():.2%} of {len(j)} players match")
        # independent cross-check against the vaastav archive for GW1 where it has the season
        v = pd.read_csv(DATA / "raw/vaastav" / cur / "gws/merged_gw.csv", low_memory=False)
        v = v[v["round"] == 1].drop_duplicates(["element", "fixture"])
        l1 = live[live["round"] == 1].drop_duplicates(["element", "fixture"])
        jj = l1.merge(v, on=["element", "fixture"], suffixes=("", "_v"))
        for c in ["minutes", "total_points", "goals_scored", "bps", "expected_goals"]:
            a, b = jj[c].astype(float), jj[f"{c}_v"].astype(float)
            check(f"F4 live API vs vaastav GW1 agree ({c})", np.isclose(a, b, equal_nan=True).mean() > 0.99,
                  f"{np.isclose(a, b, equal_nan=True).mean():.2%} of {len(jj)}")

    # ----- recover scoring rules per season and test the points identity
    rec_rows, summary = [], []
    for s, g in pm.groupby("season"):
        g = g[g.total_points.notna() & g.position.isin(POSITIONS)]  # managers (MNG) excluded
        X = design(g)
        keep = [c for c in X.columns if X[c].abs().sum() > 0]
        X = X[keep]
        y = g.total_points.astype(float).to_numpy()
        beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
        pred = X.to_numpy() @ beta
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        exact = (np.round(pred) == y).mean()
        summary.append({"season": s, "rows": len(g), "R2": round(r2, 5), "exact_match": round(exact, 4)})
        for c, b in zip(keep, beta):
            rec_rows.append({"season": s, "feature": c, "points": round(b, 2)})
    rec = pd.DataFrame(rec_rows)
    wide = rec.pivot(index="feature", columns="season", values="points")
    wide.to_csv(DOCS / "FPL_RULES_RECOVERED.csv")
    sm = pd.DataFrame(summary)
    for _, r in sm.iterrows():
        check(f"F5 {r.season} points identity: R2={r.R2}, exact={r.exact_match:.1%}", r.R2 > 0.99)

    # compare recovered current-season coefficients with the API's published rules
    if cur in wide.columns and len(rules_api):
        pub = rules_api[rules_api.snapshot_utc == rules_api.snapshot_utc.max()]
        pubd = {(r.stat, r.position): r.points for r in pub.itertuples()}
        pairs = [(f"goal_{p}", pubd.get(("goals_scored", p))) for p in POSITIONS] + \
                [(f"cs_{p}", pubd.get(("clean_sheets", p))) for p in POSITIONS] + \
                [("assist", pubd.get(("assists", "ALL"))), ("pen_save", pubd.get(("penalties_saved", "ALL"))),
                 ("pen_miss", pubd.get(("penalties_missed", "ALL"))), ("red", pubd.get(("red_cards", "ALL"))),
                 ("yellow", pubd.get(("yellow_cards", "ALL"))), ("own_goal", pubd.get(("own_goals", "ALL")))]
        for f, p in pairs:
            if f in wide.index and p is not None and not np.isnan(wide.loc[f, cur]):
                check(f"F6 {cur} recovered '{f}' == API published ({p})",
                      abs(wide.loc[f, cur] - p) < 0.25, f"recovered {wide.loc[f, cur]}")

    npass = sum(ok for _, ok, _ in results)
    with open(DOCS / "FPL_QC.md", "w") as f:
        f.write(f"# FPL data QC ({date.today()})\n\n{npass}/{len(results)} checks passed.\n\n")
        f.write("## Points identity by season\n\n" + sm.to_markdown(index=False) + "\n\n")
        f.write("## Recovered points per event (regression on data; see FPL_RULES_RECOVERED.csv)\n\n"
                + wide.to_markdown() + "\n\n## Checks\n\n| check | result | detail |\n|---|---|---|\n")
        for n, ok, d in results:
            f.write(f"| {n} | {'PASS' if ok else '**FAIL**'} | {d} |\n")
    print(f"\n{npass}/{len(results)} passed -> docs/FPL_QC.md")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
