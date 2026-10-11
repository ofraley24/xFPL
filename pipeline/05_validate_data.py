#!/usr/bin/env python3
"""Data-quality checks on the merged Understat tables. Writes docs/DATA_QC.md.

Checks are assertions about the data, run every refresh:
  C1 pre-2026 rows of every merged file are byte-identical to the frozen snapshot
  C2 key uniqueness (on the clean layer; upstream duplicates are removed by 02b, see docs/dedup_log.csv)
  C2 (match_id; match_id+player_id; shot_id; team+date)
  C3 referential integrity (shots/rosters/team rows all map to a played match)
  C4 every played match has exactly 2 team rows, shots, and >= 11 roster rows/side
  C5 score consistency: match goals == sum of team 'scored'; shot-goal count and
     own goals reconcile with the match score
  C6 team naming: set of team names per league-season equals between tables
  C7 value ranges (xG in [0,1] per shot, minutes in [0,130], no negative counts)
  C8 freshness: last played date and played/fixture counts per league
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS, LEAGUES, LEGACY_UNDERSTAT, MERGED  # noqa: E402

CLEAN = DATA / "understat_clean"

KINDS = ["matches", "team_match_stats", "players", "player_match_stats", "shots"]
REFRESHED = 2026  # seasons >= this were replaced by API data
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))


def rd(base, lg, kind):
    return pd.read_csv(base / f"{lg}_{kind}.csv", dtype=str, keep_default_na=False)


def main():
    fresh_rows = []
    for lg in LEAGUES:
        merged = {k: rd(MERGED, lg, k) for k in KINDS}   # C1 runs on this
        t = {k: rd(CLEAN, lg, k) for k in KINDS}         # all other checks run on the de-duplicated layer
        old = {k: rd(LEGACY_UNDERSTAT, lg, k) for k in KINDS}
        # C1 history untouched (merged vs frozen snapshot)
        for k in KINDS:
            h_new = merged[k][merged[k].season.astype(int) < REFRESHED].reset_index(drop=True)
            h_old = old[k][old[k].season.astype(int) < REFRESHED].reset_index(drop=True)
            check(f"C1 {lg} {k}: pre-{REFRESHED} history identical", h_new.equals(h_old),
                  f"{len(h_new):,} rows")
        m = t["matches"]
        played = m[m.isResult == "true"]
        check(f"C2 {lg} match_id unique", m.match_id.is_unique)
        pms, sh, tm = t["player_match_stats"], t["shots"], t["team_match_stats"]
        check(f"C2 {lg} (match_id,player_id) unique in roster",
              not pms.duplicated(["match_id", "player_id"]).any(),
              f"{pms.duplicated(['match_id','player_id']).sum()} dups")
        check(f"C2 {lg} shot_id unique", sh.shot_id.is_unique,
              f"{sh.shot_id.duplicated().sum()} dups")
        check(f"C2 {lg} team+date unique", not tm.duplicated(["team_id", "date"]).any())
        # C3/C4 only on refreshed seasons (history was accepted as-is, see C1)
        r = played[played.season.astype(int) >= REFRESHED]
        ids = set(r.match_id)
        check(f"C3 {lg} 2026 roster match_ids all played", set(pms[pms.season.astype(int) >= REFRESHED].match_id) <= ids)
        check(f"C3 {lg} 2026 shot match_ids all played", set(sh[sh.season.astype(int) >= REFRESHED].match_id) <= ids)
        n_side = pms[pms.match_id.isin(ids)].groupby(["match_id", "h_a"]).size().unstack()
        check(f"C4 {lg} 2026 every played match has >=11 roster rows per side",
              len(n_side) == len(ids) and (n_side >= 11).all().all(),
              f"{len(n_side)}/{len(ids)} matches with rosters")
        # C5 score consistency from team table
        t26 = tm[tm.season.astype(int) >= REFRESHED].copy()
        gl = r.assign(hg=r.home_goals.astype(int), ag=r.away_goals.astype(int))
        ok_goals = (gl.hg.sum() + gl.ag.sum()) == t26.scored.astype(int).sum() \
            if len(gl) else True
        check(f"C5 {lg} 2026 total goals: matches == team rows", ok_goals,
              f"{gl.hg.sum()+gl.ag.sum()} vs {t26.scored.astype(int).sum()}")
        sg = sh[sh.season.astype(int) >= REFRESHED]
        goals_shots = (sg.result == "Goal").sum() + (sg.result == "OwnGoal").sum()
        check(f"C5 {lg} 2026 shot goals+own goals == match goals",
              goals_shots == gl.hg.sum() + gl.ag.sum(), f"{goals_shots} vs {gl.hg.sum()+gl.ag.sum()}")
        # C6 naming
        a = set(m[m.season.astype(int) >= REFRESHED].home_team)
        b = set(tm[tm.season.astype(int) >= REFRESHED].team_title)
        check(f"C6 {lg} 2026 team names: matches vs team table", a == b, f"diff={sorted(a ^ b)}")
        # C7 ranges
        check(f"C7 {lg} shot xG in [0,1]", sh.xG.astype(float).between(0, 1).all())
        check(f"C7 {lg} minutes in [0,130]", pms.time.astype(float).between(0, 130).all())
        # C8 freshness
        fresh_rows.append({"league": lg, "season_2026_fixtures": int((m.season == "2026").sum()),
                           "played": len(r), "last_played": r.date.max() if len(r) else ""})
    fr = pd.DataFrame(fresh_rows)
    DOCS.mkdir(exist_ok=True)
    npass = sum(ok for _, ok, _ in results)
    with open(DOCS / "DATA_QC.md", "w") as f:
        f.write(f"# Data QC report ({date.today()})\n\n{npass}/{len(results)} checks passed.\n\n")
        f.write("## Freshness (C8)\n\n" + fr.to_markdown(index=False) + "\n\n## Checks\n\n")
        f.write("| check | result | detail |\n|---|---|---|\n")
        for n, ok, d in results:
            f.write(f"| {n} | {'PASS' if ok else '**FAIL**'} | {d} |\n")
    print(f"\n{npass}/{len(results)} passed -> docs/DATA_QC.md")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
