#!/usr/bin/env python3
"""Crosswalk FPL <-> Understat (EPL only), in three layers.

A. FIXTURES  FPL (season, fixture) -> Understat match_id.
   Evidence: kickoff time (+-36h) and final score. Where several Understat matches fit
   (same score, similar time), the team map learnt from the unambiguous fixtures of the
   same season decides. Home/away team ids per FPL fixture come from `opponent_team`
   (correct per row, unlike the end-of-season team on the player record).
B. TEAMS     (season, FPL team id) -> Understat team, by vote over matched fixtures;
   asserted to be one-to-one within each season.
C. PLAYERS   (season, fixture, element) -> Understat player_id, inside each matched
   fixture and side. Hungarian assignment on a cost made of name dissimilarity, minutes
   difference, and goal/assist/card disagreement. Pairs are then consolidated across all
   matches into a persistent FPL `code` -> Understat player_id map, which is re-applied
   (pass 2) to pin known players before assigning the rest.

Only players with minutes > 0 can be matched: Understat has no row for unused substitutes
or non-playing squad members. Their xG/xA are therefore missing, not zero.

Outputs (data/crosswalk/): fixture_crosswalk.csv, team_crosswalk.csv,
player_crosswalk.csv (persistent), player_match_crosswalk.csv, and docs/CROSSWALK_QC.md.
"""
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402

OUT = DATA / "crosswalk"
FPL = DATA / "fpl"
US = DATA / "understat_clean"
TIME_TOL_H = 36


# ---------------------------------------------------------------- A + B: fixtures, teams
def fpl_fixtures(pm):
    p = pm[(pm.minutes > 0) & pm.fixture_finished & pm.team_h_score.notna()].copy()
    p["was_home"] = p.was_home.astype(str).eq("True")
    rows = []
    for (s, fx), g in p.groupby(["season", "fixture"]):
        home_id = g[~g.was_home].opponent_team.mode()
        away_id = g[g.was_home].opponent_team.mode()
        home_name = g[g.was_home].team_name.dropna().mode()
        away_name = g[~g.was_home].team_name.dropna().mode()
        rows.append({"season": s, "fixture": fx, "kick": g.kickoff_time.min(),
                     "hs": int(g.team_h_score.iloc[0]), "as_": int(g.team_a_score.iloc[0]),
                     "home_id": home_id.iloc[0] if len(home_id) else np.nan,
                     "away_id": away_id.iloc[0] if len(away_id) else np.nan,
                     "home_name": home_name.iloc[0] if len(home_name) else None,
                     "away_name": away_name.iloc[0] if len(away_name) else None})
    f = pd.DataFrame(rows)
    f["kick"] = pd.to_datetime(f.kick, utc=True).dt.tz_localize(None)
    return f


def match_fixtures(f, us):
    out, tmaps, vetoed = [], {}, []
    for s, fs in f.groupby("season"):
        u = us[us.season == int(s[:4])].copy()
        cand = {}
        for r in fs.itertuples():
            dt = (u.date - r.kick).abs() <= pd.Timedelta(hours=TIME_TOL_H)
            same = (u.home_goals == r.hs) & (u.away_goals == r.as_)
            cand[r.fixture] = list(u[dt & same].match_id)
        fs = fs.set_index("fixture")
        res, tmap = {}, {"h": defaultdict(Counter)}
        for _ in range(6):
            changed = False
            for fx, c in cand.items():
                if fx in res:
                    continue
                r = fs.loc[fx]
                if len(c) == 1:
                    res[fx] = (c[0], "score+time", r)
                    changed = True
                elif len(c) > 1 and tmap["h"]:
                    ok = []
                    for mid in c:
                        m = u[u.match_id == mid].iloc[0]
                        th = max(tmap["h"][r.home_id].items(), key=lambda kv: kv[1])[0] \
                            if tmap["h"].get(r.home_id) else None
                        ta = max(tmap["h"][r.away_id].items(), key=lambda kv: kv[1])[0] \
                            if tmap["h"].get(r.away_id) else None
                        if (th in (None, m.home_team)) and (ta in (None, m.away_team)) and \
                                (th is not None or ta is not None):
                            ok.append(mid)
                    if len(ok) == 1:
                        res[fx] = (ok[0], "score+time+teams", r)
                        changed = True
            # (re)learn team map from everything resolved so far, one vote per fixture
            tmap = {"h": defaultdict(Counter)}
            for fx, (mid, _, r) in res.items():
                m = u[u.match_id == mid].iloc[0]
                tmap["h"][r.home_id][m.home_team] += 1
                tmap["h"][r.away_id][m.away_team] += 1
            if not changed:
                break
        # fixtures with no (score+time) candidate: relax score, keep teams+time
        for fx, c in cand.items():
            if fx in res:
                continue
            r = fs.loc[fx]
            dt = (u.date - r.kick).abs() <= pd.Timedelta(hours=TIME_TOL_H)
            ok = []
            for m in u[dt].itertuples():
                th = tmap["h"].get(r.home_id)
                ta = tmap["h"].get(r.away_id)
                if th and ta and max(th, key=th.get) == m.home_team and max(ta, key=ta.get) == m.away_team:
                    ok.append(m.match_id)
            if len(ok) == 1:
                res[fx] = (ok[0], "teams+time (score differs)", r)
        # final veto: a candidate whose teams contradict the season's team map (built by majority
        # over ALL resolved fixtures) is a coincidence of score and date, not the same match.
        final = defaultdict(Counter)
        for fx, (mid, _, r) in res.items():
            m = u[u.match_id == mid].iloc[0]
            final[r.home_id][m.home_team] += 1
            final[r.away_id][m.away_team] += 1
        for fx in list(res):
            mid, how, r = res[fx]
            m = u[u.match_id == mid].iloc[0]
            for tid, name in ((r.home_id, m.home_team), (r.away_id, m.away_team)):
                c = final[tid]
                top, n = c.most_common(1)[0]
                if top != name and n >= 3:
                    del res[fx]
                    vetoed.append({"season": s, "fixture": fx, "us_match_id": mid, "reason": "teams contradict season team map"})
                    break
        used = Counter(v[0] for v in res.values())
        for fx, (mid, how, r) in res.items():
            m = u[u.match_id == mid].iloc[0]
            out.append({"season": s, "fixture": fx, "us_match_id": mid, "method": how,
                        "dup_use": used[mid] > 1, "fpl_kick": r.kick, "us_date": m.date,
                        "hours_apart": (m.date - r.kick).total_seconds() / 3600,
                        "fpl_home_id": r.home_id, "fpl_away_id": r.away_id,
                        "fpl_home_name": r.home_name, "fpl_away_name": r.away_name,
                        "us_home": m.home_team, "us_away": m.away_team})
        tmaps[s] = tmap["h"]
    return pd.DataFrame(out), pd.DataFrame(vetoed, columns=['season', 'fixture', 'us_match_id', 'reason'])


def team_crosswalk(fxw, us_teams):
    rows = []
    for s, g in fxw.groupby("season"):
        votes = defaultdict(Counter)
        names = defaultdict(Counter)
        for r in g.itertuples():
            votes[r.fpl_home_id][r.us_home] += 1
            votes[r.fpl_away_id][r.us_away] += 1
            if r.fpl_home_name:
                names[r.fpl_home_id][r.fpl_home_name] += 1
            if r.fpl_away_name:
                names[r.fpl_away_id][r.fpl_away_name] += 1
        for tid, c in votes.items():
            best, n = c.most_common(1)[0]
            rows.append({"season": s, "fpl_team_id": int(tid),
                         "fpl_team_name": names[tid].most_common(1)[0][0] if names[tid] else None,
                         "us_team": best, "us_team_id": us_teams.get((int(s[:4]), best)),
                         "n_fixtures": sum(c.values()), "vote_share": n / sum(c.values())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- C: players
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"_\d+$", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


@lru_cache(maxsize=None)
def name_sim(a, b):
    if not a or not b:
        return 0.0
    full = SequenceMatcher(None, a, b).ratio()
    ta, tb = a.split(), b.split()
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    tok = np.mean([max(SequenceMatcher(None, x, y).ratio() for y in big) for x in small])
    return max(full, 0.95 * tok)


def best_sim(variants, uname):
    return max(name_sim(v, uname) for v in variants)


def cost_matrix(fp, ur):
    C = np.zeros((len(fp), len(ur)))
    N = np.zeros_like(C)
    M = np.zeros_like(C)
    for i, f in enumerate(fp.itertuples()):
        var = f.variants
        for j, u in enumerate(ur.itertuples()):
            n = 1 - best_sim(var, u.un)
            dm = abs(f.minutes - u.time)
            pen = (f.goals_scored != u.goals) * 1.0 + (f.assists != u.assists) * 0.5 \
                + (f.red_cards != u.red_card) * 0.3
            N[i, j], M[i, j] = n, dm
            C[i, j] = n + min(dm, 30) / 30 + pen
    return C, N, M


def assign(fp, ur, pinned=None):
    """Hungarian within one team-match. Returns list of (fpl_idx, us_idx, cost, name_cost, dmin, how)."""
    pairs, used_f, used_u = [], set(), set()
    if pinned:
        for i, f in enumerate(fp.itertuples()):
            j = pinned.get(f.code)
            if j is not None:
                hit = np.where(ur.player_id.to_numpy() == j)[0]
                if len(hit):
                    u = ur.iloc[hit[0]]
                    pairs.append((i, hit[0], 0.0, 1 - best_sim(f.variants, u.un), abs(f.minutes - u.time),
                                  "pinned (persistent map)"))
                    used_f.add(i)
                    used_u.add(hit[0])
    fi = [i for i in range(len(fp)) if i not in used_f]
    ui = [j for j in range(len(ur)) if j not in used_u]
    if fi and ui:
        C, N, M = cost_matrix(fp.iloc[fi], ur.iloc[ui])
        r, c = linear_sum_assignment(C)
        for a, b in zip(r, c):
            if C[a, b] <= 1.6:  # cap: beyond this the pair is not credible
                strong = N[a, b] <= 0.45 and M[a, b] <= 5
                pairs.append((fi[a], ui[b], C[a, b], N[a, b], M[a, b],
                              "assigned (strong)" if strong else "assigned (weak)"))
    return pairs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pm = pd.read_csv(FPL / "fpl_player_match.csv", low_memory=False, parse_dates=["kickoff_time"])
    ids = pd.read_csv(FPL / "fpl_player_ids.csv")
    us_m = pd.read_csv(US / "EPL_matches.csv", parse_dates=["date"])
    us_m = us_m[us_m.isResult == True]  # noqa: E712
    us_t = pd.read_csv(US / "EPL_team_match_stats.csv", usecols=["season", "team_id", "team_title"]).drop_duplicates()
    us_teams = {(int(r.season), r.team_title): int(r.team_id) for r in us_t.itertuples()}

    f = fpl_fixtures(pm)
    fxw, vetoed = match_fixtures(f, us_m)
    vetoed.to_csv(OUT / 'fixture_vetoed.csv', index=False)
    fxw.to_csv(OUT / "fixture_crosswalk.csv", index=False)
    txw = team_crosswalk(fxw, us_teams)
    txw.to_csv(OUT / "team_crosswalk.csv", index=False)

    # ---- players
    usp = pd.read_csv(US / "EPL_player_match_stats.csv", low_memory=False)
    usp = usp[usp.time > 0].copy()
    usp["un"] = usp.player.map(norm)
    ids["variants_base"] = [
        tuple(dict.fromkeys(x for x in (norm(f"{a} {b}"), norm(w), norm(b)) if x))
        for a, b, w in zip(ids.first_name.fillna(""), ids.second_name.fillna(""), ids.web_name.fillna(""))]
    pmp = pm[(pm.minutes > 0) & pm.fixture_finished & pm.position.isin(["GKP", "DEF", "MID", "FWD"])].copy()
    pmp = pmp.merge(ids[["season", "element", "variants_base"]], on=["season", "element"], how="left")
    pmp["variants"] = [tuple(dict.fromkeys(list(v if isinstance(v, tuple) else ()) + [norm(n)]))
                       for v, n in zip(pmp.variants_base, pmp.name)]
    pmp["was_home"] = pmp.was_home.astype(str).eq("True")
    pmp = pmp.merge(fxw[["season", "fixture", "us_match_id"]], on=["season", "fixture"], how="inner")

    def run_pass(pinned):
        recs = []
        usg = {k: g for k, g in usp.groupby(["match_id", "h_a"])}
        for (s, fx, side), fp in pmp.groupby(["season", "fixture", "was_home"]):
            fp = fp.reset_index(drop=True)
            mid = fp.us_match_id.iloc[0]
            ur = usg.get((mid, "h" if side else "a"))
            if ur is None:
                continue
            ur = ur.reset_index(drop=True)
            for i, j, c, n, dm, how in assign(fp, ur, pinned):
                recs.append({"season": s, "fixture": fx, "element": fp.element[i], "code": fp.code[i],
                             "us_match_id": mid, "us_player_id": ur.player_id[j], "us_name": ur.player[j],
                             "fpl_name": fp.variants[i][0] if fp.variants[i] else fp["name"][i],
                             "fpl_minutes": fp.minutes[i], "us_minutes": ur.time[j],
                             "cost": c, "name_cost": n, "minute_diff": dm, "method": how})
        return pd.DataFrame(recs)

    def consolidate(pairs, min_votes=1):
        good = pairs[(pairs.name_cost <= 0.45) & (pairs.minute_diff <= 5)]
        votes = good.groupby(["code", "us_player_id"]).size().rename("n").reset_index()
        tot = pairs.groupby("code").size().rename("total")
        best = votes.sort_values("n", ascending=False).drop_duplicates("code").merge(tot, on="code")
        best["share"] = best.n / best.total
        return best[best.n >= min_votes]

    if "--report-only" in sys.argv:  # reuse saved matching, only rebuild the QC report
        p2 = pd.read_csv(OUT / "player_match_crosswalk.csv")
        cons2 = consolidate(p2)
    else:
        p1 = run_pass(None)
        cons1 = consolidate(p1)
        pinned = dict(zip(cons1.code, cons1.us_player_id))
        p2 = run_pass(pinned)
        cons2 = consolidate(p2)
        p2.to_csv(OUT / "player_match_crosswalk.csv", index=False)
    names = ids.drop_duplicates("code").set_index("code")
    cons2["fpl_name"] = cons2.code.map(lambda c: f"{names.loc[c, 'first_name']} {names.loc[c, 'second_name']}"
                                       if c in names.index else None)
    cons2["us_name"] = cons2.us_player_id.map(usp.drop_duplicates("player_id").set_index("player_id").player)
    # conflicts: one code -> several us ids (>=2 votes each), or one us id -> several codes
    allv = p2[(p2.name_cost <= 0.45) & (p2.minute_diff <= 5)].groupby(["code", "us_player_id"]).size()
    c_multi = allv[allv >= 2].reset_index().groupby("code").size()
    u_multi = allv[allv >= 2].reset_index().groupby("us_player_id").size()
    cons2["conflict"] = cons2.code.isin(c_multi[c_multi > 1].index) | cons2.us_player_id.isin(u_multi[u_multi > 1].index)
    cons2.to_csv(OUT / "player_crosswalk.csv", index=False)

    # ---------------- QC report
    fpl_n = f.groupby("season").size()
    mt = fxw.groupby("season").size()
    us_n = us_m.assign(s=us_m.season.map(lambda x: f"{x}-{str(x+1)[2:]}")).groupby("s").size()
    cov = pd.DataFrame({"fpl_fixtures": fpl_n, "matched": mt, "understat_matches": us_n}).dropna(subset=["fpl_fixtures"])
    cov["fpl_unmatched"] = cov.fpl_fixtures - cov.matched
    cov["dup_use"] = fxw.groupby("season").dup_use.sum()
    cov = cov.fillna(0).astype(int)
    meth = fxw.groupby(["season", "method"]).size().unstack(fill_value=0)
    tq = txw.groupby("season").agg(teams=("fpl_team_id", "size"), min_vote_share=("vote_share", "min"),
                                   bijective=("us_team", lambda x: x.is_unique))
    tot_rows = pmp.groupby("season").size()
    matched = p2.groupby("season").size()
    pc = pd.DataFrame({"fpl_rows_min>0": tot_rows, "matched": matched})
    pc["rate"] = (pc.matched / pc["fpl_rows_min>0"]).round(4)
    strong = p2[p2.method != "assigned (weak)"].groupby("season").size()
    pc["strong_or_pinned_rate"] = (strong / pc["fpl_rows_min>0"]).round(4)
    gm = p2.merge(pmp[["season", "fixture", "element", "goals_scored", "assists"]], on=["season", "fixture", "element"])
    gm = gm.merge(usp[["match_id", "player_id", "goals", "assists"]].rename(
        columns={"match_id": "us_match_id", "player_id": "us_player_id", "assists": "us_assists"}),
        on=["us_match_id", "us_player_id"])
    agree = pd.DataFrame({"minutes within 3": (gm.minute_diff <= 3).groupby(gm.season).mean().round(4),
                          "goals equal": (gm.goals_scored == gm.goals).groupby(gm.season).mean().round(4),
                          "assists equal": (gm.assists == gm.us_assists).groupby(gm.season).mean().round(4)})
    # understat side: share of understat appearances (time>0) that got a FPL partner
    usg = usp.drop(columns="season").merge(
        fxw[["us_match_id", "season"]].rename(columns={"us_match_id": "match_id"}), on="match_id")
    usg["matched"] = usg.set_index(["match_id", "player_id"]).index.isin(
        p2.set_index(["us_match_id", "us_player_id"]).index)
    us_cov = usg.groupby("season").matched.mean().round(4).rename("understat rows matched")

    with open(DOCS / "CROSSWALK_QC.md", "w") as fh:
        fh.write(f"# FPL <-> Understat crosswalk QC ({date.today()})\n\n")
        fh.write("## Fixtures (FPL fixture -> Understat match)\n\n" + cov.to_markdown() + "\n\n")
        fh.write("Method used:\n\n" + meth.to_markdown() + "\n\n")
        fh.write("## Teams\n\n" + tq.to_markdown() + "\n\n")
        fh.write("## Players: coverage of FPL appearances (minutes > 0) in matched fixtures\n\n" + pc.to_markdown() + "\n\n")
        fh.write("Understat appearances that found an FPL partner:\n\n" + us_cov.to_markdown() + "\n\n")
        fh.write("## Players: agreement of independent attributes on matched pairs\n\n" + agree.to_markdown() + "\n\n")
        weak = p2[p2.method == "assigned (weak)"][["season", "fixture", "fpl_name", "us_name", "fpl_minutes",
                                                     "us_minutes", "name_cost"]]
        fh.write("## Evidence that does not depend on goals/assists (which are inputs to the match cost)\n\n"
                 f"- pairs where the NAMES alone agree (name cost <= 0.25): {(p2.name_cost <= 0.25).mean():.2%} "
                 f"of {len(p2):,}; <= 0.45: {(p2.name_cost <= 0.45).mean():.2%}\n"
                 f"- one-to-one persistent map: Understat ids per FPL code > 1: "
                 f"{int((cons2.groupby('code').us_player_id.nunique() > 1).sum())}; "
                 f"FPL codes per Understat id > 1: {int((cons2.groupby('us_player_id').code.nunique() > 1).sum())}\n"
                 f"- pairs that contradict the persistent map: "
                 f"{int((p2.merge(cons2[['code', 'us_player_id']], on='code', suffixes=('', '_map')).eval('us_player_id != us_player_id_map')).sum())}\n\n"
                 "Weak pairs left after pass 2 (name or minutes disagree; review manually):\n\n"
                 + (weak.to_markdown(index=False) if len(weak) else "none") + "\n\n")
        fh.write(f"## Persistent map\n\n{len(cons2):,} FPL codes mapped; {int(cons2.conflict.sum())} flagged as conflicts "
                 f"(see player_crosswalk.csv).\n")
    print(cov.to_string()); print(); print(pc.to_string()); print(); print(agree.to_string())
    print(f"\npersistent map: {len(cons2)} codes, conflicts: {int(cons2.conflict.sum())}")


if __name__ == "__main__":
    main()
