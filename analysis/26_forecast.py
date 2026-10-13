#!/usr/bin/env python3
"""Live forecast: xP GIVEN THE PLAYER STARTS for every registered player's fixtures in the next H gameweeks,
with every feature frozen at the next gameweek deadline (the decision time).

ROWS. Unplayed fixtures of the next H gameweeks (latest FPL API snapshot) become placeholder player-match
rows for every registered player (status other than 'u'); 06.build and 12.build compute their features
together with the finished rows, so each player's first placeholder (the ANCHOR) carries exactly the state at
the deadline; 22.season_rows then builds every later fixture from the anchor, as in the horizon backtest
(docs/HORIZON_RESULTS.md: no loss of accuracy up to 8 gameweeks). Team ratings for upcoming fixtures come
from the ratings replay's current state (pipeline/11 via 17).
MODEL. The saved live model (13_xp_model.py, fitted on seasons before the current one), unchanged.
BONUS COMPETITION. Bonus is ranked within the match, so each fixture is simulated with a PROBABLE XI per
team (1 goalkeeper + 10 outfield players, ranked by: available for the next gameweek (FPL status not
injured / suspended and chance of playing >= 50), starts in the last 10 fixtures, recent minutes, price).
Every other player is simulated in his own copy of the match, replacing the lowest-ranked XI player of his
position (or the lowest-ranked outfield player). xP is still conditional on the player starting.
GATES (all must pass, else nothing is written): deadline not passed; FPL tables and ratings include every
fixture the snapshot reports finished; finished rows rebuilt here equal master.parquet; the saved model's
training data hash matches the data on disk; every registered player has every fixture of his team in the
window; ratings and model inputs present; xP sane.
Output: data/forecasts/forecast_<season>_gw<t>_<snapshot>.json (+ .parquet). Options: --horizon N (8),
--dry-run (run and report the gates, write nothing), --snapshot <dir name> (default latest).
"""
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
os.environ.setdefault("XFPL_RATINGS", str(DATA / "ratings_eval"))
RATINGS = Path(os.environ["XFPL_RATINGS"])
FORECASTS = DATA / "forecasts"
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SCHEMA = 1


def load_mod(name, path):
    sp = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


M06 = load_mod("m06", "analysis/06_build_minutes_dataset.py")
M12 = load_mod("m12", "analysis/12_build_master.py")
H22 = load_mod("h22", "analysis/22_build_horizon_rows.py")
X13 = load_mod("x13", "analysis/13_xp_model.py")
R21 = load_mod("r21", "analysis/21_regression_check.py")


def latest_snapshot(name=None):
    snaps = sorted(p for p in (DATA / "raw" / "fpl_api").iterdir() if (p / "bootstrap.json").exists())
    snap = [p for p in snaps if p.name == name][0] if name else snaps[-1]
    return snap, json.load(open(snap / "bootstrap.json")), json.load(open(snap / "fixtures.json"))


def placeholders(boot, fixtures, season, rounds):
    """Player-match rows (fpl_player_match schema) for every registered player x his team's unplayed
    fixtures in `rounds`. Outcomes unknown (minutes NaN)."""
    el = pd.DataFrame(boot["elements"])
    el = el[(el.status != "u") & el.element_type.isin(POS)]
    total = boot.get("total_players") or 1
    fx = pd.DataFrame(fixtures)
    fx = fx[~fx.finished & fx.event.isin(rounds)]
    rows = []
    for home in (True, False):
        side = fx.rename(columns={"team_h" if home else "team_a": "team"})
        j = side[["id", "event", "team", "kickoff_time"]].merge(el, on="team", suffixes=("_fx", ""))
        rows.append(pd.DataFrame({
            "season": season, "round": j.event.astype(int), "element": j["id"], "code": j.code,
            "position": j.element_type.map(POS), "fixture": j.id_fx, "was_home": home, "minutes": np.nan,
            "value": j.now_cost.astype(float), "selected": j.selected_by_percent.astype(float) / 100 * total,
            "kickoff_time": j.kickoff_time, "fixture_finished": False}))
    return pd.concat(rows, ignore_index=True)


def league_yellow_at(hist, deadline, days=365):
    """log league yellow cards per 90 over the `days` before the deadline (as 13.league_yellow_rate)."""
    w = hist[(hist.kickoff_time < deadline) & (hist.kickoff_time >= deadline - pd.Timedelta(days=days))]
    mins = w.minutes.sum()
    assert mins >= 50 * 22 * 90, "too little history for the league yellow rate"
    return float(np.log(90 * w.yellow_cards.sum() / mins))


def match_sets(d, avail):
    """Simulation rows: per fixture a probable-XI match, plus one copy per other player (see header)."""
    key = d.assign(_av=d.element.map(avail).fillna(True).astype(int), _st=d.st_n10.fillna(0), _m5=d.m5.fillna(0))
    idx, sim_fx, focal, in_xi = [], [], [], []

    def add(rows, fx_id, foc, xi):
        idx.extend(rows)
        sim_fx.extend([fx_id] * len(rows))
        focal.extend([r in foc for r in rows])
        in_xi.extend([xi] * len(rows))

    for f, g in key.groupby("fixture", sort=True):
        sides = {}
        for home, s in g.groupby("was_home"):
            s = s.sort_values(["_av", "_st", "_m5", "value", "element"], ascending=[False, False, False, False, True])
            xi = list(s.index[s.position == "GKP"][:1]) + list(s.index[s.position != "GKP"][:10])
            sides[home] = (xi, [i for i in s.index if i not in xi])
        xi_all = [i for v in sides.values() for i in v[0]]
        add(xi_all, f * 1000, set(xi_all), True)
        k = 1
        for home, (xi, bench) in sides.items():
            other = [i for h, v in sides.items() if h != home for i in v[0]]
            for p in bench:
                same = [i for i in xi if d.position[i] == d.position[p]]
                pool = same or [i for i in xi if d.position[i] != "GKP"]
                rows = [i for i in xi if i != pool[-1]] + [p] + other
                add(rows, f * 1000 + k, {p}, False)
                k += 1
    S = d.loc[idx].reset_index().rename(columns={"index": "row"})
    S["sim_fixture"], S["focal"], S["in_xi"] = sim_fx, focal, in_xi
    return S


def main():
    args = sys.argv[1:]
    H = int(args[args.index("--horizon") + 1]) if "--horizon" in args else 8
    dry = "--dry-run" in args
    snap, boot, fixtures = latest_snapshot(args[args.index("--snapshot") + 1] if "--snapshot" in args else None)
    ev = pd.DataFrame(boot["events"])
    ev["deadline"] = pd.to_datetime(ev.deadline_time, utc=True).dt.tz_localize(None)
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    nxt = ev[ev.deadline > now].sort_values("deadline").iloc[0]
    t, deadline = int(nxt.id), nxt.deadline
    rounds = list(range(t, min(t + H - 1, int(ev.id.max())) + 1))
    season = pd.read_csv(DATA / "fpl" / "fpl_events.csv").season.iloc[0]
    gates = []

    def gate(name, ok, detail=""):
        gates.append((name, bool(ok), detail))
        print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""), flush=True)

    print(f"snapshot {snap.name}; season {season}; next deadline GW{t} {deadline} UTC; gameweeks {rounds[0]}-{rounds[-1]}")
    gate("G1 deadline not passed", now < deadline, f"now {now:%Y-%m-%d %H:%M} UTC")
    snap_t = pd.Timestamp(datetime.strptime(snap.name, "%Y%m%dT%H%M%SZ"))
    gate("G1 snapshot taken before the deadline", snap_t < deadline, f"snapshot {snap_t}")

    # ---- data current: every finished fixture in the FPL tables and rated from a deadline snapshot
    fx = pd.DataFrame(fixtures)
    done = set(fx[fx.finished].id)
    pm = pd.read_csv(DATA / "fpl" / "fpl_player_match.csv", low_memory=False,
                     usecols=["season", "round", "element", "code", "position", "fixture", "was_home", "minutes",
                              "value", "selected", "kickoff_time", "fixture_finished", "yellow_cards"])
    cur = pm[pm.season == season]
    in_tables = set(cur[cur.fixture_finished.astype(bool)].fixture)
    gate("G2 FPL tables include every finished fixture", done <= in_tables, f"missing {sorted(done - in_tables)[:10]}")
    rt = pd.read_csv(RATINGS / "fpl_team_fixture_ratings.csv")
    rt["was_home"] = rt.was_home.astype(str).eq("True")
    rated_live = set(rt[(rt.season == season) & rt.is_prediction].fixture)
    gate("G2 every finished fixture rated from a deadline snapshot (results in the ratings replay)",
         not (done & rated_live), f"rated from current state: {sorted(done & rated_live)[:10]}")

    # ---- features: finished rows + placeholders through 06 and 12
    fin = pm[pm.position.isin(list(POS.values())) & pm.fixture_finished.astype(bool)].drop(columns="yellow_cards")
    ph = placeholders(boot, fixtures, season, rounds)
    print(f"placeholders: {len(ph):,} rows ({ph.element.nunique()} players, {ph.fixture.nunique()} fixtures)", flush=True)
    us_states = M12.understat_post_states()
    md = M06.build(pd.concat([fin, ph], ignore_index=True))
    allm = M12.build(md, us_states)
    master = pd.read_parquet(DATA / "analysis" / "master.parquet")
    played = allm[allm.minutes.notna()].reset_index(drop=True)
    key = ["season", "element", "fixture"]
    a_ = played.sort_values(key).reset_index(drop=True).drop(columns="row_id")   # row_id: a running number
    b_ = master.sort_values(key).reset_index(drop=True)[a_.columns]
    probs = R21.compare(a_, b_, "finished rows")   # values, not dtypes (placeholder NaNs upcast integers)
    gate("G3 finished rows rebuilt here equal master.parquet (features up to date)", not probs,
         f"{len(a_):,} vs {len(b_):,} rows" + ("; " + "; ".join(probs[:3]) if probs else ""))
    dl_m = allm.loc[(allm.season == season) & (allm["round"] == t), "deadline"].iloc[0]
    gate("G3 deadline used for the features = FPL's published deadline", dl_m == deadline, f"{dl_m} vs {deadline}")

    # ---- model
    M = X13.load_model("live", season)
    full = X13.load(season)
    tr = full[full.season < season]
    gate("G4 saved model's training data unchanged", X13.frame_hash(tr) == M["train_hash"],
         f"model {M['tag']} {M['season']} commit {M['code_commit']}")
    lg = league_yellow_at(full, deadline)
    last_dl = full.deadline.max()
    gate("G4 league yellow rate helper reproduces the model's value at the last deadline",
         abs(league_yellow_at(full, last_dl) - full.loc[full.deadline == last_dl, "log_lg_yel"].iloc[0]) < 1e-12)

    # ---- horizon rows from the deadline anchor
    m = allm[allm.season == season].merge(
        rt[["season", "fixture", "was_home", "fpl_team_id"]].rename(columns={"fpl_team_id": "team_id"}),
        on=["season", "fixture", "was_home"], how="left")
    ra = rt[(rt.season == season) & rt.event.isin(rounds)].assign(asof_round=t)
    d = H22.season_rows(m, rt[rt.season == season], ra, season, H, us_states, rounds=[t])
    d = d[d.minutes.isna()].reset_index(drop=True)
    want = ph[["element", "fixture"]].drop_duplicates()
    got = d[["element", "fixture"]].drop_duplicates()
    miss = want.merge(got, how="left", indicator=True).query("_merge == 'left_only'")
    gate("G5 every registered player has every fixture of his team in the window", miss.empty,
         f"{len(got):,} of {len(want):,} player-fixtures; missing players {miss.element.nunique()}")
    gate("G5 team ratings present for every row", d.xg_for_pi.notna().all())
    d = X13.prepare(d, pd.Series({deadline: lg}))
    glm_x = sorted({c for s in X13.GLM_SPECS.values() for c in s["x"]} - set(X13.EB_COLS) -
                   {"log_fr_pen", "log_yel90", "log_red90", "log_sv90"})
    finite = np.isfinite(d[glm_x].to_numpy(float)).all()
    gate("G5 model inputs finite", finite, ", ".join(glm_x))

    # ---- simulate
    el = pd.DataFrame(boot["elements"]).set_index("id")
    cop = el.chance_of_playing_next_round
    avail = (~el.status.isin(["i", "s", "u"])) & (cop.isna() | (cop >= 50))
    S = match_sets(d, avail)
    print(f"simulating {len(S):,} rows in {S.sim_fixture.nunique():,} match copies ...", flush=True)
    te = S.drop(columns=["fixture"]).rename(columns={"sim_fixture": "fixture"})
    seed = X13.SEED + int(deadline.timestamp()) % 1_000_000
    res, _, _ = X13.predict_rows(M, te, np.random.default_rng(seed))
    S = pd.concat([S, res], axis=1)
    out = S[S.focal].set_index("row").sort_index()
    out = d.join(out[[c for c in res.columns] + ["in_xi"]])
    gate("G6 one prediction per player-fixture", out.xp.notna().all() and len(out) == len(d), f"{len(out):,} rows")
    gate("G6 xP finite and within [0, 25]", bool(np.isfinite(out.xp).all() and out.xp.between(0, 25).all()),
         f"range {out.xp.min():.2f}-{out.xp.max():.2f}")
    mx = out[out.in_xi.astype(bool)].xp.mean()
    gate("G6 mean xP of probable starters within [2.5, 4.5]", 2.5 <= mx <= 4.5, f"{mx:.3f}")

    ok = all(g[1] for g in gates)
    print(f"\n{sum(g[1] for g in gates)}/{len(gates)} gates passed")
    if dry or not ok:
        print("dry run: nothing written" if dry else "GATES FAILED: nothing written")
        if not ok:
            sys.exit(1)
        return
    write(out, el, boot, fixtures, season, t, deadline, rounds, snap.name, M, gates)


def write(out, el, boot, fixtures, season, t, deadline, rounds, snap_name, M, gates):
    FORECASTS.mkdir(parents=True, exist_ok=True)
    stem = f"forecast_{season}_gw{t:02d}_{snap_name}"
    teams = {tm["id"]: tm for tm in boot["teams"]}
    out = out.assign(team_id=out.team_id.astype(int))
    opp = {(f["id"], True): f["team_a"] for f in fixtures} | {(f["id"], False): f["team_h"] for f in fixtures}
    out["opp_id"] = [opp[(f, h)] for f, h in zip(out.fixture, out.was_home)]
    out["var_ale"] = out.var_total - out.var_epi
    keep = ["season", "round", "horizon", "fixture", "element", "code", "position", "team_id", "opp_id", "was_home",
            "kickoff_time", "in_xi", "xp", "var_total", "var_epi", "var_ale", "q10", "q25", "q50", "q75", "q90",
            "p_ge2", "p_ge5", "p_ge10", "e_app", "e_goals", "e_assists", "e_cs", "e_gc_pts", "e_saves_pts", "e_bonus",
            "e_yellow", "e_red", "e_og", "e_pen_miss", "e_pen_saved", "e_minutes", "e_dc", "value", "xg_for_pi",
            "xg_against_pi"]
    out[keep].to_parquet(FORECASTS / f"{stem}.parquet", index=False)
    r3 = lambda x: None if pd.isna(x) else round(float(x), 3)
    players = []
    for pid, g in out.sort_values("kickoff_time").groupby("element"):
        e = el.loc[pid]
        players.append({
            "id": int(pid), "code": int(e.code), "name": e.web_name, "full_name": f"{e.first_name} {e.second_name}",
            "team": int(e.team), "pos": POS[int(e.element_type)], "price": e.now_cost / 10,
            "status": e.status, "chance_next": None if pd.isna(e.chance_of_playing_next_round) else int(e.chance_of_playing_next_round),
            "news": e.news or "",
            "fixtures": [{"gw": int(r["round"]), "h": int(r.horizon), "fixture": int(r.fixture), "opp": int(r.opp_id),
                          "home": bool(r.was_home), "kickoff": r.kickoff_time.strftime("%Y-%m-%dT%H:%MZ"),
                          "xi": bool(r.in_xi), "xp": r3(r.xp), "q10": r3(r.q10), "q50": r3(r.q50), "q90": r3(r.q90),
                          "p5": r3(r.p_ge5), "p10": r3(r.p_ge10), "var_ale": r3(r.var_ale), "sd_epi": r3(np.sqrt(max(r.var_epi, 0))),
                          "parts": {k: r3(r[f"e_{k}"]) for k in ("app", "goals", "assists", "cs", "gc_pts", "saves_pts",
                                                               "bonus", "dc", "yellow", "red", "og", "pen_miss", "pen_saved")}}
                         for _, r in g.iterrows()]})
    doc = {"schema_version": SCHEMA, "target": "expected FPL points GIVEN THAT THE PLAYER STARTS",
           "season": season, "deadline_gw": t, "deadline_utc": deadline.strftime("%Y-%m-%dT%H:%MZ"),
           "gameweeks": rounds, "snapshot_utc": snap_name,
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "model": {k: M[k] for k in ("tag", "season", "code_commit", "train_hash", "created_utc")},
           "intervals": "not validated for sums over gameweeks (docs/HORIZON_RESULTS.md); per-fixture quantiles only",
           "gates": [{"name": n, "pass": p, "detail": dt} for n, p, dt in gates],
           "teams": [{"id": tm["id"], "name": tm["name"], "short": tm["short_name"]} for tm in teams.values()],
           "players": players}
    (FORECASTS / f"{stem}.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {FORECASTS / stem}.json and .parquet ({len(players)} players, {len(out):,} player-fixtures)")


if __name__ == "__main__":
    main()
