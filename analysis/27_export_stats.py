#!/usr/bin/env python3
"""Stats sheet for the website: current-season player and team statistics (descriptive, not model output).

Sources: FPL per-match history (data/fpl/fpl_player_match.csv), Understat EPL player-match stats and shots
(data/understat_clean, joined to FPL rows through the crosswalk), Understat team-match stats, and the team
ratings the model uses (data/ratings_eval). Two windows: the whole season so far and the last 4 finished
gameweeks. Totals only; the site derives per-90 figures.
Output: data/forecasts/stats_<season>.json (latest only; overwritten each refresh).
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RATINGS = Path(os.environ.get("XFPL_RATINGS", DATA / "ratings_eval"))
OUT = DATA / "forecasts"
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
LAST = 4
BIG_CHANCE = 0.3   # shots with xG >= 0.3 counted as big chances (an xG proxy; Opta's definition is not public)
SHOT_CODE = {"Goal": "G", "SavedShot": "S", "MissedShots": "M", "BlockedShot": "B", "ShotOnPost": "P"}

FPL_SUM = {"mins": "minutes", "starts": "starts", "pts": "total_points", "bonus": "bonus", "bps": "bps",
           "g": "goals_scored", "a": "assists", "cs": "clean_sheets", "gc": "goals_conceded", "saves": "saves",
           "pen_saved": "penalties_saved", "pen_missed": "penalties_missed", "yc": "yellow_cards", "rc": "red_cards",
           "og": "own_goals", "ict": "ict_index", "inf": "influence", "cre": "creativity", "thr": "threat",
           "fxg": "expected_goals", "fxa": "expected_assists", "fxgi": "expected_goal_involvements",
           "fxgc": "expected_goals_conceded", "cbi": "clearances_blocks_interceptions", "tkl": "tackles",
           "rec": "recoveries", "dc": "defensive_contribution"}


def latest_bootstrap():
    snaps = sorted(p for p in (DATA / "raw" / "fpl_api").iterdir() if (p / "bootstrap.json").exists())
    return json.load(open(snaps[-1] / "bootstrap.json")), snaps[-1].name


def r(x, d=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def main():
    boot, snap = latest_bootstrap()
    season = pd.read_csv(DATA / "fpl" / "fpl_events.csv").season.iloc[0]
    us_season = int(season[:4])
    pm = pd.read_csv(DATA / "fpl" / "fpl_player_match.csv", low_memory=False)
    pm = pm[(pm.season == season) & pm.fixture_finished.astype(bool)].copy()
    for c in FPL_SUM.values():
        pm[c] = pd.to_numeric(pm[c], errors="coerce").fillna(0.0)
    rounds = sorted(pm["round"].unique())
    last = rounds[-LAST:]
    windows = {"season": rounds, "last4": last}

    # ---- Understat per FPL appearance (crosswalk) + shot-derived stats
    cw = pd.read_csv(DATA / "crosswalk" / "player_match_crosswalk.csv",
                     usecols=["season", "fixture", "element", "us_match_id", "us_player_id"])
    cw = cw[cw.season == season]
    ums = pd.read_csv(DATA / "understat_clean" / "EPL_player_match_stats.csv", low_memory=False)
    ums = ums[ums.season == us_season][["match_id", "player_id", "time", "shots", "xG", "xA", "xGChain", "xGBuildup", "goals"]]
    sh = pd.read_csv(DATA / "understat_clean" / "EPL_shots.csv", low_memory=False)
    sh = sh[(sh.season == us_season) & (sh.result != "OwnGoal")].copy()
    sh["pen"] = sh.situation == "Penalty"
    sh["npxg"] = np.where(sh.pen, 0.0, sh.xG)
    sh["sot"] = sh.result.isin(["Goal", "SavedShot"])
    sh["big"] = sh.xG >= BIG_CHANCE
    sh["npg"] = (sh.result == "Goal") & ~sh.pen
    per = sh.groupby(["match_id", "player_id"]).agg(npxg=("npxg", "sum"), sot=("sot", "sum"), big=("big", "sum"),
                                                   npg=("npg", "sum")).reset_index()
    # key passes: shots a player set up (Understat names the assister); matched by name within the match and side
    kp = sh[sh.player_assisted.notna()].groupby(["match_id", "h_a", "player_assisted"]).size().rename("kp").reset_index()
    names = pd.read_csv(DATA / "understat_clean" / "EPL_player_match_stats.csv", low_memory=False,
                        usecols=["season", "match_id", "player_id", "player", "h_a"])
    names = names[names.season == us_season]
    kp = kp.merge(names, left_on=["match_id", "h_a", "player_assisted"], right_on=["match_id", "h_a", "player"])[["match_id", "player_id", "kp"]]
    u = cw.merge(ums, left_on=["us_match_id", "us_player_id"], right_on=["match_id", "player_id"], how="left") \
        .merge(per, on=["match_id", "player_id"], how="left").merge(kp, on=["match_id", "player_id"], how="left")
    ucols = {"shots": "shots", "uxg": "xG", "uxa": "xA", "xgchain": "xGChain", "xgbuildup": "xGBuildup", "npxg": "npxg",
             "sot": "sot", "big": "big", "npg": "npg", "kp": "kp"}
    pm = pm.merge(u[["fixture", "element"] + list(ucols.values())], on=["fixture", "element"], how="left")
    for c in ucols.values():
        pm[c] = pm[c].fillna(0.0)
    pm["app"] = (pm.minutes > 0).astype(float)
    pm["thr_dc"] = np.where(pm.position == "DEF", 10, 12)
    pm["dc_hit"] = ((pm.defensive_contribution >= pm.thr_dc) & (pm.position != "GKP")).astype(float)

    def agg(rows):
        out = {"apps": rows.app.sum(), "dc_hits": rows.dc_hit.sum()}
        out.update({k: rows[v].sum() for k, v in FPL_SUM.items()})
        out.update({k: rows[v].sum() for k, v in ucols.items()})
        return {k: r(v) for k, v in out.items()}

    # shots for shot maps (season), FPL element via the crosswalk
    shp = sh.merge(cw.drop_duplicates(["us_match_id", "us_player_id"]), left_on=["match_id", "player_id"],
                   right_on=["us_match_id", "us_player_id"])
    fx_round = pm.drop_duplicates("fixture").set_index("fixture")["round"]
    shots_by = {el: [[r(x.X, 3), r(x.Y, 3), r(x.xG, 3), SHOT_CODE.get(x.result, "M"), int(x.minute),
                      int(fx_round.get(x.fixture, 0)), x.situation == "Penalty"] for x in g.itertuples()]
                for el, g in shp.groupby("element")}

    def recent(g, n=6):
        """The player's last n team fixtures, appearances or not (most recent first)."""
        rows = g.sort_values("kickoff_time").tail(n).iloc[::-1]
        return [{"gw": int(x["round"]), "opp": int(x.opponent_team), "home": str(x.was_home) == "True",
                 "mins": int(x.minutes), "start": int(x.starts), "g": int(x.goals_scored), "a": int(x.assists),
                 "xg": r(x.xG, 3), "xa": r(x.xA, 3), "shots": int(x.shots), "kp": int(x.kp), "cs": int(x.clean_sheets),
                 "bonus": int(x.bonus), "pts": int(x.total_points),
                 "score": f"{int(x.team_h_score)}-{int(x.team_a_score)}" if pd.notna(x.team_h_score) else None}
                for _, x in rows.iterrows()]

    el = pd.DataFrame(boot["elements"])
    el = el[el.element_type.isin(POS)]
    by_el = {e: g for e, g in pm.groupby("element")}
    players = []
    for e in el.itertuples():
        g = by_el.get(e.id)
        if g is None or g.minutes.sum() == 0:
            continue
        players.append({"id": int(e.id), "name": e.web_name, "team": int(e.team), "pos": POS[int(e.element_type)],
                        "price": e.now_cost / 10, "own": float(e.selected_by_percent), "status": e.status,
                        "form": float(e.form), "s": {w: agg(g[g["round"].isin(rs)]) for w, rs in windows.items()},
                        "shots": shots_by.get(e.id, []), "recent": recent(g)})

    # ---- teams: Understat team-match stats (by crosswalk name) and model ratings
    txw = pd.read_csv(DATA / "crosswalk" / "team_crosswalk.csv")
    txw = txw[txw.season == season].set_index("us_team").fpl_team_id.to_dict()
    tm = pd.read_csv(DATA / "understat_clean" / "EPL_team_match_stats.csv", parse_dates=["date"])
    tm = tm[tm.season == us_season].copy()
    tm["fpl_id"] = tm.team_title.map(txw)
    fxk = pd.read_csv(DATA / "crosswalk" / "fixture_crosswalk.csv", parse_dates=["us_date"])
    fxk = fxk[fxk.season == season]
    round_of = pm.drop_duplicates("fixture").set_index("fixture")["round"].to_dict()
    date_round = {(d.normalize(), t): round_of.get(f) for d, f, t in
                  list(zip(fxk.us_date, fxk.fixture, fxk.us_home)) + list(zip(fxk.us_date, fxk.fixture, fxk.us_away))}
    tm["round"] = [date_round.get((d.normalize(), t)) for d, t in zip(tm.date, tm.team_title)]
    rt = pd.read_csv(RATINGS / "fpl_team_fixture_ratings.csv")
    rt = rt[rt.season == season].sort_values("kickoff_time")

    def tagg(rows):
        s = rows.sum(numeric_only=True)
        return {"mp": r(len(rows), 0), "w": r(s.wins, 0), "d": r(s.draws, 0), "l": r(s.loses, 0), "pts": r(s.pts, 0),
                "xpts": r(s.xpts), "gf": r(s.scored, 0), "ga": r(s.missed, 0), "xg": r(s.xG), "xga": r(s.xGA),
                "npxg": r(s.npxG), "npxga": r(s.npxGA), "cs": r((rows.missed == 0).sum(), 0),
                "ppda": r(s.ppda_att / s.ppda_def) if s.ppda_def else None,
                "oppda": r(s.ppda_allowed_att / s.ppda_allowed_def) if s.ppda_allowed_def else None,
                "deep": r(s.deep, 0), "deep_allowed": r(s.deep_allowed, 0)}

    teams = []
    for t in boot["teams"]:
        rows = tm[tm.fpl_id == t["id"]]
        hist = rt[(rt.fpl_team_id == t["id"]) & ~rt.is_prediction.astype(str).eq("True")]
        nxt = rt[(rt.fpl_team_id == t["id"]) & rt.is_prediction.astype(str).eq("True")].head(1)
        series = [{"gw": int(h.event), "att": r(h.team_att), "def": r(h.team_def), "xgf": r(h.xg_for_pi), "xga": r(h.xg_against_pi)}
                  for h in hist.itertuples()]
        if len(nxt):
            n = nxt.iloc[0]
            series.append({"gw": int(n.event), "att": r(n.team_att), "def": r(n.team_def), "xgf": r(n.xg_for_pi),
                           "xga": r(n.xg_against_pi), "next": True})
        teams.append({"id": t["id"], "name": t["name"], "short": t["short_name"],
                      "s": {w: tagg(rows[rows["round"].isin(rs)]) for w, rs in windows.items()}, "ratings": series})

    doc = {"schema_version": 1, "season": season, "through_gw": int(rounds[-1]), "windows": {k: [int(x) for x in v] for k, v in windows.items()},
           "snapshot_utc": snap, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "notes": {"big": f"shots with Understat xG >= {BIG_CHANCE}", "kp": "shots set up (Understat assister field)",
                     "fx": "FPL's own expected stats (Opta)", "u": "Understat expected stats",
                     "dc_hits": "matches reaching the defensive-contribution threshold (DEF 10, MID/FWD 12)",
                     "ratings": "the model's pre-match attack/defence ratings and expected goals per fixture"},
           "players": players, "teams": teams}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stats_{season}.json").write_text(json.dumps(doc, separators=(",", ":")))
    tot = sum(p["s"]["season"]["g"] for p in players)
    print(f"wrote stats_{season}.json: {len(players)} players, {len(teams)} teams, through GW{rounds[-1]}; "
          f"goals {tot:.0f}, shots on maps {sum(len(p['shots']) for p in players)}")


if __name__ == "__main__":
    main()
