#!/usr/bin/env python3
"""Master table for the xP model: ONE ROW PER FPL PLAYER PER FINISHED FIXTURE, zeros included.

Contents
  * every FPL points outcome (minutes, goals, assists, clean sheet, goals conceded, saves, bonus, bps,
    cards, own goals, penalties saved/missed, CBIT/CBIRT and defensive contribution, total points),
    FPL's own per-match xP (benchmark, 2020-21 on), and Understat same-match goal split (targets only)
  * minutes features (06_build_minutes_dataset.py)
  * FPL-history lags per player code (time-decayed, half-life 365 d, strictly earlier rows):
    yellow, red, bps, bonus, saves, own goals, penalties saved, CBIT, CBIRT, each with its own exposure
    (CBIT/CBIRT are only recorded in 2016-19 and 2025-26 on; exposure counts only those rows)
  * Understat form AS OF THE GAMEWEEK DEADLINE for every row (also non-appearances): decayed sums at
    180/540 days from the player's last Understat match before min(deadline, kickoff - 36 h), decayed to
    kickoff. EPL Understat matches are timed by FPL's exact kickoff.
    (Understat timestamps in early seasons are off by up to 12 h, so a plain `date < kickoff` join
    could pick up the current match; no player plays twice within 36 h.)
  * team ratings from the folder in $XFPL_RATINGS (evaluation uses hyperparameters fitted before
    2019-08-01, so the ratings carry no look-ahead into any test season)
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from config import DATA  # noqa: E402

sp = importlib.util.spec_from_file_location("b1", ROOT / "analysis" / "01_build_scorer_assist_dataset.py")
B1 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(B1)
OUT = DATA / "analysis"
RATINGS = Path(os.environ.get("XFPL_RATINGS", DATA / "ratings"))
US_STATS = ["time", "np_shots", "npxg", "xa", "xgchain", "xgbuildup", "np_goals", "pen_att", "pen_goals", "assists",
            "team_np_on", "team_shots_on", "team_pen_on", "n_app"]
HLS = (180, 540)
OUTCOMES = ["goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "bonus", "bps", "yellow_cards",
            "red_cards", "own_goals", "penalties_saved", "penalties_missed", "tackles", "clearances_blocks_interceptions",
            "recoveries", "defensive_contribution", "total_points", "xP", "starts"]


def understat_match_table():
    """One row per Understat player-match (all leagues) with the per-match statistics US_STATS."""
    pms, shots, matches = B1.load_understat()
    matches = matches[matches.isResult == True].drop_duplicates("match_id")  # noqa: E712
    s = shots[shots.result != "OwnGoal"].copy()
    s["pen"] = s.situation == "Penalty"
    s["goal"] = s.result == "Goal"
    s["npxg"] = np.where(s.pen, 0.0, s.xG)
    s["np_goal"] = s.goal & ~s.pen
    s["pen_goal"] = s.goal & s.pen
    s["np_shot"] = ~s.pen
    pl = s.groupby(["match_id", "player_id"]).agg(np_shots=("np_shot", "sum"), npxg=("npxg", "sum"),
                                                   np_goals=("np_goal", "sum"), pen_att=("pen", "sum"),
                                                   pen_goals=("pen_goal", "sum")).reset_index()
    tm = s.groupby(["match_id", "h_a"]).agg(team_npxg=("npxg", "sum"), team_shots=("np_shot", "sum"),
                                            team_pen=("pen", "sum")).reset_index()
    d = pms.merge(matches[["match_id", "date"]], on="match_id").merge(pl, on=["match_id", "player_id"], how="left") \
        .merge(tm, on=["match_id", "h_a"], how="left")
    for c in ("np_shots", "npxg", "np_goals", "pen_att", "pen_goals", "team_npxg", "team_shots", "team_pen"):
        d[c] = d[c].fillna(0.0)
    d = d.rename(columns={"xA": "xa", "xGChain": "xgchain", "xGBuildup": "xgbuildup"})
    d["team_np_on"] = d.team_npxg * d.time / 90
    d["team_shots_on"] = d.team_shots * d.time / 90
    d["team_pen_on"] = d.team_pen * d.time / 90
    d["n_app"] = 1.0
    # EPL match clock = FPL's exact kickoff (Understat times in early seasons are up to 12 h off)
    fxw = pd.read_csv(DATA / "crosswalk" / "fixture_crosswalk.csv", usecols=["us_match_id", "fpl_kick"],
                      parse_dates=["fpl_kick"]).drop_duplicates("us_match_id")
    fk = d.match_id.map(fxw.set_index("us_match_id").fpl_kick)
    d["date"] = np.where(fk.notna(), fk, d.date)
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["player_id", "date", "match_id"]).reset_index(drop=True)


def understat_post_states():
    """Per Understat player-match: decayed sums AFTER that match (pre-state + the match itself)."""
    d = understat_match_table()
    day = ((d.date - pd.Timestamp("2010-01-01")).dt.total_seconds() / 86400).to_numpy()
    X = d[US_STATS].to_numpy(float)
    out = d[["player_id", "date"]].copy()
    for hl in HLS:
        pre, _, _ = B1.decayed_pre(d.player_id.to_numpy(np.int64), day, X, float(hl))
        post = pre + X
        for j, c in enumerate(US_STATS):
            out[f"P{hl}_{c}"] = post[:, j]
    return out


def asof_understat(rows, states):
    cw = pd.read_csv(DATA / "crosswalk" / "player_crosswalk.csv", usecols=["code", "us_player_id"])
    r = rows[["row_id", "code", "kickoff_time", "deadline"]].merge(cw, on="code", how="left")
    # as of the gameweek deadline, and never within 36 h of kickoff
    r["asof"] = np.minimum(r.kickoff_time - pd.Timedelta(hours=36), r.deadline)
    has = r.us_player_id.notna()
    a = r[has].copy()
    a["us_player_id"] = a.us_player_id.astype(np.int64)
    st = states.rename(columns={"player_id": "us_player_id"}).sort_values("date")
    a = pd.merge_asof(a.sort_values("asof"), st, left_on="asof", right_on="date", by="us_player_id",
                      direction="backward", allow_exact_matches=False)
    dt_days = (a.kickoff_time - a.date).dt.total_seconds() / 86400
    for hl in HLS:
        f = 0.5 ** (dt_days / hl)
        for c in US_STATS:
            a[f"L{hl}_{c}"] = (a[f"P{hl}_{c}"] * f).fillna(0.0)
    a["us_days_since"] = dt_days
    keep = ["row_id", "us_player_id", "us_days_since"] + [f"L{hl}_{c}" for hl in HLS for c in US_STATS]
    out = r[["row_id"]].merge(a[keep], on="row_id", how="left")
    for c in keep[3:]:
        out[c] = out[c].fillna(0.0)
    return out


def fpl_lags(d):
    """Time-decayed (365 d) sums of FPL event counts over each player's strictly earlier rows."""
    d = d.sort_values(["code", "kickoff_time", "fixture"]).copy()
    obs_dc = d.tackles.notna() & d.clearances_blocks_interceptions.notna()
    m = d.minutes.fillna(0).to_numpy(float)
    cbit = (d.clearances_blocks_interceptions.fillna(0) + d.tackles.fillna(0)).to_numpy(float)
    cbirt = cbit + d.recoveries.fillna(0).to_numpy(float)
    cols = {"f_min": m, "f_yel": d.yellow_cards.fillna(0).to_numpy(float), "f_red": d.red_cards.fillna(0).to_numpy(float),
            "f_bps": d.bps.fillna(0).to_numpy(float), "f_bonus": d.bonus.fillna(0).to_numpy(float),
            "f_saves": d.saves.fillna(0).to_numpy(float), "f_og": d.own_goals.fillna(0).to_numpy(float),
            "f_pensv": d.penalties_saved.fillna(0).to_numpy(float),
            "f_cbit": np.where(obs_dc, cbit, 0.0), "f_cbirt": np.where(obs_dc, cbirt, 0.0),
            "f_min_dc": np.where(obs_dc, m, 0.0)}
    X = np.column_stack(list(cols.values()))
    day = ((d.kickoff_time - pd.Timestamp("2010-01-01")).dt.total_seconds() / 86400).to_numpy()
    pre, _, _ = B1.decayed_pre(d.code.to_numpy(np.int64), day, X, 365.0)
    for j, c in enumerate(cols):
        d[f"L365_{c}"] = pre[:, j]
    # double gameweek: the second fixture must not see the first (played after the deadline) -> use the
    # first fixture's pre-state, decayed on to this kickoff
    g = d.groupby(["code", "season", "round"], sort=False)
    later = (g.cumcount() > 0).to_numpy()
    if later.any():
        t_first = g.kickoff_time.transform("first")
        f = 0.5 ** (((d.kickoff_time - t_first).dt.total_seconds() / 86400) / 365.0)
        for c in cols:
            col = f"L365_{c}"
            first = g[col].transform("first")
            d.loc[later, col] = (first * f)[later]
    return d


def main():
    md = pd.read_parquet(OUT / "minutes_dataset.parquet")
    md = md.drop(columns=[c for c in ("L180_time", "L540_time") if c in md.columns])
    pm = pd.read_csv(DATA / "fpl" / "fpl_player_match.csv", low_memory=False,
                     usecols=["season", "element", "fixture", "kickoff_time", "position"] + OUTCOMES)
    pm = pm[pm.position.isin(["GKP", "DEF", "MID", "FWD"])].drop(columns="position")
    pm["kickoff_time"] = pd.to_datetime(pm.kickoff_time, utc=True).dt.tz_localize(None)
    pm = pm.sort_values("kickoff_time", ascending=False).drop_duplicates(["season", "element", "fixture"])
    d = md.merge(pm.drop(columns="kickoff_time"), on=["season", "element", "fixture"], how="left")
    assert len(d) == len(md)
    d = fpl_lags(d)
    d = d.reset_index(drop=True)
    d["row_id"] = np.arange(len(d))
    print("Understat as-of features ...", flush=True)
    U = asof_understat(d, understat_post_states())
    d = d.merge(U, on="row_id", how="left")
    # team ratings (evaluation set) replace whatever the minutes table carried
    rt = pd.read_csv(RATINGS / "fpl_team_fixture_ratings.csv")
    rt["was_home"] = rt.was_home.astype(str).eq("True")
    rcols = ["team_att", "team_def", "opp_att", "opp_def", "xg_for_pi", "xg_against_pi", "p_win", "p_draw", "p_loss",
             "p_clean_sheet", "params_label"]
    d = d.drop(columns=[c for c in rcols + ["supremacy", "total_goals"] if c in d.columns])
    d = d.merge(rt[["season", "fixture", "was_home"] + rcols], on=["season", "fixture", "was_home"], how="left")
    d["supremacy"] = d.xg_for_pi - d.xg_against_pi
    d["total_goals"] = d.xg_for_pi + d.xg_against_pi
    # Understat same-match goal split (targets only, appearance rows)
    tg = pd.read_parquet(OUT / "scorer_assist_dataset.parquet",
                         columns=["season", "fixture", "element", "tgt_np_goals", "tgt_pen_goals", "tgt_pen_att"])
    d = d.merge(tg, on=["season", "fixture", "element"], how="left")
    d["sealed"] = d.season >= "2025-26"
    d.to_parquet(OUT / "master.parquet", index=False)
    print(f"{len(d):,} rows, {d.shape[1]} cols; ratings: {d.params_label.dropna().unique()}")
    print("rows without ratings:", int(d.xg_for_pi.isna().sum()), "| rows with Understat id:",
          f"{d.us_player_id.notna().mean():.3f}", "| appearance rows with Understat id:",
          f"{d.us_player_id[d.minutes > 0].notna().mean():.4f}")


if __name__ == "__main__":
    main()
