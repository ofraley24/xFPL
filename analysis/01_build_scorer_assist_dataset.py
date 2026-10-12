#!/usr/bin/env python3
"""Build the player-match modelling table for the goals and assists models.

One row per FPL appearance (minutes > 0) in a finished EPL fixture, 2016-17 onward.

TARGETS (FPL definitions): goals_scored, assists.   EXPOSURE: minutes.

FEATURES, all strictly PRE-MATCH:
  Player form from Understat, ALL six leagues (Understat ids are global, so a player's history
  follows him across leagues). For two half-lives (180 and 540 days) the decayed SUMS of
  minutes, non-penalty shots, npxG, xA, xGChain, xGBuildup, non-penalty goals, penalty
  attempts and goals, assists, and "team npxG while he was on the pitch". Sums, not rates, are
  stored so empirical-Bayes shrinkage can be applied later with the exposure visible.
  Also: appearances before this match, days since his previous appearance.
  Team context: attack/defence pi-ratings and expected goals for/against for this fixture
  (fpl_team_fixture_ratings.csv).
  Market proxies available at the deadline: FPL price (`value`) and ownership (`selected`).

NOT USED as features, because they are only known after the deadline: starting/bench status of
THIS match (Understat `position`) and anything from the match itself. They are kept out so the
tables evaluate a pre-deadline model. Minutes are treated as exposure, i.e. the rate model is
evaluated conditional on playing; the minutes model is a separate component.
"""
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from config import DATA, LEAGUES  # noqa: E402

US = DATA / "understat_clean"
OUT = DATA / "analysis"
HALF_LIVES = (180, 540, 1080)
STATS = ["time", "np_shots", "npxg", "xa", "xgchain", "xgbuildup", "np_goals", "pen_att", "pen_goals",
         "assists", "team_np_on", "team_shots_on", "team_pen_on", "n_app", "r_gk", "r_def", "r_dm", "r_mid", "r_am", "r_fw", "r_sub"]
ROLE = {"GK": "r_gk", "DC": "r_def", "DL": "r_def", "DR": "r_def", "DMC": "r_dm", "DML": "r_dm", "DMR": "r_dm",
        "MC": "r_mid", "ML": "r_mid", "MR": "r_mid", "AMC": "r_am", "AML": "r_am", "AMR": "r_am",
        "FW": "r_fw", "FWL": "r_fw", "FWR": "r_fw", "Sub": "r_sub"}


@numba.njit(cache=False)
def decayed_pre(pid, day, X, hl):
    """Time-decayed sum of X over each player's STRICTLY EARLIER rows (rows pre-sorted by pid, day)."""
    n, k = X.shape
    out = np.zeros((n, k))
    cnt = np.zeros(n)
    gap = np.full(n, np.nan)
    S = np.zeros(k)
    cur = -1
    last = 0.0
    c = 0
    for i in range(n):
        if pid[i] != cur:
            cur = pid[i]
            S[:] = 0.0
            last = day[i]
            c = 0
        else:
            d = day[i] - last
            gap[i] = d
            if d > 0:
                S *= 0.5 ** (d / hl)
        out[i] = S
        cnt[i] = c
        S += X[i]
        last = day[i]
        c += 1
    return out, cnt, gap


def load_understat():
    pms, shots, matches = [], [], []
    for lg in LEAGUES:
        pms.append(pd.read_csv(US / f"{lg}_player_match_stats.csv", low_memory=False)
                   .assign(league=lg))
        shots.append(pd.read_csv(US / f"{lg}_shots.csv", usecols=["match_id", "player_id", "h_a", "situation",
                                                                  "result", "xG"], low_memory=False))
        matches.append(pd.read_csv(US / f"{lg}_matches.csv", usecols=["match_id", "date", "isResult"],
                                   parse_dates=["date"]))
    return pd.concat(pms), pd.concat(shots), pd.concat(matches)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pms, shots, matches = load_understat()
    matches = matches[matches.isResult == True].drop_duplicates("match_id")  # noqa: E712

    s = shots[shots.result != "OwnGoal"].copy()
    s["pen"] = s.situation == "Penalty"
    s["goal"] = s.result == "Goal"
    s["npxg"] = np.where(s.pen, 0.0, s.xG)
    pl = s.groupby(["match_id", "player_id"]).apply(lambda g: pd.Series({
        "np_shots": (~g.pen).sum(), "npxg": g.npxg.sum(), "np_goals": (g.goal & ~g.pen).sum(),
        "pen_att": g.pen.sum(), "pen_goals": (g.goal & g.pen).sum()})).reset_index()
    team = s.groupby(["match_id", "h_a"]).agg(team_npxg=("npxg", "sum"),
                                              team_shots=("pen", lambda x: (~x).sum()),
                                              team_pen=("pen", "sum")).reset_index()

    d = pms.merge(matches[["match_id", "date"]], on="match_id", how="inner")
    d = d.merge(pl, on=["match_id", "player_id"], how="left").merge(team, on=["match_id", "h_a"], how="left")
    for c in ("np_shots", "npxg", "np_goals", "pen_att", "pen_goals", "team_npxg", "team_shots", "team_pen"):
        d[c] = d[c].fillna(0.0)
    d = d.rename(columns={"xA": "xa", "xGChain": "xgchain", "xGBuildup": "xgbuildup"})
    d["team_np_on"] = d.team_npxg * d.time / 90.0
    d["team_shots_on"] = d.team_shots * d.time / 90.0
    d["team_pen_on"] = d.team_pen * d.time / 90.0
    # lagged ROLE: where he started (Understat position), and how often he came off the bench.
    # Built from PAST appearances only (the decayed sums below exclude the current row).
    d["n_app"] = 1.0
    for c in set(ROLE.values()):
        d[c] = (d.position.map(ROLE) == c).astype(float)
    d = d.sort_values(["player_id", "date", "match_id"]).reset_index(drop=True)
    day = ((d.date - pd.Timestamp("2010-01-01")).dt.total_seconds() / 86400).to_numpy()
    X = d[STATS].to_numpy(float)
    pid = d.player_id.to_numpy(np.int64)
    feats = {}
    for hl in HALF_LIVES:
        out, cnt, gap = decayed_pre(pid, day, X, float(hl))
        for j, name in enumerate(STATS):
            feats[f"L{hl}_{name}"] = out[:, j]
    feats["n_prev_apps"] = cnt
    feats["days_since_last"] = gap
    # same-match outcomes from Understat, used ONLY as targets for the penalty / open-play split
    tgt = d[["np_goals", "pen_goals", "pen_att"]].rename(columns=lambda c: f"tgt_{c}").reset_index(drop=True)
    U = pd.concat([d[["match_id", "player_id"]].reset_index(drop=True), tgt, pd.DataFrame(feats)], axis=1)
    print(f"Understat player-match rows with pre-match form: {len(U):,}")

    # ---- FPL side
    pm = pd.read_csv(DATA / "fpl" / "fpl_player_match.csv", low_memory=False,
                     usecols=["season", "round", "element", "code", "position", "fixture", "was_home", "minutes",
                              "goals_scored", "assists", "value", "selected", "total_points", "fixture_finished",
                              "kickoff_time"])
    pm = pm[(pm.minutes > 0) & pm.fixture_finished & pm.position.isin(["GKP", "DEF", "MID", "FWD"])].copy()
    pm["was_home"] = pm.was_home.astype(str).eq("True")
    cw = pd.read_csv(DATA / "crosswalk" / "player_match_crosswalk.csv",
                     usecols=["season", "fixture", "element", "us_match_id", "us_player_id"])
    pm = pm.merge(cw, on=["season", "fixture", "element"], how="inner")
    pm = pm.merge(U, left_on=["us_match_id", "us_player_id"], right_on=["match_id", "player_id"], how="left")
    rt = pd.read_csv(DATA / "ratings" / "fpl_team_fixture_ratings.csv")
    rt["was_home"] = rt.was_home.astype(str).eq("True")
    keep = ["season", "fixture", "was_home", "team_att", "team_def", "opp_att", "opp_def", "xg_for_pi",
            "xg_against_pi", "team_n_prev", "opp_n_prev"]
    pm = pm.merge(rt[keep], on=["season", "fixture", "was_home"], how="left")
    pm["sealed"] = pm.season >= "2025-26"  # protocol: 2025-26 is the sealed holdout, 2026-27 is live
    pm = pm.drop(columns=["match_id", "player_id"])
    pm.to_parquet(OUT / "scorer_assist_dataset.parquet", index=False)
    g = pm.groupby("season").agg(rows=("element", "size"), goals=("goals_scored", "sum"), assists=("assists", "sum"),
                                 no_form=("L540_time", lambda x: (x.fillna(0) == 0).mean()))
    print(g.round(3).to_string())
    print(f"\n{len(pm):,} rows, {pm.shape[1]} columns -> {OUT/'scorer_assist_dataset.parquet'}")
    print("missing team ratings:", int(pm.xg_for_pi.isna().sum()))


if __name__ == "__main__":
    main()
