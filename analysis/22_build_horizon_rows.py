#!/usr/bin/env python3
"""Horizon rows: every player-fixture predicted from an EARLIER gameweek deadline (horizon backtest).

For deadline t (gameweek t of a season) and h = 1..H, a row for each player's fixture in gameweek t + h - 1,
with every feature as of deadline t. h = 1 is the usual evaluation row (master.parquet).

CONSTRUCTION (no new feature code; the master table already holds the state at every deadline)
  anchor   the player's first master row kicking off at or after deadline t, in the same season and the same
           team; it must be in gameweek t, or later only if the team has no fixture in between (blank
           gameweek). Its features are exactly the state at deadline t: the rows before it are the rows before
           the deadline (the double-gameweek rule of 06/12 generalised; no player plays twice within 36 h,
           so the Understat as-of time adds nothing between the deadline and the anchor's kickoff).
           Players with no anchor (not registered with the team at deadline t) get no rows.
  copied   from the anchor: recent-minutes and start-history counts, season / previous-season minutes,
           price and ownership (previous gameweek's).
  clock    days since last appearance: anchor value + days from the anchor's kickoff.
  decayed  time-decayed FPL sums (365 d): anchor value x 0.5^(days / half-life), i.e. the deadline state
           decayed on to the target kickoff (as 12 does for double gameweeks).
  Understat recomputed with 12's own as-of join: last Understat state before min(target kickoff - 36 h,
           deadline t), decayed to the target kickoff (for h > 1 that is the deadline itself).
  fixture  the target row's schedule context (home, rest days, gap to the next fixture, fixtures in the
           gameweek, season progress, rule eras) from the final fixture list. Limitation: a fixture moved
           after deadline t is placed where it was eventually played.
  ratings  team ratings and expected goals from the snapshot at deadline t
           (data/ratings_eval/fpl_team_fixture_ratings_asof.csv, 17_build_asof_ratings.py --horizon).
  squad    price rank, price gap, ownership share, minutes rank and active players in the position,
           recomputed within each (deadline, team-fixture) among the players anchored at t. Limitation:
           players who left the club before the target fixture have no target row and drop out of the group.
  outcomes the target row's (minutes, points, events, started): used only for evaluation.

CHECK: rows with h = 1 must equal master.parquet (relative tolerance 1e-9 for decayed sums, exact otherwise); the script stops if they do not.
Output: data/analysis/horizon_rows.parquet (starters only unless --all), one row per (asof_round, player,
fixture). Options: --horizon N (default 8), --seasons 2019-20,2020-21,... (default 2019-20..2025-26: the
current season is excluded because its fixtures played after the last Understat refresh are not in the
crosswalk and carry current-state ratings in the master table, so they cannot be rated as of a deadline).
"""
import sys
from pathlib import Path

import importlib.util

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("b12", ROOT / "analysis" / "12_build_master.py")
B12 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(B12)
OUT = ROOT / "data" / "analysis"
RATINGS = ROOT / "data" / "ratings_eval"

COPY = ["value", "selected", "m1", "m3", "m5", "m10", "f_play5", "f_long5", "f_full5", "ewm3", "absent_streak",
        "fix_since_app", "std_mpf", "prev_season_mpf", "career_apps", "n_prior_rows", "st_n10", "st_full5",
        "st_mean5", "st_since"]
CLOCK = ["days_since_app"]
RATING = ["team_att", "team_def", "opp_att", "opp_def", "xg_for_pi", "xg_against_pi", "params_label"]
SQUAD = ["price_rank", "price_gap_top", "own_share_team", "m5_rank_pos", "n_active_pos"]


def decay_cols(cols):
    out = {}
    for c in cols:
        for hl in (365,):
            if c.startswith(f"L{hl}_"):
                out[c] = hl
    return out


def squad_features(d):
    """As 06_build_minutes_dataset.py, within (deadline, team-fixture, position)."""
    grp = ["asof_round", "season", "fixture", "was_home", "position"]
    d["price_rank"] = d.groupby(grp).value.rank(ascending=False, method="min")
    d["price_gap_top"] = d.groupby(grp).value.transform("max") - d.value
    team = ["asof_round", "season", "fixture", "was_home"]
    d["own_share_team"] = d.selected / d.groupby(team).selected.transform("sum").clip(lower=1)
    d["m5_rank_pos"] = d.groupby(grp).m5.rank(ascending=False, method="min")
    d["n_active_pos"] = d.f_play5.gt(0).groupby([d[c] for c in grp]).transform("sum").astype("int64")
    return d


def season_rows(m, rt, ra, season, H, us_states, rounds=None):
    """Horizon rows of one season. m: master rows of the season with team id; rt: fixture schedule;
    rounds: the deadlines (gameweeks) to predict from (default every gameweek in m)."""
    m = m.sort_values(["code", "kickoff_time", "fixture"]).reset_index(drop=True)
    m["mrow"] = np.arange(len(m))
    deadlines = m.groupby("round").deadline.first()
    if rounds is not None:
        deadlines = deadlines[deadlines.index.isin(rounds)]
    team_rounds = rt.groupby("fpl_team_id").event.apply(lambda x: np.sort(x.unique())).to_dict()
    dcols = decay_cols(m.columns)
    pieces = []
    for t, dl in deadlines.items():
        # anchor: first row at or after the deadline, per player
        later = m[m.kickoff_time >= dl]
        anc = later.groupby("code", sort=False).head(1)
        # registered with that team at t: anchor in gameweek t, or later only across the team's blank gameweeks
        nxt = anc.team_id.map(lambda tm: next((r for r in team_rounds.get(tm, []) if r >= t), None))
        anc = anc[(anc["round"] >= t) & (anc["round"].to_numpy() == nxt.to_numpy())]
        tg = later[later["round"].between(t, t + H - 1)]
        tg = tg.merge(anc[["code", "team_id", "mrow", "kickoff_time"]].rename(
            columns={"mrow": "anchor_row", "kickoff_time": "anchor_kick"}), on=["code", "team_id"])
        if tg.empty:
            continue
        a = m.loc[tg.anchor_row.to_numpy()]
        days = ((tg.kickoff_time - tg.anchor_kick).dt.total_seconds() / 86400).to_numpy()
        for c in COPY:
            tg[c] = a[c].to_numpy()
        # the clock in 06's arithmetic (days = ns / 8.64e13), so values match the master table bit for bit
        clock = (tg.kickoff_time.astype("int64").to_numpy() / 8.64e13) - (tg.anchor_kick.astype("int64").to_numpy() / 8.64e13)
        for c in CLOCK:
            tg[c] = a[c].to_numpy() + clock
        for c, hl in dcols.items():
            tg[c] = a[c].to_numpy() * 0.5 ** (days / hl)
        tg["asof_round"], tg["asof_deadline"], tg["horizon"] = t, dl, tg["round"] - t + 1
        tg["deadline"] = dl
        pieces.append(tg)
    d = pd.concat(pieces, ignore_index=True)
    # Understat form as of min(target kickoff - 36 h, deadline t), exactly as 12_build_master.py
    us_cols = ["us_player_id", "us_days_since"] + [f"L{hl}_{c}" for hl in B12.HLS for c in B12.US_STATS]
    d["row_id"] = np.arange(len(d))
    U = B12.asof_understat(d, us_states)
    d = d.drop(columns=us_cols).merge(U, on="row_id", how="left")
    # ratings as of the deadline
    r_ = ra[ra.season == season][["fixture", "was_home", "asof_round"] + RATING]
    d = d.drop(columns=RATING).merge(r_, on=["fixture", "was_home", "asof_round"], how="left")
    d["supremacy"] = d.xg_for_pi - d.xg_against_pi
    d["total_goals"] = d.xg_for_pi + d.xg_against_pi
    return squad_features(d)


def check_h1(d, m):
    """h = 1 rows must reproduce the master table."""
    h1 = d[d.horizon == 1]
    k = ["season", "element", "fixture"]
    j = h1.merge(m, on=k, suffixes=("", "_m"))
    probs = []
    if len(j) != len(m):
        probs.append(f"h=1 rows {len(j):,} vs master rows {len(m):,}")
    # decayed sums of a double gameweek's second fixture are computed by a different but equivalent route
    tol = set(decay_cols(m.columns)) | {"us_days_since"} | {c for c in m.columns if c[:5] in ("L180_", "L540_")}
    for c in m.columns:
        if c in k or c == "row_id" or c + "_m" not in j:
            continue
        x, y = j[c].to_numpy(), j[c + "_m"].to_numpy()
        if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            x, y = x.astype(float), y.astype(float)
            ok = (np.isnan(x) & np.isnan(y)) | ((np.abs(x - y) <= 1e-9 * np.maximum(1, np.abs(y))) if c in tol
                                                else (x == y))
        else:
            ok = (pd.isna(x) & pd.isna(y)) | (x == y)
        if not ok.all():
            probs.append(f"{c}: {(~ok).sum()} of {len(ok):,} differ")
    return probs


def main():
    args = sys.argv[1:]
    H = int(args[args.index("--horizon") + 1]) if "--horizon" in args else 8
    keep_all = "--all" in args
    master = pd.read_parquet(OUT / "master.parquet")
    rt = pd.read_csv(RATINGS / "fpl_team_fixture_ratings.csv")
    rt["was_home"] = rt.was_home.astype(str).eq("True")
    ra = pd.read_csv(RATINGS / "fpl_team_fixture_ratings_asof.csv")
    ra["was_home"] = ra.was_home.astype(str).eq("True")
    seasons = (args[args.index("--seasons") + 1].split(",") if "--seasons" in args
               else sorted(s for s in master.season.unique() if "2019-20" <= s <= "2025-26"))
    master = master.merge(rt[["season", "fixture", "was_home", "fpl_team_id"]].rename(columns={"fpl_team_id": "team_id"}),
                          on=["season", "fixture", "was_home"], how="left")
    assert master.team_id.notna().all(), "master rows without a team id"
    print("Understat states ...", flush=True)
    us_states = B12.understat_post_states()
    out, bad = [], False
    for s in seasons:
        m = master[master.season == s]
        d = season_rows(m, rt[rt.season == s], ra, s, H, us_states)
        probs = check_h1(d, m.drop(columns="team_id"))
        st = d[d.started == 1]
        print(f"{s}: {len(d):,} rows ({len(st):,} starters); by horizon "
              f"{d.groupby('horizon').size().to_dict()}; ratings missing {int(d.xg_for_pi.isna().sum())}; "
              f"h=1 vs master: {'IDENTICAL' if not probs else 'DIFFERENT'}", flush=True)
        for p in probs[:15]:
            print("    ", p)
        bad |= bool(probs)
        out.append(d if keep_all else st)
    if bad:
        sys.exit("h = 1 rows do not reproduce the master table; nothing written")
    R = pd.concat(out, ignore_index=True).drop(columns=["mrow", "anchor_row"])
    R.to_parquet(OUT / "horizon_rows.parquet", index=False)
    print(f"saved {len(R):,} rows -> {OUT / 'horizon_rows.parquet'}")


if __name__ == "__main__":
    main()
