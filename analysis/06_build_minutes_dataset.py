#!/usr/bin/env python3
"""Build the minutes-component dataset: one row per FPL player per finished fixture, INCLUDING
zero-minute rows (the 'did not play' outcome is exactly what xP needs).

TARGET: minutes (0-90), later bucketed. FEATURES, all known at the FPL deadline (no team sheets):
  Everything is AS OF THE GAMEWEEK DEADLINE: the second fixture of a double gameweek gets the first
  fixture's features (the first one is played after the deadline).
  Recent usage, from the player's own FPL history (rows for every team fixture, zeros included):
    minutes in the last 1/3/5/10 fixtures, share of last 5 with any minutes / >=60 / full 90,
    consecutive absent fixtures, fixtures and calendar days since last appearance,
    exponentially weighted minutes, season-to-date minutes per fixture, previous-season minutes
    per fixture (NaN-flagged for newcomers), career FPL appearances.
  Price and ownership are taken from the player's PREVIOUS gameweek (see main).
  Squad competition (position-relative): price rank, price gap to the dearest teammate in the
    position, ownership share of the team, rank of recent minutes among teammates in the position.
  Fixture context: home, team/opponent rest days, days to the team's NEXT fixture (fixture lists
    are known at the deadline), fixtures in the same gameweek (double gameweeks), team
    supremacy/total from the pi-ratings, win probability, season progress.
  Understat history (any league) for newcomers: decayed minutes at 180/540 days.
NOT AVAILABLE historically: injury/availability news (the API keeps no history; snapshots are
being collected from 2026-10-11 and will be added as features once a season has accrued).
Rows are NOT filtered to a universe here; the evaluation script defines it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from config import BCD_PERIODS, DATA, FIVE_SUB_PERIODS_EPL, in_periods  # noqa: E402

OUT = DATA / "analysis"


def per_player_sequences(d):
    """Lag features computed per player code in kickoff order; everything uses rows STRICTLY before."""
    d = d.sort_values(["code", "kickoff_time", "fixture"]).reset_index(drop=True)
    code = d.code.to_numpy()
    m = d.minutes.to_numpy(float)
    t = d.kickoff_time.astype("int64").to_numpy() / 8.64e13        # days
    season = d.season.to_numpy()
    rnd = d["round"].to_numpy()
    n = len(d)
    cols = {k: np.full(n, np.nan) for k in ("m1", "m3", "m5", "m10", "f_play5", "f_long5", "f_full5", "ewm3",
                                            "absent_streak", "fix_since_app", "days_since_app", "std_mpf",
                                            "prev_season_mpf", "career_apps", "n_prior_rows")}
    start = 0
    while start < n:
        end = start
        while end < n and code[end] == code[start]:
            end += 1
        mm, tt, ss, rr = m[start:end], t[start:end], season[start:end], rnd[start:end]
        ew = np.nan
        streak = 0
        last_app_i = -1
        apps = 0
        sums_by_season = {}
        for j in range(end - start):
            i = start + j
            if j > 0 and ss[j] == ss[j - 1] and rr[j] == rr[j - 1]:
                # second fixture of a double gameweek: features are AS OF THE DEADLINE, so they are the
                # first fixture's features (only the clock-based gap moves on); then update state as usual
                for k_ in cols:
                    cols[k_][i] = cols[k_][i - 1]
                cols["days_since_app"][i] = cols["days_since_app"][i - 1] + (tt[j] - tt[j - 1])
                ew = mm[j] if np.isnan(ew) else 0.5 ** (1 / 3) * ew + (1 - 0.5 ** (1 / 3)) * mm[j]
                streak = streak + 1 if mm[j] == 0 else 0
                s_sum, s_n = sums_by_season.get(ss[j], (0.0, 0))
                if mm[j] > 0:
                    last_app_i = j
                    apps += 1
                sums_by_season[ss[j]] = (s_sum + mm[j], s_n + 1)
                continue
            lo = max(0, j - 10)
            hist = mm[lo:j]
            cols["n_prior_rows"][i] = j
            if j > 0:
                cols["m1"][i] = mm[j - 1]
                cols["m3"][i] = mm[max(0, j - 3):j].mean()
                cols["m5"][i] = mm[max(0, j - 5):j].mean()
                cols["m10"][i] = hist.mean()
                h5 = mm[max(0, j - 5):j]
                cols["f_play5"][i] = (h5 > 0).mean()
                cols["f_long5"][i] = (h5 >= 60).mean()
                cols["f_full5"][i] = (h5 >= 90).mean()
                cols["ewm3"][i] = ew
            cols["absent_streak"][i] = streak
            cols["fix_since_app"][i] = (j - last_app_i) if last_app_i >= 0 else np.nan
            cols["days_since_app"][i] = (tt[j] - tt[last_app_i]) if last_app_i >= 0 else np.nan
            s_sum, s_n = sums_by_season.get(ss[j], (0.0, 0))
            cols["std_mpf"][i] = s_sum / s_n if s_n else np.nan
            prev = [v for k, v in sums_by_season.items() if k < ss[j] and v[1] > 0]
            if prev:
                k_prev = max(k for k, v in sums_by_season.items() if k < ss[j] and v[1] > 0)
                ps, pn = sums_by_season[k_prev]
                cols["prev_season_mpf"][i] = ps / pn
            cols["career_apps"][i] = apps
            # update with the current row (becomes history for the NEXT row only)
            ew = mm[j] if np.isnan(ew) else 0.5 ** (1 / 3) * ew + (1 - 0.5 ** (1 / 3)) * mm[j]
            streak = streak + 1 if mm[j] == 0 else 0
            if mm[j] > 0:
                last_app_i = j
                apps += 1
            sums_by_season[ss[j]] = (s_sum + mm[j], s_n + 1)
        start = end
    for k, v in cols.items():
        d[k] = v
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pm = pd.read_csv(DATA / "fpl" / "fpl_player_match.csv", low_memory=False,
                     usecols=["season", "round", "element", "code", "position", "fixture", "was_home", "minutes",
                              "value", "selected", "kickoff_time", "fixture_finished"])
    pm = pm[pm.position.isin(["GKP", "DEF", "MID", "FWD"]) & pm.fixture_finished].copy()
    pm["was_home"] = pm.was_home.astype(str).eq("True")
    pm["kickoff_time"] = pd.to_datetime(pm.kickoff_time, utc=True).dt.tz_localize(None)
    # a postponed fixture appears twice (zero-minute row in the original gameweek, real row in the
    # rescheduled one). Keep the LATEST kickoff, i.e. the occurrence actually played. (Choosing by
    # minutes, as an earlier version did, selects rows using the outcome.)
    pm = pm.sort_values("kickoff_time", ascending=False).drop_duplicates(["season", "element", "fixture"]).copy()
    # gameweek deadline: first kickoff of the gameweek - 90 minutes (FPL's rule since 2019; earlier 60 min)
    pm["deadline"] = pm.groupby(["season", "round"]).kickoff_time.transform("min") - pd.Timedelta(minutes=90)
    # PRICE AND OWNERSHIP FROM THE PREVIOUS GAMEWEEK. The gameweek's own `value`/`selected` may be recorded
    # after the deadline, so each row gets the player's values from his previous gameweek in the season.
    # Fallbacks where none exists: price = the listed price (published before the deadline); ownership =
    # the player's final ownership of the previous season, else 0.
    pm["value_raw"], pm["selected_raw"] = pm.value, pm.selected
    gw = (pm.sort_values(["code", "season", "round", "kickoff_time"])
            .groupby(["code", "season", "round"], as_index=False)[["value", "selected"]].first())
    gw[["value_prev", "selected_prev"]] = gw.groupby(["code", "season"])[["value", "selected"]].shift(1)
    last_sel = gw.groupby(["code", "season"]).selected.last().reset_index()
    seasons = sorted(pm.season.unique())
    last_sel["season"] = last_sel.season.map({s_: seasons[i + 1] for i, s_ in enumerate(seasons[:-1])})
    gw = gw.merge(last_sel.rename(columns={"selected": "sel_prev_season"}), on=["code", "season"], how="left")
    gw["value_prev"] = gw.value_prev.fillna(gw.value)
    gw["selected_prev"] = gw.selected_prev.fillna(gw.sel_prev_season).fillna(0)
    pm = pm.drop(columns=["value", "selected"]).merge(
        gw[["code", "season", "round", "value_prev", "selected_prev"]].rename(
            columns={"value_prev": "value", "selected_prev": "selected"}), on=["code", "season", "round"], how="left")
    d = per_player_sequences(pm)
    # rule/era context known before every match: 5-substitute rule, closed doors (COVID)
    d["five_subs"] = in_periods(d.kickoff_time, FIVE_SUB_PERIODS_EPL).astype(float)
    d["bcd"] = in_periods(d.kickoff_time, BCD_PERIODS["EPL"]).astype(float)

    # ---- team/fixture context from the pi-ratings table (covers every fixture, incl. future ones)
    import os
    rt = pd.read_csv(Path(os.environ.get("XFPL_RATINGS", DATA / "ratings")) / "fpl_team_fixture_ratings.csv")
    rt["kickoff_time"] = pd.to_datetime(rt.kickoff_time, utc=True).dt.tz_localize(None)
    rt["was_home"] = rt.was_home.astype(str).eq("True")
    rt = rt.sort_values(["season", "fpl_team_id", "kickoff_time"])
    g = rt.groupby(["season", "fpl_team_id"]).kickoff_time
    rt["rest_days"] = (rt.kickoff_time - g.shift(1)).dt.total_seconds() / 86400
    rt["next_gap_days"] = (g.shift(-1) - rt.kickoff_time).dt.total_seconds() / 86400
    rt["fix_in_event"] = rt.groupby(["season", "fpl_team_id", "event"]).fixture.transform("size")
    opp = rt[["season", "fixture", "was_home", "rest_days"]].copy()
    opp["was_home"] = ~opp.was_home
    rt = rt.merge(opp.rename(columns={"rest_days": "opp_rest_days"}), on=["season", "fixture", "was_home"], how="left")
    keep = ["season", "fixture", "was_home", "xg_for_pi", "xg_against_pi", "p_win", "p_draw", "p_loss",
            "rest_days", "opp_rest_days", "next_gap_days", "fix_in_event"]
    d = d.merge(rt[keep], on=["season", "fixture", "was_home"], how="left")
    d["supremacy"] = d.xg_for_pi - d.xg_against_pi
    d["total_goals"] = d.xg_for_pi + d.xg_against_pi
    d["season_progress"] = d["round"] / 38.0

    # ---- squad competition (within team-fixture, within position)
    grp = ["season", "fixture", "was_home", "position"]
    d["price_rank"] = d.groupby(grp).value.rank(ascending=False, method="min")
    d["price_gap_top"] = d.groupby(grp).value.transform("max") - d.value
    team = ["season", "fixture", "was_home"]
    d["own_share_team"] = d.selected / d.groupby(team).selected.transform("sum").clip(lower=1)
    d["m5_rank_pos"] = d.groupby(grp).m5.rank(ascending=False, method="min")
    d["n_active_pos"] = d.groupby(grp).f_play5.transform(lambda x: (x > 0).sum())

    # (Understat history is attached as-of kickoff in 12_build_master.py; joining the appearance-only
    #  scorer table here leaked whether the player appeared.)
    d.to_parquet(OUT / "minutes_dataset.parquet", index=False)
    print(f"{len(d):,} rows, {d.shape[1]} columns -> {OUT/'minutes_dataset.parquet'}")
    print("share zero-minute rows:", round((d.minutes == 0).mean(), 3))
    print("rows missing fixture context:", int(d.xg_for_pi.isna().sum()))


if __name__ == "__main__":
    main()
