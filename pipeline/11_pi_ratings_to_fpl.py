#!/usr/bin/env python3
"""Attach dual (attack / defence) pi-ratings to FPL fixtures.

1. Replay model_v1's PiRatingModel over matches_unified.csv (all six leagues, chronological) with a
   FIXED parameter file, recording for every match the PRE-match state: attack and defence rating of
   both teams, league intercept, lambda / mu, plus staleness (matches previously seen, days since the
   team's last match). Optional online terms: closed-doors home advantage, scoring-level drift and
   home-advantage drift (see replay).
2. Assert, with all options off, that the replay reproduces walk_forward_predictions.csv (lambda, mu)
   to 1e-9, i.e. the ratings really are the ones behind the model's published predictions.
3. Snapshot the ratings after the last played match (`pi_ratings_current.csv`).
4. Build `fpl_team_fixture_ratings.csv`: ONE ROW PER TEAM PER FPL FIXTURE (2 rows per fixture),
   including fixtures not yet played. Played fixtures are rated from the snapshot taken at their
   gameweek DEADLINE; upcoming fixtures from the current state (`is_prediction = True`).
   Join to player rows on (season, fixture, was_home).

Row content for team T vs opponent O: own/opp attack and defence rating, expected goals for
(lambda_T) and against (lambda_O), staleness counters, parameter label.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import DATA, DOCS, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "model_v1"))
from pi_rating_model import PiRatingModel  # noqa: E402

SRC = DATA / "interim" / "merged" / "model_v1"
OUT = DATA / "ratings"
CW = DATA / "crosswalk"
FPL = DATA / "fpl"


def replay(matches, params, bcd=False, bcd_prior_n=50.0, drift_lr=0.0, ha_lr=0.0, snap_times=None):
    """Walk-forward replay. Options (all off by default -> identical to model_v1):
    bcd      home advantage in behind-closed-doors matches is estimated ONLINE from closed-doors matches
             already played (all leagues pooled), shrunk towards the current crowd value with a prior worth
             `bcd_prior_n` matches:  h_t = (n*h_hat + n0*h_crowd) / (n + n0),
             h_hat = log(sum home blend / sum home lambda_0) - log(sum away blend / sum away lambda_0),
             lambda_0 = rating-only rates (no home term).
    drift_lr league intercept follows the league's scoring level online:
             c_L <- c_L + drift_lr * ((blend_h + blend_a) - (lambda_h + lambda_a)).
    ha_lr    crowd home advantage follows the home-minus-away residual online, on crowd matches only:
             h <- h + ha_lr * ((blend_h - lambda_h) - (blend_a - lambda_a)).
    Shared quantities (intercepts, home advantage, closed-doors sums) are updated only from matches that
    kicked off STRICTLY EARLIER: updates from simultaneous kick-offs are held and applied together once
    the clock moves on. Team ratings update immediately (simultaneous matches never share a team).
    Match time = column `t` when present (FPL kickoff for EPL matches), else `date`.
    snap_times: sorted timestamps (e.g. gameweek deadlines); the state just before each is recorded and
    returned as model.snapshots {time: state} (see snapshot_state / lambdas_from_snapshot).
    """
    from collections import defaultdict
    from config import BCD_PERIODS, in_periods
    leagues = sorted(matches.league.unique())
    model = PiRatingModel(
        home_adv=params["home_adv"], scale=params["scale"], alpha=params["alpha"],
        lr_attack=params["lr_attack"], lr_defence=params["lr_defence"],
        league_intercepts={lg: params[f"c_{lg}"] for lg in leagues})
    h0 = params["home_adv"]
    tcol = matches["t"] if "t" in matches.columns else matches["date"]
    times = pd.to_datetime(tcol).to_numpy()
    is_bcd = np.zeros(len(matches), bool)   # always computed (reporting); only USED when bcd=True
    for lg, per in BCD_PERIODS.items():
        mk = (matches.league == lg).to_numpy()
        if per and mk.any():
            is_bcd[mk] = in_periods(matches.date[mk], per)
    st_ = {"Sh": 0.0, "Sa": 0.0, "Lh0": 0.0, "La0": 0.0, "nb": 0, "h": h0}
    pend = {"c": defaultdict(float), "h": 0.0, "Sh": 0.0, "Sa": 0.0, "Lh0": 0.0, "La0": 0.0, "nb": 0}

    def apply_pending():
        for lg, v in pend["c"].items():
            model.league_intercepts[lg] += v
        st_["h"] += pend["h"]
        for k in ("Sh", "Sa", "Lh0", "La0", "nb"):
            st_[k] += pend[k]
        pend["c"].clear()
        pend["h"] = 0.0
        for k in ("Sh", "Sa", "Lh0", "La0", "nb"):
            pend[k] = 0
    snaps = {}
    snap_times = list(pd.to_datetime(snap_times)) if snap_times is not None else []
    ks = 0
    cur_t = None
    seen, last = {}, {}
    rows = []
    for i, r in enumerate(matches.itertuples(index=False)):
        t = pd.Timestamp(times[i])
        while ks < len(snap_times) and snap_times[ks] <= t:
            apply_pending()
            snaps[snap_times[ks]] = snapshot_state(model, st_, bcd, bcd_prior_n)
            ks += 1
        if cur_t is not None and t > cur_t:
            apply_pending()
        cur_t = t
        st = model._state(r.league)
        st.ensure(r.home_team)
        st.ensure(r.away_team)
        ah, dh = st.get(r.home_team)
        aa, da = st.get(r.away_team)
        kh, ka = (r.league, r.home_team), (r.league, r.away_team)
        model.home_adv = bcd_home_adv(st_, bcd_prior_n) if (bcd and is_bcd[i]) else st_["h"]
        h_used = model.home_adv
        c_used = model.league_intercepts[r.league]
        lam, mu = model.update(r.league, r.home_team, r.away_team, r.home_goals, r.away_goals,
                               r.home_npxG, r.away_npxG)
        bh = model.alpha * r.home_npxG + (1 - model.alpha) * r.home_goals
        ba = model.alpha * r.away_npxG + (1 - model.alpha) * r.away_goals
        if bcd and is_bcd[i]:
            pend["Sh"] += bh; pend["Sa"] += ba; pend["Lh0"] += lam / np.exp(h_used); pend["La0"] += mu; pend["nb"] += 1
        elif ha_lr:
            pend["h"] += ha_lr * ((bh - lam) - (ba - mu))
        if drift_lr:
            pend["c"][r.league] += drift_lr * ((bh + ba) - (lam + mu))
        rows.append({
            "league": r.league, "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
            "home_att": ah, "home_def": dh, "away_att": aa, "away_def": da,
            "lam": lam, "mu": mu, "c_league": c_used, "home_adv_used": h_used, "bcd": bool(is_bcd[i]),
            "home_n_prev": seen.get(kh, 0), "away_n_prev": seen.get(ka, 0),
            "home_gap_days": (r.date - last[kh]).days if kh in last else np.nan,
            "away_gap_days": (r.date - last[ka]).days if ka in last else np.nan})
        for k in (kh, ka):
            seen[k] = seen.get(k, 0) + 1
            last[k] = r.date
    apply_pending()
    while ks < len(snap_times):
        snaps[snap_times[ks]] = snapshot_state(model, st_, bcd, bcd_prior_n)
        ks += 1
    model.home_adv = st_["h"]   # future fixtures are played with crowds
    model.snapshots = snaps
    model.shared_state = dict(st_)
    return model, pd.DataFrame(rows)


def bcd_home_adv(st_, n0):
    nb = st_["nb"]
    h_hat = (np.log(st_["Sh"] / st_["Lh0"]) - np.log(st_["Sa"] / st_["La0"])) if nb > 0 else st_["h"]
    return (nb * h_hat + n0 * st_["h"]) / (nb + n0)


def snapshot_state(model, st_, bcd, n0, league="EPL"):
    """Everything needed to price a fixture of `league` as of now (copies, not references)."""
    rs = model.ratings.get(league)
    return {"att": dict(rs.attack) if rs else {}, "def": dict(rs.defence) if rs else {},
            "c": model.league_intercepts[league], "h_crowd": st_["h"],
            "h_bcd": bcd_home_adv(st_, n0) if bcd else st_["h"], "scale": model.scale}


def lambdas_from_snapshot(snap, home, away, closed_doors):
    h = snap["h_bcd"] if closed_doors else snap["h_crowd"]
    ah, dh = snap["att"].get(home, 0.0), snap["def"].get(home, 0.0)
    aa, da = snap["att"].get(away, 0.0), snap["def"].get(away, 0.0)
    lam = np.exp(snap["c"] + h + (ah - da) / snap["scale"])
    mu = np.exp(snap["c"] + (aa - dh) / snap["scale"])
    return lam, mu, ah, dh, aa, da


def tidy(d, label):
    """Two rows (home, away) per fixture from one wide fixture row."""
    rows = []
    for r in d.itertuples(index=False):
        for home in (True, False):
            rows.append({
                "season": r.season, "fixture": r.fixture, "event": r.event, "kickoff_time": r.kickoff,
                "was_home": home, "fpl_team_id": r.home_id if home else r.away_id,
                "fpl_opp_id": r.away_id if home else r.home_id,
                "team_att": r.home_att if home else r.away_att, "team_def": r.home_def if home else r.away_def,
                "opp_att": r.away_att if home else r.home_att, "opp_def": r.away_def if home else r.home_def,
                "team_n_prev": r.home_n_prev if home else r.away_n_prev,
                "opp_n_prev": r.away_n_prev if home else r.home_n_prev,
                "team_gap_days": r.home_gap_days if home else r.away_gap_days,
                "opp_gap_days": r.away_gap_days if home else r.home_gap_days,
                "xg_for_pi": r.lam if home else r.mu, "xg_against_pi": r.mu if home else r.lam,
                "is_prediction": r.is_prediction, "params_label": label})
    return pd.DataFrame(rows)


def load_matches_with_clock():
    """matches_unified with a match clock `t`: FPL's exact kickoff for EPL matches (Understat times in early
    seasons are up to 12 h off), Understat's time elsewhere; sorted by (t, league, home_team)."""
    matches = pd.read_csv(SRC / "matches_unified.csv", parse_dates=["date"])
    fxw0 = pd.read_csv(CW / "fixture_crosswalk.csv", parse_dates=["us_date", "fpl_kick"])
    kmap = fxw0.set_index(["us_date", "us_home", "us_away"]).fpl_kick
    key = pd.MultiIndex.from_frame(matches[["date", "home_team", "away_team"]])
    fk = pd.Series(kmap.reindex(key).to_numpy(), index=matches.index)
    matches["t"] = pd.to_datetime(np.where((matches.league == "EPL") & fk.notna(), fk, matches.date))
    return matches.sort_values(["t", "league", "home_team"], kind="mergesort").reset_index(drop=True)


def gameweek_deadlines():
    """(deadline per (season, round), round per (season, fixture)). Deadline = first kickoff of the gameweek
    minus 90 minutes; the current season uses FPL's published deadline_time."""
    pmk = pd.read_csv(FPL / "fpl_player_match.csv", usecols=["season", "round", "fixture", "kickoff_time"],
                      low_memory=False)
    pmk["kickoff_time"] = pd.to_datetime(pmk.kickoff_time, utc=True).dt.tz_localize(None)
    fx_round = pmk.groupby(["season", "fixture"])["round"].agg(lambda x: x.mode().iloc[0])
    fx_kick = pmk.groupby(["season", "fixture"]).kickoff_time.max()
    gw = pd.DataFrame({"round": fx_round, "kick": fx_kick}).reset_index()
    deadlines = (gw.groupby(["season", "round"]).kick.min() - pd.Timedelta(minutes=90)).rename("deadline")
    ev_now = pd.read_csv(FPL / "fpl_events.csv")
    cur = ev_now.season.iloc[0]
    for r_ in ev_now.itertuples():
        deadlines.loc[(cur, r_.id)] = pd.to_datetime(r_.deadline_time, utc=True).tz_localize(None)
    return deadlines, fx_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(SRC / "fitted_params_v1.json"))
    ap.add_argument("--label", default="full_sample_merged")
    ap.add_argument("--out", default=str(OUT), help="output folder (default data/ratings)")
    ap.add_argument("--bcd", action="store_true", help="online closed-doors home advantage")
    ap.add_argument("--bcd-prior-n", type=float, default=50.0)
    ap.add_argument("--drift-lr", type=float, default=0.0, help="online league scoring-level drift")
    ap.add_argument("--ha-lr", type=float, default=0.0, help="online (crowd) home-advantage drift")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = json.load(open(a.params))
    matches = load_matches_with_clock()
    deadlines, fx_round = gameweek_deadlines()
    snap_times = sorted(set(deadlines.dropna()))

    model, hist = replay(matches, params, bcd=a.bcd, bcd_prior_n=a.bcd_prior_n, drift_lr=a.drift_lr,
                         ha_lr=a.ha_lr, snap_times=snap_times)
    hist["params_label"] = a.label
    hist.to_csv(out_dir / "pi_ratings_match.csv", index=False)

    # (2) reproducibility: same lambda/mu as the published walk-forward predictions
    wf = pd.read_csv(SRC / "walk_forward_predictions.csv", parse_dates=["date"])
    k = ["league", "date", "home_team", "away_team"]
    j = hist.merge(wf[k + ["lam", "mu"]], on=k, suffixes=("", "_wf"))
    d = max((j.lam - j.lam_wf).abs().max(), (j.mu - j.mu_wf).abs().max())
    same_params = abs(params["home_adv"] - json.load(open(SRC / "fitted_params_v1.json"))["home_adv"]) < 1e-12
    print(f"replay rows {len(hist):,}; matched to walk_forward_predictions {len(j):,}; max |dlambda| = {d:.2e}")
    if same_params and not a.bcd and not a.drift_lr and not a.ha_lr:
        assert len(j) == len(hist) and d < 1e-9, "replay does not reproduce walk_forward_predictions"

    # (3) current ratings snapshot
    cur = []
    for lg, st in model.ratings.items():
        for t in st.attack:
            cur.append({"league": lg, "team": t, "attack": st.attack[t], "defence": st.defence[t]})
    cur = pd.DataFrame(cur)
    last_seen = pd.concat([matches[["league", "date", "home_team"]].rename(columns={"home_team": "team"}),
                           matches[["league", "date", "away_team"]].rename(columns={"away_team": "team"})]
                          ).groupby(["league", "team"]).date.agg(last_match="max", n_matches="size").reset_index()
    cur = cur.merge(last_seen, on=["league", "team"]).assign(params_label=a.label)
    cur.to_csv(out_dir / "pi_ratings_current.csv", index=False)

    # (4) FPL fixtures: played (via crosswalk) + upcoming (via team crosswalk and current snapshot)
    fxw = pd.read_csv(CW / "fixture_crosswalk.csv", parse_dates=["us_date"])
    txw = pd.read_csv(CW / "team_crosswalk.csv")
    fx = pd.read_csv(FPL / "fpl_fixtures.csv", parse_dates=["kickoff_time"])
    epl = hist[hist.league == "EPL"]
    pl = fxw.merge(epl, left_on=["us_date", "us_home", "us_away"], right_on=["date", "home_team", "away_team"],
                   how="left")
    assert pl.lam.notna().all(), "some crosswalked fixtures lack pre-match ratings"
    pl = pl.assign(event=np.nan, kickoff=pl.fpl_kick, home_id=pl.fpl_home_id, away_id=pl.fpl_away_id,
                   is_prediction=False)
    pl["event"] = pl.set_index(["season", "fixture"]).index.map(fx_round)
    # AS OF THE GAMEWEEK DEADLINE: ratings, intercept and home advantage from the snapshot taken just before
    # the deadline (a double gameweek's second fixture must not see the first one's result)
    from config import BCD_PERIODS, in_periods
    pl["deadline"] = [deadlines.get((s_, e_)) for s_, e_ in zip(pl.season, pl.event)]
    pl["closed"] = in_periods(pl.us_date, BCD_PERIODS["EPL"]) & bool(a.bcd)
    vals = [lambdas_from_snapshot(model.snapshots[dl_], h_, v_, c_)
            for dl_, h_, v_, c_ in zip(pl.deadline, pl.us_home, pl.us_away, pl.closed)]
    pl[["lam", "mu", "home_att", "home_def", "away_att", "away_def"]] = np.array(vals)

    cur_season = fx.season.iloc[0]
    played_ids = set(pl[pl.season == cur_season].fixture)
    up = fx[~fx.id.isin(played_ids)].copy()
    t_map = txw[txw.season == cur_season].set_index("fpl_team_id").us_team
    cr = cur[cur.league == "EPL"].set_index("team")
    rows = []
    last_date = matches.date.max()
    for r in up.itertuples():
        th, ta = t_map.get(r.team_h), t_map.get(r.team_a)
        if th is None or ta is None or th not in cr.index or ta not in cr.index:
            continue
        lam, mu = model.predict_lambdas("EPL", th, ta)
        lh, la = cr.loc[th], cr.loc[ta]
        rows.append({"season": cur_season, "fixture": r.id, "event": r.event, "kickoff": r.kickoff_time,
                     "home_id": r.team_h, "away_id": r.team_a, "home_att": lh.attack, "home_def": lh.defence,
                     "away_att": la.attack, "away_def": la.defence, "lam": lam, "mu": mu,
                     "home_n_prev": lh.n_matches, "away_n_prev": la.n_matches,
                     "home_gap_days": (last_date - lh.last_match).days, "away_gap_days": (last_date - la.last_match).days,
                     "is_prediction": True})
    allfx = pd.concat([pl[["season", "fixture", "event", "kickoff", "home_id", "away_id", "home_att", "home_def",
                           "away_att", "away_def", "lam", "mu", "home_n_prev", "away_n_prev", "home_gap_days",
                           "away_gap_days", "is_prediction"]], pd.DataFrame(rows)], ignore_index=True)
    out = tidy(allfx, a.label)
    out.to_csv(out_dir / "fpl_team_fixture_ratings.csv", index=False)

    # ---- QC summary
    s = out[out.was_home & ~out.is_prediction].groupby("season").agg(
        fixtures=("fixture", "size"), xg_for_home=("xg_for_pi", "mean"), xg_against_home=("xg_against_pi", "mean"))
    print(s.round(3).to_string())
    print(f"\nupcoming fixtures rated: {int(out.is_prediction.sum() / 2)} of {len(up)} unplayed FPL fixtures")
    qc_name = "PI_RATINGS_QC.md" if out_dir == OUT else f"PI_RATINGS_QC_{a.label}.md"
    with open(DOCS / qc_name, "w") as f:
        f.write(f"# Pi-rating attachment QC\n\nparams: `{a.label}` (`{Path(a.params).name}`)\n\n"
                f"- replay reproduces walk_forward_predictions: max |d lambda| = {d:.2e} over {len(j):,} matches\n"
                f"- rated FPL team-fixtures: {len(out):,}; of which predictions for upcoming fixtures: {int(out.is_prediction.sum())}\n\n"
                + s.round(3).to_markdown() + "\n")


if __name__ == "__main__":
    main()
