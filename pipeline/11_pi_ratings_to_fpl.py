#!/usr/bin/env python3
"""Attach dual (attack / defence) pi-ratings to FPL fixtures.

1. Replay model_v1's PiRatingDixonColesModel over matches_unified.csv (all six leagues,
   chronological) with a FIXED parameter file, recording for every match the PRE-match
   state: attack and defence rating of both teams, league intercept, lambda / mu, plus
   staleness (matches previously seen, days since the team's last match).
2. Assert the replay reproduces walk_forward_predictions.csv (lambda, mu) to 1e-9, i.e. the
   ratings really are the ones behind the model's published predictions.
3. Snapshot the ratings after the last played match (`pi_ratings_current.csv`).
4. Build `fpl_team_fixture_ratings.csv`: ONE ROW PER TEAM PER FPL FIXTURE (2 rows per
   fixture), including fixtures not yet played. Played fixtures use the pre-match ratings
   of that match; upcoming fixtures use the current snapshot (`is_prediction = True`).
   Join to player rows on (season, fixture, was_home).

Row content for team T vs opponent O: own/opp attack and defence rating, expected goals for
(lambda_T) and against (lambda_O) from the Dixon-Coles model, P(clean sheet), the
distribution of goals conceded (0,1,2,3,4,5+), win/draw/loss probabilities.

PARAMETER PROVENANCE. `--params` selects the hyperparameter file; its label is stored in
every output row. The default is the full-sample fit, which has seen all matches including
those it is later used to describe, a mild look-ahead on 8-11 scalar hyperparameters (not on
team ratings, which are strictly walk-forward). For leak-free use on a given test period, pass
the train-only file from the Stage 4 harness (stage4_v1_train_params.json) and re-run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import DATA, DOCS, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "model_v1"))
from pi_rating_model import PiRatingDixonColesModel  # noqa: E402

SRC = DATA / "interim" / "merged" / "model_v1"
OUT = DATA / "ratings"
CW = DATA / "crosswalk"
FPL = DATA / "fpl"


def replay(matches, params, rho):
    leagues = sorted(matches.league.unique())
    model = PiRatingDixonColesModel(
        home_adv=params["home_adv"], scale=params["scale"], alpha=params["alpha"],
        lr_attack=params["lr_attack"], lr_defence=params["lr_defence"],
        league_intercepts={lg: params[f"c_{lg}"] for lg in leagues}, rho=rho)
    seen, last = {}, {}
    rows = []
    for r in matches.itertuples(index=False):
        st = model._state(r.league)
        st.ensure(r.home_team)
        st.ensure(r.away_team)
        ah, dh = st.get(r.home_team)
        aa, da = st.get(r.away_team)
        kh, ka = (r.league, r.home_team), (r.league, r.away_team)
        lam, mu = model.update(r.league, r.home_team, r.away_team, r.home_goals, r.away_goals,
                               r.home_npxG, r.away_npxG)
        rows.append({
            "league": r.league, "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
            "home_att": ah, "home_def": dh, "away_att": aa, "away_def": da,
            "lam": lam, "mu": mu, "c_league": params[f"c_{r.league}"],
            "home_n_prev": seen.get(kh, 0), "away_n_prev": seen.get(ka, 0),
            "home_gap_days": (r.date - last[kh]).days if kh in last else np.nan,
            "away_gap_days": (r.date - last[ka]).days if ka in last else np.nan})
        for k in (kh, ka):
            seen[k] = seen.get(k, 0) + 1
            last[k] = r.date
    return model, pd.DataFrame(rows)


def score_probs(lam, mu, rho, max_goals=10):
    """Dixon-Coles score matrix -> (win, draw, loss for home), P(away=k), P(home=k)."""
    xs = np.arange(max_goals + 1)
    mat = np.outer(poisson.pmf(xs, lam), poisson.pmf(xs, mu))
    mat[0, 0] *= 1 - lam * mu * rho
    mat[0, 1] *= 1 + lam * rho
    mat[1, 0] *= 1 + mu * rho
    mat[1, 1] *= 1 - rho
    mat /= mat.sum()
    ph, pd_, pa = np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()
    conc_home = mat.sum(axis=0)  # distribution of away goals = goals conceded by home
    conc_away = mat.sum(axis=1)  # distribution of home goals = goals conceded by away
    return (ph, pd_, pa), conc_home, conc_away


def tidy(side, d, rho, label):
    """Two rows (home, away) per fixture from one wide fixture row."""
    rows = []
    for r in d.itertuples(index=False):
        (ph, px, pa), conc_h, conc_a = score_probs(r.lam, r.mu, rho)
        for home in (True, False):
            conc = conc_h if home else conc_a
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
                "p_win": ph if home else pa, "p_draw": px, "p_loss": pa if home else ph,
                "p_clean_sheet": conc[0], "p_conc_0": conc[0], "p_conc_1": conc[1], "p_conc_2": conc[2],
                "p_conc_3": conc[3], "p_conc_4": conc[4], "p_conc_5plus": conc[5:].sum(),
                "is_prediction": r.is_prediction, "rho": rho, "params_label": label})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(SRC / "fitted_params_v1.json"))
    ap.add_argument("--label", default="full_sample_merged")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.load(open(a.params))
    rho = params.get("rho", -0.13)
    matches = pd.read_csv(SRC / "matches_unified.csv", parse_dates=["date"])
    matches = matches.sort_values(["date", "league", "home_team"], kind="mergesort").reset_index(drop=True)

    model, hist = replay(matches, params, rho)
    hist["params_label"] = a.label
    hist.to_csv(OUT / "pi_ratings_match.csv", index=False)

    # (2) reproducibility: same lambda/mu as the published walk-forward predictions
    wf = pd.read_csv(SRC / "walk_forward_predictions.csv", parse_dates=["date"])
    k = ["league", "date", "home_team", "away_team"]
    j = hist.merge(wf[k + ["lam", "mu"]], on=k, suffixes=("", "_wf"))
    d = max((j.lam - j.lam_wf).abs().max(), (j.mu - j.mu_wf).abs().max())
    same_params = abs(params["home_adv"] - json.load(open(SRC / "fitted_params_v1.json"))["home_adv"]) < 1e-12
    print(f"replay rows {len(hist):,}; matched to walk_forward_predictions {len(j):,}; max |dlambda| = {d:.2e}")
    if same_params:
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
    cur.to_csv(OUT / "pi_ratings_current.csv", index=False)

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
    pm = pd.read_csv(FPL / "fpl_player_match.csv", usecols=["season", "fixture", "round"], low_memory=False)
    ev = pm.groupby(["season", "fixture"])["round"].agg(lambda x: x.mode().iloc[0]).rename("event_pm")
    pl["event"] = pl.set_index(["season", "fixture"]).index.map(ev)

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
    out = tidy(None, allfx, rho, a.label)
    out.to_csv(OUT / "fpl_team_fixture_ratings.csv", index=False)

    # ---- QC summary
    s = out[out.was_home & ~out.is_prediction].groupby("season").agg(
        fixtures=("fixture", "size"), xg_for_home=("xg_for_pi", "mean"), xg_against_home=("xg_against_pi", "mean"),
        mean_p_cs=("p_clean_sheet", "mean"))
    print(s.round(3).to_string())
    print(f"\nupcoming fixtures rated: {int(out.is_prediction.sum() / 2)} of {len(up)} unplayed FPL fixtures")
    with open(DOCS / "PI_RATINGS_QC.md", "w") as f:
        f.write(f"# Pi-rating attachment QC\n\nparams: `{a.label}` (`{Path(a.params).name}`), rho={rho:.4f}\n\n"
                f"- replay reproduces walk_forward_predictions: max |d lambda| = {d:.2e} over {len(j):,} matches\n"
                f"- rated FPL team-fixtures: {len(out):,}; of which predictions for upcoming fixtures: {int(out.is_prediction.sum())}\n\n"
                + s.round(3).to_markdown() + "\n")


if __name__ == "__main__":
    main()
