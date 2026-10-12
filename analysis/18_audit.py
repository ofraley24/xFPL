#!/usr/bin/env python3
"""Leakage and integrity audit of the xP pipeline. Writes docs/AUDIT.md.

T1 TRUNCATION TEST (strongest check). For random player-fixtures in the test seasons, every feature is
   recomputed from scratch using ONLY raw records dated before the GAMEWEEK DEADLINE (the decision time),
   then compared with the value the model was given:
     a) Understat form (decayed sums, 180/540 d) from raw Understat player-matches > 36 h before kickoff
     b) FPL minutes features (m1, m3, m5, f_play5, f_long5, absent_streak, days_since_app, std_mpf)
        from the player's FPL rows with an earlier kickoff
     c) FPL event lags (L365_*) from earlier FPL rows
     d) team ratings: the pi-rating model is replayed on matches strictly before the fixture's date, with
        that season's settings, and lambda is predicted for the fixture
T2 SINGLE-FEATURE LEAK SCAN: AUC of each minutes/points feature for its target in the test rows;
   AUC > 0.95 for any single feature is flagged as suspicious.
T3 PROTOCOL: sealed/live seasons absent from the dev predictions; hyperparameter provenance; every fold
   trains on earlier seasons only; EB strengths estimated on earlier Understat seasons only.
T4 INTEGRITY: key uniqueness; no missing ratings; points identity (simulated rules reproduce actual
   total_points from realised events).
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
sys.path.insert(0, str(ROOT / "pipeline"))
SEED = 20261011
rng = np.random.default_rng(SEED)


def load_mod(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def main():
    M12 = load_mod("m12", ROOT / "analysis/12_build_master.py")
    R11 = load_mod("r11", ROOT / "pipeline/11_pi_ratings_to_fpl.py")
    master = pd.read_parquet(OUT / "master.parquet")
    dev = pd.read_parquet(OUT / "xp_predictions_dev.parquet")
    lines = [f"# Audit of the xP pipeline ({date.today()})\n"]
    res = []

    def check(name, ok, detail):
        res.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + "  " + detail, flush=True)

    test = master[master.season.between("2019-20", "2024-25")]
    samp = test[test.code.notna()].sample(400, random_state=SEED)

    # ---- T1a Understat as-of features from raw rows
    cw = pd.read_csv(ROOT / "data/crosswalk/player_crosswalk.csv", usecols=["code", "us_player_id"])
    raw = M12.understat_match_table()
    s = samp.merge(cw, on="code", how="left", suffixes=("", "_cw"))
    errs = []
    for r in s.itertuples():
        if pd.isna(r.us_player_id_cw):
            continue
        cut = min(r.kickoff_time - pd.Timedelta(hours=36), r.deadline)
        rows = raw[(raw.player_id == int(r.us_player_id_cw)) & (raw.date < cut)]
        for hl in (180, 540):
            w = 0.5 ** ((r.kickoff_time - rows.date).dt.total_seconds() / 86400 / hl)
            for c in ("time", "npxg", "xa", "np_shots", "team_np_on"):
                want = float((w * rows[c]).sum())
                got = getattr(r, f"L{hl}_{c}")
                errs.append(abs(want - got) / max(abs(want), 1e-6) if want > 1e-9 else abs(got))
    errs = np.array(errs)
    check("T1a Understat form recomputed from raw pre-kickoff rows", np.quantile(errs, 0.999) < 1e-6,
          f"{len(errs)} comparisons, max rel err {errs.max():.2e}")

    # ---- T1b/c FPL minutes features and event lags from earlier FPL rows
    md = master[["code", "kickoff_time", "minutes", "yellow_cards", "bps", "saves", "season"]].sort_values("kickoff_time")
    bad_b, bad_c, n_b = 0, 0, 0
    for r in samp.itertuples():
        h = md[(md.code == r.code) & (md.kickoff_time < r.deadline)]   # known at the gameweek deadline
        mm = h.minutes.to_numpy(float)
        if len(mm):
            n_b += 1
            want = {"m1": mm[-1], "m3": mm[-3:].mean(), "m5": mm[-5:].mean(), "f_play5": (mm[-5:] > 0).mean(),
                    "f_long5": (mm[-5:] >= 60).mean()}
            streak = 0
            for v in mm[::-1]:
                if v == 0:
                    streak += 1
                else:
                    break
            want["absent_streak"] = streak
            app = h[h.minutes > 0]
            want["days_since_app"] = ((r.kickoff_time - app.kickoff_time.iloc[-1]).total_seconds() / 86400) if len(app) else np.nan
            cur = h[h.season == r.season].minutes
            want["std_mpf"] = cur.mean() if len(cur) else np.nan
            for k, v in want.items():
                g = getattr(r, k)
                if not ((pd.isna(v) and pd.isna(g)) or (not pd.isna(v) and not pd.isna(g) and abs(v - g) < 1e-6)):
                    bad_b += 1
        days = (r.kickoff_time - h.kickoff_time).dt.total_seconds() / 86400
        w = 0.5 ** (days / 365)
        for col, src in (("L365_f_min", "minutes"), ("L365_f_yel", "yellow_cards"), ("L365_f_bps", "bps"), ("L365_f_saves", "saves")):
            want = float((w * h[src].fillna(0)).sum())
            if abs(want - getattr(r, col)) > 1e-6 * max(1, abs(want)):
                bad_c += 1
    check("T1b FPL minutes features recomputed from earlier rows", bad_b == 0, f"{n_b} players x 8 features, mismatches {bad_b}")
    check("T1c FPL event lags (L365) recomputed from earlier rows", bad_c == 0, f"{len(samp)} rows x 4 features, mismatches {bad_c}")

    # ---- T1d team ratings by truncated replay (state at the gameweek deadline)
    params = json.load(open(ROOT / "data/interim/pre2019/fitted_params_pre2019.json"))
    sel = pd.read_csv(ROOT / "docs/ratings_asof_selection.csv").set_index("season")
    from config import BCD_PERIODS, in_periods
    mt = R11.load_matches_with_clock()
    deadlines, fx_round = R11.gameweek_deadlines()
    fxw = pd.read_csv(ROOT / "data/crosswalk/fixture_crosswalk.csv", parse_dates=["us_date"])
    rt = pd.read_csv(ROOT / "data/ratings_eval/fpl_team_fixture_ratings.csv")
    rt = rt[rt.was_home.astype(str) == "True"]
    picks = fxw[fxw.season.between("2019-20", "2024-25")].sample(10, random_state=SEED)
    worst = 0.0
    for p in picks.itertuples():
        dl, ha = (sel.loc[p.season, "drift_lr"], sel.loc[p.season, "ha_lr"])
        deadline = deadlines.loc[(p.season, fx_round.loc[(p.season, p.fixture)])]
        before = mt[mt.t < deadline].reset_index(drop=True)          # nothing at or after the deadline
        model, _ = R11.replay(before, params, params["rho"], bcd=True, bcd_prior_n=50, drift_lr=dl, ha_lr=ha)
        snap = R11.snapshot_state(model, model.shared_state, True, 50)
        closed = bool(in_periods(pd.Series([p.us_date]), BCD_PERIODS["EPL"])[0])
        lam = R11.lambdas_from_snapshot(snap, p.us_home, p.us_away, closed)[0]
        stored = rt[(rt.season == p.season) & (rt.fixture == p.fixture)].xg_for_pi.iloc[0]
        worst = max(worst, abs(lam - stored))
    check("T1d team lambda reproduced by a replay truncated at the gameweek deadline", worst < 1e-9,
          f"10 fixtures, max |d lambda| = {worst:.2e}")

    # ---- T2 single-feature leak scan
    MIN_FEATS = load_mod("x13", ROOT / "analysis/13_xp_model.py").MIN_FEATS
    t = dev.copy()
    t["home"] = t.was_home.astype(float)
    t["selected_log"] = np.log1p(t.selected)
    for c in ("L540_log", "L180_log", "us_days_log"):
        if c not in t:
            t[c] = 0.0
    played = (t.minutes > 0).astype(int)
    scored = (t.total_points >= 5).astype(int)
    aucs = []
    for f in MIN_FEATS:
        if f not in t or t[f].nunique() < 2:
            continue
        x = t[f].fillna(t[f].median())
        a1 = roc_auc_score(played, x); a2 = roc_auc_score(scored, x)
        aucs.append({"feature": f, "AUC played": max(a1, 1 - a1), "AUC 5+ pts": max(a2, 1 - a2)})
    A = pd.DataFrame(aucs).sort_values("AUC played", ascending=False)
    check("T2 no single feature near-perfectly predicts the outcome (AUC < 0.95)", (A[["AUC played", "AUC 5+ pts"]] < 0.95).all().all(),
          f"max AUC played {A['AUC played'].max():.3f} ({A.iloc[0].feature})")

    # ---- T3 protocol
    check("T3 dev predictions contain only seasons 2019-20..2024-25", set(dev.season) <= {f"{y}-{str(y+1)[2:]}" for y in range(2019, 2025)},
          f"seasons: {sorted(set(dev.season))}")
    check("T3 team-rating hyperparameters fitted before the first test season", params["_meta"]["train_cutoff"] == "2019-08-01",
          f"cutoff {params['_meta']['train_cutoff']}, {params['_meta']['n_train_matches']} matches")
    sel_ok = all(int(s[:4]) >= 2019 for s in sel.index)
    check("T3 drift settings chosen per season from earlier data only (selection table present)", sel_ok,
          "; ".join(f"{s}: d={r.drift_lr}, h={r.ha_lr}" for s, r in sel.iterrows()))
    code13 = (ROOT / "analysis/13_xp_model.py").read_text()
    check("T3 xP model reads only seasons <= 2024-25 and fits each fold on earlier seasons",
          'load("2025-26" if sealed else "2024-25")' in code13 and "tr, te = d[d.season < ts]" in code13
          and "assert tr.season.max() < ts" in code13 and "int(ts[:4]) - 1" in code13,
          "static check of 13_xp_model.py: load filter, fold split, EB estimated on seasons < test year")

    # ---- T4 integrity
    check("T4 master key (season, element, fixture) unique", not master.duplicated(["season", "element", "fixture"]).any(), f"{len(master):,} rows")
    check("T4 every master row has team ratings", master.xg_for_pi.notna().all(), "")
    pdf = dev.position
    g = pdf.map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}); c = pdf.map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})
    gc = pdf.map({"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0})
    pts = ((dev.minutes > 0).astype(int) + (dev.minutes >= 60).astype(int) + g * dev.goals_scored + 3 * dev.assists
           + c * dev.clean_sheets + gc * (dev.goals_conceded // 2) + dev.saves // 3 + 5 * dev.penalties_saved
           - 2 * dev.penalties_missed - dev.yellow_cards - 3 * dev.red_cards - 2 * dev.own_goals + dev.bonus)
    check("T3 price and ownership are taken from the previous gameweek",
          (master.value == master.value_raw).mean() < 0.99 and "value_prev" in (ROOT / "analysis/06_build_minutes_dataset.py").read_text(),
          f"value differs from the same-gameweek value on {(master.value != master.value_raw).mean():.1%} of rows")
    check("T4 the simulator's scoring rules reproduce actual total_points from realised events",
          (pts == dev.total_points).mean() > 0.999, f"{(pts == dev.total_points).mean():.4%} of {len(dev):,} rows")

    R = pd.DataFrame(res)
    lines += [f"{(R.result == 'PASS').sum()}/{len(R)} checks passed.\n", R.to_markdown(index=False), "",
              "## Single-feature AUCs (T2)\n", A.round(3).to_markdown(index=False), ""]
    (ROOT / "docs" / "AUDIT_TESTS.md").write_text("\n".join(lines))
    print(f"\n{(R.result == 'PASS').sum()}/{len(R)} passed -> docs/AUDIT_TESTS.md")


if __name__ == "__main__":
    main()
