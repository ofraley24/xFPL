#!/usr/bin/env python3
"""Truncation test for the horizon rows (22_build_horizon_rows.py), as T1 of 18_audit.py.

For random horizon rows with h >= 2, every feature is recomputed from scratch using ONLY raw records dated
before the EARLIER deadline the row is predicted from (asof_deadline), and compared with the stored value:
  a) Understat form (decayed 180/540 d sums) from raw Understat matches before min(kickoff - 36 h, deadline)
  b) FPL minutes features from the player's FPL rows kicked off before the deadline
  c) FPL event lags (L365) from the same rows, decayed to the target kickoff
  d) team expected goals: pi-rating replay truncated at the deadline, priced for the target fixture
Writes docs/AUDIT_HORIZON.md. Exit status 1 on any failure.
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
sys.path.insert(0, str(ROOT / "pipeline"))
SEED = 20261013


def load_mod(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def main():
    M12 = load_mod("m12", ROOT / "analysis/12_build_master.py")
    R11 = load_mod("r11", ROOT / "pipeline/11_pi_ratings_to_fpl.py")
    hz = pd.read_parquet(OUT / "horizon_rows.parquet")
    master = pd.read_parquet(OUT / "master.parquet")
    res = []

    def check(name, ok, detail):
        res.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + "  " + detail, flush=True)

    samp = hz[hz.horizon >= 2].sample(400, random_state=SEED)
    check("H0 sample covers horizons 2-8 and every season", samp.horizon.nunique() == 7 and samp.season.nunique() == hz.season.nunique(),
          f"horizons {sorted(samp.horizon.unique())}, seasons {samp.season.nunique()}")
    check("H0 every row is predicted from a deadline before its kickoff, h gameweeks earlier",
          bool((hz.asof_deadline < hz.kickoff_time).all() and (hz.horizon == hz["round"] - hz.asof_round + 1).all()),
          f"{len(hz):,} rows")

    # ---- a) Understat
    cw = pd.read_csv(ROOT / "data/crosswalk/player_crosswalk.csv", usecols=["code", "us_player_id"])
    raw = M12.understat_match_table()
    s = samp.merge(cw, on="code", how="left", suffixes=("", "_cw"))
    errs = []
    for r in s.itertuples():
        if pd.isna(r.us_player_id_cw):
            continue
        cut = min(r.kickoff_time - pd.Timedelta(hours=36), r.asof_deadline)
        rows = raw[(raw.player_id == int(r.us_player_id_cw)) & (raw.date < cut)]
        for hl in (180, 540):
            w = 0.5 ** ((r.kickoff_time - rows.date).dt.total_seconds() / 86400 / hl)
            for c in ("time", "npxg", "xa", "np_shots", "team_np_on"):
                want = float((w * rows[c]).sum())
                got = getattr(r, f"L{hl}_{c}")
                errs.append(abs(want - got) / max(abs(want), 1e-6) if want > 1e-9 else abs(got))
    errs = np.array(errs)
    check("H1a Understat form recomputed from raw rows before the earlier deadline", np.quantile(errs, 0.999) < 1e-6,
          f"{len(errs)} comparisons, max rel err {errs.max():.2e}")

    # ---- b/c) FPL minutes features and event lags
    md = master[["code", "kickoff_time", "minutes", "yellow_cards", "bps", "saves", "season"]].sort_values("kickoff_time")
    bad_b, bad_c, n_b = 0, 0, 0
    for r in samp.itertuples():
        h = md[(md.code == r.code) & (md.kickoff_time < r.asof_deadline)]
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
    check("H1b FPL minutes features recomputed from rows before the earlier deadline", bad_b == 0,
          f"{n_b} rows x 8 features, mismatches {bad_b}")
    check("H1c FPL event lags (L365) recomputed from rows before the earlier deadline", bad_c == 0,
          f"{len(samp)} rows x 4 features, mismatches {bad_c}")

    # ---- d) team ratings by a replay truncated at the earlier deadline
    params = json.load(open(ROOT / "data/interim/pre2019/fitted_params_pre2019.json"))
    sel = pd.read_csv(ROOT / "docs/ratings_asof_selection.csv").set_index("season")
    from config import BCD_PERIODS, in_periods
    mt = R11.load_matches_with_clock()
    fxw = pd.read_csv(ROOT / "data/crosswalk/fixture_crosswalk.csv", parse_dates=["us_date"])
    picks = samp[samp.was_home].drop_duplicates(["season", "fixture", "asof_round"]).sample(10, random_state=SEED)
    worst = 0.0
    for p in picks.itertuples():
        f = fxw[(fxw.season == p.season) & (fxw.fixture == p.fixture)].iloc[0]
        dl, ha = (sel.loc[p.season, "drift_lr"], sel.loc[p.season, "ha_lr"])
        before = mt[mt.t < p.asof_deadline].reset_index(drop=True)
        model, _ = R11.replay(before, params, bcd=True, bcd_prior_n=50, drift_lr=dl, ha_lr=ha)
        snap = R11.snapshot_state(model, model.shared_state, True, 50)
        closed = bool(in_periods(pd.Series([f.us_date]), BCD_PERIODS["EPL"])[0])
        lam = R11.lambdas_from_snapshot(snap, f.us_home, f.us_away, closed)[0]
        worst = max(worst, abs(lam - p.xg_for_pi))
    check("H1d team lambda reproduced by a replay truncated at the earlier deadline", worst < 1e-9,
          f"10 fixtures, max |d lambda| = {worst:.2e}")

    tab = pd.DataFrame(res)
    (ROOT / "docs" / "AUDIT_HORIZON.md").write_text(
        f"# Truncation test of the horizon rows ({date.today()})\n\n"
        "Rows predicted h >= 2 gameweeks ahead; every feature recomputed from records dated before the earlier "
        "deadline (`analysis/23_audit_horizon.py`).\n\n" + tab.to_markdown(index=False) + "\n")
    sys.exit(0 if (tab.result == "PASS").all() else 1)


if __name__ == "__main__":
    main()
