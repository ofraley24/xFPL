#!/usr/bin/env python3
"""Live track record: every archived pre-deadline forecast scored once its gameweek is finished.

For each deadline gameweek t with a forecast in data/forecasts (the latest one written before the deadline), the
forecast's gameweek-t xP is compared with what happened and with FPL's own projection (ep_next):
  * FPL, same snapshot: ep_next from the snapshot the forecast was built from (same information time);
  * FPL, at the deadline: ep_next from the last snapshot before the deadline (later team news; a tougher test).
Rows: player-gameweeks in which the player STARTED every fixture of the gameweek (our xP is conditional on starting;
FPL's ep_next also prices the chance of not playing, which these rows remove). Double gameweeks are summed.
Metrics per gameweek and cumulatively: bias, MAE, RMSE, within-gameweek Spearman rank correlation, the mean actual
points of each method's top 10, and actual/predicted. Output: data/forecasts/track_record.json.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FC = DATA / "forecasts"
SNAPS = DATA / "raw" / "fpl_api"
r = lambda x, d=3: None if x is None or not np.isfinite(x) else round(float(x), d)
NAME = re.compile(r"forecast_(\d{4}-\d{2})_gw(\d+)_(\d{8}T\d{6}Z)\.parquet")


def ep_next(snap, gw):
    """FPL's projection for gameweek gw from a snapshot; ep_next refers to the snapshot's NEXT gameweek, so it is
    used only when that is gw (else empty)."""
    b = json.load(open(SNAPS / snap / "bootstrap.json"))
    nxt = next((e["id"] for e in b["events"] if e.get("is_next")), None)
    if nxt != gw:
        return {}
    return {e["id"]: float(e["ep_next"]) for e in b["elements"] if e.get("ep_next") not in (None, "")}


def metrics(g, col):
    e = g.actual - g[col]
    by = g.groupby("gw")
    rank = by.apply(lambda x: spearmanr(x[col], x.actual).correlation if len(x) > 20 else np.nan)
    top = by.apply(lambda x: x.nlargest(10, col).actual.mean())
    return {"bias": r(-e.mean()), "mae": r(e.abs().mean()), "rmse": r(np.sqrt((e ** 2).mean())), "rank": r(rank.mean()),
            "top10": r(top.mean(), 2), "ratio": r(g.actual.sum() / g[col].sum())}


def main():
    master = pd.read_parquet(DATA / "analysis" / "master.parquet",
                             columns=["season", "round", "element", "fixture", "started", "total_points", "minutes"])
    events = pd.read_csv(DATA / "fpl" / "fpl_events.csv")
    season_now = events.season.iloc[0]
    deadlines = dict(zip(events.id, pd.to_datetime(events.deadline_time, utc=True).dt.tz_localize(None)))
    finished = set(events[events.finished.astype(bool)].id)
    snaps = sorted(p.name for p in SNAPS.iterdir() if (p / "bootstrap.json").exists())
    snap_t = {s: pd.Timestamp(datetime.strptime(s, "%Y%m%dT%H%M%SZ")) for s in snaps}

    latest = {}
    for f in sorted(FC.glob("forecast_*.parquet")):
        m = NAME.match(f.name)
        if m and m.group(1) == season_now:
            latest[int(m.group(2))] = (f, m.group(3))   # sorted: later snapshot wins
    rows = []
    for gw, (f, snap) in sorted(latest.items()):
        if gw not in finished:
            continue
        fc = pd.read_parquet(f, columns=["round", "element", "fixture", "xp"])
        fc = fc[fc["round"] == gw]
        act = master[(master.season == season_now) & (master["round"] == gw)]
        j = fc.merge(act, on=["element", "fixture"], how="inner")
        per = j.groupby("element").agg(xp=("xp", "sum"), actual=("total_points", "sum"), n=("fixture", "size"),
                                       started=("started", "min")).reset_index()
        per = per[per.started == 1]
        before = [s for s in snaps if snap_t[s] < deadlines[gw]]
        per["fpl_same"] = per.element.map(ep_next(snap, gw))
        per["fpl_deadline"] = per.element.map(ep_next(before[-1], gw)) if before else np.nan
        per["gw"] = gw
        per["forecast_snapshot"], per["deadline_snapshot"] = snap, before[-1] if before else None
        rows.append(per)
    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "season": season_now,
           "rows": "player-gameweeks in which the player started every fixture", "gameweeks": [], "overall": None}
    if rows:
        all_ = pd.concat(rows, ignore_index=True).dropna(subset=["fpl_same", "fpl_deadline"])
        for gw, g in all_.groupby("gw"):
            out["gameweeks"].append({"gw": int(gw), "n": int(len(g)), "forecast_snapshot": g.forecast_snapshot.iloc[0],
                                     "deadline_snapshot": g.deadline_snapshot.iloc[0], "model": metrics(g, "xp"),
                                     "fpl_same": metrics(g, "fpl_same"), "fpl_deadline": metrics(g, "fpl_deadline")})
        out["overall"] = {"n": int(len(all_)), "gws": int(all_.gw.nunique()), "model": metrics(all_, "xp"),
                          "fpl_same": metrics(all_, "fpl_same"), "fpl_deadline": metrics(all_, "fpl_deadline")}
    FC.mkdir(parents=True, exist_ok=True)
    (FC / "track_record.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"track record: {len(out['gameweeks'])} scored gameweeks"
          + (f"; overall model MAE {out['overall']['model']['mae']} vs FPL {out['overall']['fpl_same']['mae']}" if out["overall"] else
             f" (forecasts waiting: {sorted(latest)}; finished: {sorted(g for g in latest if g in finished)})"))


if __name__ == "__main__":
    main()
