#!/usr/bin/env python3
"""Build tidy FPL tables in data/fpl/ from the raw API snapshots and vaastav archive.

Outputs
  fpl_player_match.csv     ONE ROW PER PLAYER PER FIXTURE (per gameweek), 2016-17 .. current.
                           Key: (season, element, fixture, round). Includes the points stats,
                           ICT components (influence/creativity/threat/ict_index), BPS, bonus,
                           expected_goals / expected_assists / expected_goal_involvements /
                           expected_goals_conceded (2022-23 on), tackles / recoveries / CBI /
                           defensive_contribution (2025-26 on), FPL's own per-match xP
                           (2020-21 on), price (`value`), ownership (`selected`), transfers.
                           Current season comes from the live API; earlier seasons from vaastav.
                           Pre-2020 seasons also carry the extra Opta-style fields vaastav holds.
  fpl_players_snapshot.csv APPEND-ONLY history of the live `elements` table, one row per
                           player per API snapshot: status, news, news_added,
                           chance_of_playing_this/next_round (INJURY/AVAILABILITY), prices,
                           ownership, ep_next/ep_this, set-piece orders (penalties,
                           corners/indirect FKs, direct FKs), form, season-to-date ICT and
                           xG/xA, defensive stats. Grows with every run of 06.
  fpl_player_ids.csv       (season, element, code, opta_code, web_name, position, team_id)
                           code is the persistent cross-season player id.
  fpl_teams.csv            teams with FPL attack/defence/overall strength ratings (home/away)
  fpl_events.csv           gameweeks and deadlines (current season snapshot)
  fpl_fixtures.csv         fixtures with FPL difficulty ratings (current, from API)
  fpl_scoring_rules.csv    scoring rules as published by the API at each snapshot (long)
  fpl_player_season_past.csv  per-player past-season totals as reported by the API
Duplicates removed are written to docs/fpl_dedup_log.csv.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402

API = DATA / "raw" / "fpl_api"
VAA = DATA / "raw" / "vaastav"
OUT = DATA / "fpl"
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD", 5: "MNG"}  # MNG: managers, scored in 2024-25 only
POS_NORM = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD", "MNG": "MNG", "AM": "MNG"}

CORE = ["season", "round", "element", "code", "name", "position", "team_name", "team_id",
        "opponent_team", "was_home", "fixture", "kickoff_time", "team_h_score", "team_a_score",
        "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
        "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves",
        "bonus", "bps", "influence", "creativity", "threat", "ict_index",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "clearances_blocks_interceptions", "recoveries", "tackles",
        "defensive_contribution", "total_points", "xP", "value", "selected",
        "transfers_in", "transfers_out", "transfers_balance", "fixture_finished", "source"]


def read_csv_any(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(path)


def latest(pattern_dir=API):
    dirs = sorted(p for p in pattern_dir.iterdir() if p.is_dir())
    with_sum = [d for d in dirs if (d / "element_summaries.json.gz").exists()]
    return dirs, (with_sum[-1] if with_sum else None)


def season_of(boot):
    import re
    m = re.search(r"/(\d{4})_(\d{2})/", boot["game_config"]["settings"]["static_content_url"])
    return f"{m.group(1)}-{m.group(2)}"


# ------------------------------------------------------------------ live API
def build_live():
    dirs, last_full = latest()
    snaps, rules = [], []
    for d in dirs:
        boot = json.loads((d / "bootstrap.json").read_text())
        el = pd.DataFrame(boot["elements"])
        el.insert(0, "snapshot_utc", d.name)
        el.insert(1, "season", season_of(boot))
        el.insert(2, "current_event", next((e["id"] for e in boot["events"] if e["is_current"]), np.nan))
        snaps.append(el)
        for k, v in boot["game_config"]["scoring"].items():
            if isinstance(v, dict):
                rules += [{"snapshot_utc": d.name, "stat": k, "position": p, "points": pts}
                          for p, pts in v.items()]
            else:
                rules.append({"snapshot_utc": d.name, "stat": k, "position": "ALL", "points": v})
    pd.concat(snaps, ignore_index=True).to_csv(OUT / "fpl_players_snapshot.csv", index=False)
    pd.DataFrame(rules).to_csv(OUT / "fpl_scoring_rules.csv", index=False)

    boot = json.loads((dirs[-1] / "bootstrap.json").read_text())
    season = season_of(boot)
    pd.DataFrame(boot["teams"]).assign(season=season).to_csv(OUT / "fpl_teams.csv", index=False)
    ev = pd.DataFrame(boot["events"]).drop(columns=["top_element_info", "chip_plays"], errors="ignore")
    ev.assign(season=season).to_csv(OUT / "fpl_events.csv", index=False)
    fx = pd.DataFrame(json.loads((dirs[-1] / "fixtures.json").read_text()))
    fx = fx.drop(columns=["stats"], errors="ignore").assign(season=season)
    fx.to_csv(OUT / "fpl_fixtures.csv", index=False)

    ids = pd.DataFrame(boot["elements"])[["id", "code", "opta_code", "web_name", "element_type", "team"]]
    ids = ids.rename(columns={"id": "element", "team": "team_id"})
    ids["position"] = ids.element_type.map(POS)
    ids.insert(0, "season", season)
    teams = {t["id"]: t["name"] for t in boot["teams"]}
    ids["team_name"] = ids.team_id.map(teams)

    pm = pd.DataFrame()
    if last_full is not None:
        with gzip.open(last_full / "element_summaries.json.gz", "rt") as f:
            summ = json.load(f)
        rows, past = [], []
        for pid, s in summ.items():
            rows += [{**h, "element": int(pid)} for h in s["history"]]
            past += [{**h, "element": int(pid)} for h in s["history_past"]]
        pm = pd.DataFrame(rows)
        for c in ("influence", "creativity", "threat", "ict_index", "expected_goals",
                  "expected_assists", "expected_goal_involvements", "expected_goals_conceded"):
            pm[c] = pd.to_numeric(pm[c], errors="coerce")
        pm["season"] = season
        pm = pm.merge(ids[["element", "code", "web_name", "position", "team_id", "team_name"]],
                      on="element", how="left").rename(columns={"web_name": "name"})
        pm["source"] = f"fpl_api:{last_full.name}"
        # flag fixtures that were still in play / unconfirmed when snapshotted; they MUST NOT
        # be used as training data (minutes and stats are partial)
        fxd = json.loads((last_full / "fixtures.json").read_text())
        done = {f["id"]: bool(f["finished"] or f["finished_provisional"]) for f in fxd}
        pm["fixture_finished"] = pm.fixture.map(done).fillna(False)
        pd.DataFrame(past).to_csv(OUT / "fpl_player_season_past.csv", index=False)
    return ids, pm


# ------------------------------------------------------------- vaastav history
def build_history(skip_season):
    frames, ids_all, dedup = [], [], []
    for sdir in sorted(p for p in VAA.iterdir() if p.is_dir()):
        s = sdir.name
        if s == skip_season or not (sdir / "gws" / "merged_gw.csv").exists():
            continue
        d = read_csv_any(sdir / "gws" / "merged_gw.csv")
        raw = read_csv_any(sdir / "players_raw.csv")
        ids = raw[["id", "code", "web_name", "element_type", "team"]].rename(
            columns={"id": "element", "team": "team_id"})
        if "opta_code" in raw.columns:
            ids["opta_code"] = raw["opta_code"]
        ids["position"] = ids.element_type.map(POS)
        ids.insert(0, "season", s)
        ids_all.append(ids)
        if "position" in d.columns:
            d["position"] = d.position.map(POS_NORM)
        d = d.drop(columns=[c for c in ("position",) if c in d.columns and d[c].isna().all()])
        d = d.merge(ids[["element", "code", "position", "team_id"]].rename(
            columns={"position": "position_raw"}), on="element", how="left")
        d["position"] = d["position"] if "position" in d.columns else d["position_raw"]
        d["position"] = d["position"].fillna(d["position_raw"])
        d = d.drop(columns="position_raw")
        d["team_name"] = d["team"] if "team" in d.columns else np.nan
        if "team" in d.columns:
            d = d.drop(columns="team")
        d["season"] = s
        d["source"] = f"vaastav:{s}"
        before = len(d)
        # exact duplicate rows are an archive artefact; same (element, fixture) in different
        # rounds is legitimate (postponed fixtures, e.g. 2019-20 fixture 275 in GW29 and GW39)
        dup = d.duplicated(keep="first")
        if dup.any():
            dedup.append({"season": s, "rows_removed": int(dup.sum()), "rows_before": before})
        d = d[~dup]
        k = d.duplicated(["element", "fixture", "round"], keep=False)
        assert not k.any(), f"{s}: key (element,fixture,round) not unique after exact-dedup"
        frames.append(d)
    pd.DataFrame(dedup, columns=["season", "rows_removed", "rows_before"]).to_csv(
        DOCS / "fpl_dedup_log.csv", index=False)
    return pd.concat(ids_all, ignore_index=True), pd.concat(frames, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    live_ids, live_pm = build_live()
    cur = live_ids.season.iloc[0]
    hist_ids, hist_pm = build_history(skip_season=cur if len(live_pm) else None)
    hist_pm["fixture_finished"] = True
    pm = pd.concat([hist_pm, live_pm], ignore_index=True)
    for c in CORE:
        if c not in pm.columns:
            pm[c] = np.nan
    extras = [c for c in pm.columns if c not in CORE]
    pm = pm[CORE + sorted(extras)]
    pm["kickoff_time"] = pd.to_datetime(pm.kickoff_time, utc=True, errors="coerce")
    pm = pm.sort_values(["season", "round", "element", "kickoff_time"]).reset_index(drop=True)
    pm.to_csv(OUT / "fpl_player_match.csv", index=False)
    ids = pd.concat([hist_ids, live_ids], ignore_index=True)
    ids.to_csv(OUT / "fpl_player_ids.csv", index=False)
    print(pm.groupby("season").agg(rows=("element", "size"), players=("element", "nunique"),
                                   gws=("round", "nunique")))
    print(f"\n{len(pm):,} player-match rows, {pm.shape[1]} columns -> {OUT}")


if __name__ == "__main__":
    main()
