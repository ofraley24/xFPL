"""
Stage 0 (data prep) for the dual pi-rating -> Dixon-Coles Poisson model, v1.

Builds one unified, match-level dataset across all Understat leagues:
date, league, season, home_team, away_team, home_goals, away_goals,
home_npxG, away_npxG (non-penalty xG, per the literature review's
recommendation to fix a single provider and use non-penalty xG as the base).

Source: understat/{LEAGUE}_matches.csv (result + shot-based xG) joined with
understat/{LEAGUE}_team_match_stats.csv (npxG per team per match) on
(date, team, home/away side).
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "understat"
OUT_DIR = Path(__file__).resolve().parent
LEAGUES = ["EPL", "Bundesliga", "La_liga", "Ligue_1", "Serie_A", "RFPL"]


def build_league(league: str) -> pd.DataFrame:
    matches = pd.read_csv(DATA_DIR / f"{league}_matches.csv", parse_dates=["date"])
    matches = matches[matches["isResult"] == True].copy()  # noqa: E712

    stats = pd.read_csv(DATA_DIR / f"{league}_team_match_stats.csv", parse_dates=["date"])

    home_np = stats[stats["h_a"] == "h"][["date", "team_title", "npxG"]].rename(
        columns={"team_title": "home_team", "npxG": "home_npxG"}
    )
    away_np = stats[stats["h_a"] == "a"][["date", "team_title", "npxG"]].rename(
        columns={"team_title": "away_team", "npxG": "away_npxG"}
    )

    df = matches.merge(home_np, on=["date", "home_team"], how="left")
    df = df.merge(away_np, on=["date", "away_team"], how="left")

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    missing = df["home_npxG"].isna().sum() + df["away_npxG"].isna().sum()
    if missing:
        print(f"  [{league}] warning: {missing} npxG values unmatched, filling with shot-based xG")
        df["home_npxG"] = df["home_npxG"].fillna(df["home_xG"])
        df["away_npxG"] = df["away_npxG"].fillna(df["away_xG"])

    return df[[
        "league", "season", "date", "home_team", "away_team",
        "home_goals", "away_goals", "home_xG", "away_xG",
        "home_npxG", "away_npxG",
    ]]


def main():
    frames = []
    for league in LEAGUES:
        print(f"Building {league} ...")
        frames.append(build_league(league))

    full = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    out_path = OUT_DIR / "matches_unified.csv"
    full.to_csv(out_path, index=False)
    print(f"\nWrote {len(full):,} matches to {out_path}")
    print(full.groupby("league")["date"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
