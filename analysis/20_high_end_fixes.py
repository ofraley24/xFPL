#!/usr/bin/env python3
"""Evaluate the high-end / yellow-card fixes (docs/HIGH_END_PLAN.md) on the development folds and apply the
pre-registered rules.

  before   predictions before the fixes (data/analysis/before_fixes/xp_predictions_<mode>_slope.parquet)
  fixes    step 1 as pre-registered (cloglog cards + league yellow rate, within-match bonus, mean-preserving
           draws) + linear slope (development only: data/analysis/xp_predictions_dev_slope.parquet)
  final    the final model: step 1, shared goals conceded within a team, goals x position in the bonus
           ranking, spline calibration (data/analysis/xp_predictions_<mode>.parquet)

Writes docs/HIGH_END_RESULTS.md and figures/fig9_high_end_fixes.png.
Usage: 20_high_end_fixes.py [mode]   mode = dev (default) | sealed | live
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT, FIG, DOCS = ROOT / "data" / "analysis", ROOT / "figures", ROOT / "docs"
SEED = 20261011
G = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
C = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
GC = {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0}
COLORS = {"before": "#52514e", "fixes": "#2a78d6", "final": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "axes.edgecolor": INK2})
rng = np.random.default_rng(SEED)


def ratio_ci(sub, n_boot=1000):
    g = sub.groupby("fx")[["total_points", "xp"]].sum()
    a, p = g.total_points.to_numpy(), g.xp.to_numpy()
    b = rng.integers(0, len(a), size=(n_boot, len(a)))
    r = a[b].sum(1) / p[b].sum(1)
    return a.sum() / p.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def topk(d, k):
    return d[d.groupby(["season", "round"]).xp.rank(ascending=False, method="first") <= k]


def components(d):
    """Predicted and actual points by scoring component."""
    g = d.position.map(G).astype(float).where(~((d.season >= "2026-27") & (d.position == "GKP")), 10.0)
    c, gc = d.position.map(C), d.position.map(GC)
    thr = np.where(d.position == "DEF", 10, 12)
    dc_act = ((d.defensive_contribution >= thr) & (d.position != "GKP") & (d.season >= "2025-26")).astype(int)
    P = pd.DataFrame({"appearance": d.e_app, "goals": g * d.e_goals, "assists": 3 * d.e_assists, "clean sheet": c * d.e_cs,
                      "conceded": d.e_gc_pts, "saves": d.e_saves_pts, "bonus": d.e_bonus,
                      "cards": -d.e_yellow - 3 * d.e_red, "other": -2 * d.e_og - 2 * d.e_pen_miss + 5 * d.e_pen_saved,
                      "def. contr.": 2 * d.e_dc})
    A = pd.DataFrame({"appearance": (d.minutes > 0).astype(int) + (d.minutes >= 60), "goals": g * d.goals_scored,
                      "assists": 3 * d.assists, "clean sheet": c * d.clean_sheets, "conceded": gc * (d.goals_conceded // 2),
                      "saves": d.saves // 3, "bonus": d.bonus, "cards": -d.yellow_cards - 3 * d.red_cards,
                      "other": -2 * d.own_goals - 2 * d.penalties_missed + 5 * d.penalties_saved, "def. contr.": 2 * dc_act})
    return P, A


def summarise(name, d):
    u = np.random.default_rng(SEED).random(len(d))
    pit = d.cdf_lo + u * (d.cdf_hi - d.cdf_lo)
    r_all, r10, r5, r20 = ratio_ci(d), ratio_ci(topk(d, 10)), ratio_ci(topk(d, 5)), ratio_ci(topk(d, 20))
    rp = ratio_ci(d[d.value >= 80])
    fxs = d.groupby(["season", "fixture"]).agg(pb=("e_bonus", "sum"), ob=("bonus", "sum"))
    q = pd.qcut(fxs.pb.rank(method="first"), 5, labels=False)
    bq = fxs.groupby(q)[["pb", "ob"]].mean()
    t10 = topk(d, 10)
    return {"variant": name, "overall actual/xP": r_all[0], "overall lo": r_all[1], "overall hi": r_all[2],
            "top-20": r20[0], "top-10": r10[0], "top-10 lo": r10[1], "top-10 hi": r10[2], "top-5": r5[0], "£8m+": rp[0],
            "PIT 50%": ((pit > .25) & (pit < .75)).mean(), "PIT 80%": ((pit > .1) & (pit < .9)).mean(),
            "RMSE": np.sqrt(((d.total_points - d.xp) ** 2).mean()),
            "within-GW Spearman": d.groupby(["season", "round"]).apply(
                lambda g: spearmanr(g.xp, g.total_points).correlation).mean(),
            "top-10 mean actual": t10.total_points.mean(),
            "yellow obs/pred": d.yellow_cards.sum() / d.e_yellow.sum(),
            "bonus/fixture lowest quintile pred|obs": f"{bq.pb.iloc[0]:.2f} | {bq.ob.iloc[0]:.2f}",
            "bonus/fixture highest quintile pred|obs": f"{bq.pb.iloc[-1]:.2f} | {bq.ob.iloc[-1]:.2f}",
            "top-10 bonus pred|obs": f"{t10.e_bonus.mean():.3f} | {t10.bonus.mean():.3f}"}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    tags = {"before": OUT / "before_fixes" / f"xp_predictions_{mode}_slope.parquet",
            "fixes": OUT / f"xp_predictions_{mode}_slope.parquet",
            "final": OUT / f"xp_predictions_{mode}.parquet"}
    data = {}
    for k, f in tags.items():
        if f.exists():
            d = pd.read_parquet(f)
            d["fx"] = d.season + "_" + d.fixture.astype(str)
            data[k] = d
    T = pd.DataFrame([summarise(k, d) for k, d in data.items()])

    # component gaps (predicted - actual points per player) in the top 10 per gameweek
    gaps = {}
    for k, d in data.items():
        P, A = components(d)
        m = d.index.isin(topk(d, 10).index)
        gaps[k] = (P[m].mean() - A[m].mean())
    Gp = pd.DataFrame(gaps)
    Gp = Gp[(Gp.abs() > 0.0005).any(axis=1)]

    # yellow cards by season
    Y = pd.DataFrame({k: d.groupby("season").apply(lambda g: g.yellow_cards.sum() / g.e_yellow.sum()) for k, d in data.items()})

    # pre-registered rules
    lines = [f"# High-end and yellow-card fixes: results ({mode})\n",
             "Pre-registered plan: docs/HIGH_END_PLAN.md. Starters only; ratios are actual/predicted points with "
             "fixture-cluster bootstrap 95% intervals.\n"]
    if mode == "dev" and {"before", "fixes"} <= set(data):
        b, f = T.set_index("variant").loc["before"], T.set_index("variant").loc["fixes"]
        ok1 = (abs(f["overall actual/xP"] - 1) <= abs(b["overall actual/xP"] - 1) + 0.01 and
               abs(f["PIT 50%"] - 0.5) <= abs(b["PIT 50%"] - 0.5) + 0.02 and abs(f["PIT 80%"] - 0.8) <= abs(b["PIT 80%"] - 0.8) + 0.02)
        lines.append(f"**Step 1 (correctness fixes): {'adopted' if ok1 else 'NOT adopted'}.**\n")
        if "final" in data:
            c = T.set_index("variant").loc["final"]
            band = lambda r: abs(r["overall actual/xP"] - 1) <= 0.02 or abs(r["overall actual/xP"] - 1) <= abs(f["overall actual/xP"] - 1)
            pit = lambda r: abs(r["PIT 50%"] - 0.5) <= 0.02 and abs(r["PIT 80%"] - 0.8) <= 0.02
            cand = {k: abs(r["top-10"] - 1) for k, r in (("fixes", f), ("final", c)) if band(r) and pit(r)}
            best = None
            if cand:
                best = "final" if ("final" in cand and cand["final"] < cand.get("fixes", 9) - 0.01) else (
                    "fixes" if "fixes" in cand else "final")
            lines.append(f"**Step 2 (calibration): chosen = {'linear slope' if best == 'fixes' else 'spline' if best else 'none'}.**\n")
    lines += [T.round(4).to_markdown(index=False), "",
              "## Top 10 per gameweek: predicted minus actual points by component\n", Gp.round(3).to_markdown(), "",
              "## Yellow cards, actual/predicted by season\n", Y.round(3).to_markdown(), ""]
    sfx = "" if mode == "dev" else f"_{mode}"
    (DOCS / f"HIGH_END_RESULTS{sfx}.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # figure: high-end calibration by variant, top-10 component gaps
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
    groups = ["all starters", "top 20 per GW", "top 10 per GW", "top 5 per GW", "price £8m+"]
    for i, (k, d) in enumerate(data.items()):
        vals = [ratio_ci(d), ratio_ci(topk(d, 20)), ratio_ci(topk(d, 10)), ratio_ci(topk(d, 5)), ratio_ci(d[d.value >= 80])]
        x = np.arange(len(groups)) + (i - (len(data) - 1) / 2) * 0.2
        r = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
        ax[0].errorbar(x, r, yerr=[r - lo, hi - r], fmt="o", ms=4, color=COLORS[k], elinewidth=1, label=k)
    ax[0].axhline(1, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax[0].set_xticks(np.arange(len(groups)), groups)
    ax[0].set_ylabel("actual / predicted points (95% CI)")
    ax[0].set_title("calibration overall and at the high end", loc="left", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower left")
    y = np.arange(len(Gp))
    for i, k in enumerate(Gp.columns):
        ax[1].barh(y + (i - (len(Gp.columns) - 1) / 2) * 0.27, Gp[k], height=0.25, color=COLORS[k], label=k)
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_yticks(y, Gp.index)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("predicted − actual points per player (top 10 per GW)")
    ax[1].set_title("where the top-10 gap comes from", loc="left", fontsize=9)
    ax[1].legend(fontsize=7, loc="lower right")
    for a in ax:
        a.grid(axis="x" if a is ax[0] else "y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / f"fig9_high_end_fixes{sfx}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
