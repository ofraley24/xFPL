#!/usr/bin/env python3
"""Compare the high-end fixes on the development folds (2019-20..2024-25, starters) and apply the decision
rule fixed before the runs:

  variants: base | damp (within-match dependence for goals) | slope (walk-forward calibration slope for
            goals and assists) | damp + slope. All share the same minutes predictions and random numbers.
  rule:     choose the variant whose top-10-per-gameweek actual/xP is closest to 1, among variants with
            overall actual/xP within +-2% of 1 and central 50% / 80% PIT coverage within +-2 points of nominal.
            If no variant meets the overall band, the band becomes 'no worse than the baseline'.
            Ties (top-10 ratios within 0.01) go to the simpler variant (order: base, damp, slope, both).
Writes docs/VARIANT_COMPARISON.md and figures/fig7_variants.png.
Historical record: the variant predictions were produced by 13_xp_model.py before the high-end fixes of
docs/HIGH_END_PLAN.md; the --damp option was removed afterwards (kappa 0.93-0.97, not adopted), and the
final model replaced the linear slope with a spline calibration.
"""
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
VARIANTS = [("base", "dev"), ("damp", "dev_damp"), ("slope", "dev_slope"), ("damp + slope", "dev_damp_slope")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "axes.edgecolor": INK2})
rng = np.random.default_rng(SEED)


def ratio_ci(sub, n_boot=1000):
    """actual/xP with a fixture-cluster bootstrap 95% CI."""
    g = sub.groupby("fx")[["total_points", "xp"]].sum()
    a, p = g.total_points.to_numpy(), g.xp.to_numpy()
    b = rng.integers(0, len(a), size=(n_boot, len(a)))
    r = a[b].sum(1) / p[b].sum(1)
    return a.sum() / p.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def topk(d, k):
    return d[d.groupby(["season", "round"]).xp.rank(ascending=False, method="first") <= k]


def main():
    rows, brace = [], []
    data = {}
    for name, tag in VARIANTS:
        f = OUT / f"xp_predictions_{tag}.parquet"
        if not f.exists():
            print("missing", f)
            continue
        d = pd.read_parquet(f)
        d["fx"] = d.season + "_" + d.fixture.astype(str)
        data[name] = d
        u = np.random.default_rng(SEED).random(len(d))
        pit = d.cdf_lo + u * (d.cdf_hi - d.cdf_lo)
        r_all = ratio_ci(d)
        r10, r5, r20 = ratio_ci(topk(d, 10)), ratio_ci(topk(d, 5)), ratio_ci(topk(d, 20))
        rp = ratio_ci(d[d.value >= 80])
        top_dec = d[d.p_ge10 >= d.p_ge10.quantile(0.9)]
        rows.append({"variant": name, "overall actual/xP": r_all[0], "overall lo": r_all[1], "overall hi": r_all[2],
                     "top-20 actual/xP": r20[0], "top-10 actual/xP": r10[0], "top-10 lo": r10[1], "top-10 hi": r10[2],
                     "top-5 actual/xP": r5[0], "£8m+ actual/xP": rp[0],
                     "P(10+) top decile obs/pred": (top_dec.total_points >= 10).mean() / top_dec.p_ge10.mean(),
                     "PIT central 50%": ((pit > .25) & (pit < .75)).mean(), "PIT central 80%": ((pit > .1) & (pit < .9)).mean(),
                     "RMSE": np.sqrt(((d.total_points - d.xp) ** 2).mean()),
                     "within-GW Spearman": d.groupby(["season", "round"]).apply(
                         lambda g: spearmanr(g.xp, g.total_points).correlation).mean(),
                     "top-10 mean actual pts": topk(d, 10).total_points.mean()})
        a = d[d.position.isin(["MID", "FWD"])]
        q = pd.qcut(a.e_goals.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        for qq, g in a.groupby(q):
            brace.append({"variant": name, "quintile": qq, "E goals pred": g.e_goals.mean(), "E goals obs": g.goals_scored.mean(),
                          "P(2+) pred": g.p_g2.mean(), "P(2+) obs": (g.goals_scored >= 2).mean()})
    T, B = pd.DataFrame(rows), pd.DataFrame(brace)

    # ---- decision rule
    base = T[T.variant == "base"].iloc[0]
    band = (T["overall actual/xP"] - 1).abs() <= 0.02
    if not band.any():
        band = (T["overall actual/xP"] - 1).abs() <= (base["overall actual/xP"] - 1).__abs__() + 1e-9
        band_txt = "no variant within +-2% overall, so the band is 'no worse than the baseline'"
    else:
        band_txt = "overall within +-2%"
    pit_ok = ((T["PIT central 50%"] - 0.5).abs() <= 0.02) & ((T["PIT central 80%"] - 0.8).abs() <= 0.02)
    cand = T[band & pit_ok].copy()
    cand["dist"] = (cand["top-10 actual/xP"] - 1).abs()
    order = {n: i for i, (n, _) in enumerate(VARIANTS)}
    best = None
    if len(cand):
        m = cand.dist.min()
        tied = cand[cand.dist <= m + 0.01]
        best = min(tied.variant, key=order.get)

    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
    groups = ["overall", "top-20", "top-10", "top-5", "£8m+"]
    for i, (name, _) in enumerate(VARIANTS):
        if name not in data:
            continue
        d = data[name]
        vals = [ratio_ci(d), ratio_ci(topk(d, 20)), ratio_ci(topk(d, 10)), ratio_ci(topk(d, 5)), ratio_ci(d[d.value >= 80])]
        x = np.arange(len(groups)) + (i - 1.5) * 0.17
        r = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
        ax[0].errorbar(x, r, yerr=[r - lo, hi - r], fmt="o", ms=4, color=COLORS[i], ecolor=COLORS[i], elinewidth=1, label=name)
        b = B[B.variant == name]
        ax[1].plot(np.arange(5) + (i - 1.5) * 0.12, b["P(2+) pred"], "D", ms=4, color=COLORS[i], label=f"{name}: predicted")
    obs = B[B.variant == "base"]["P(2+) obs"].to_numpy()
    ax[1].plot(np.arange(5), obs, "_", ms=18, mew=2, color=INK, label="observed")
    ax[0].axhline(1, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax[0].set_xticks(np.arange(len(groups)), ["all starters", "top 20 per GW", "top 10 per GW", "top 5 per GW", "price £8m+"])
    ax[0].set_ylabel("actual / predicted points (95% CI)")
    ax[0].set_title("calibration overall and at the high end", loc="left", fontsize=9)
    ax[1].set_xticks(np.arange(5), ["Q1", "Q2", "Q3", "Q4", "Q5 (top)"])
    ax[1].set_xlabel("quintile of expected goals (MID/FWD starters)"), ax[1].set_ylabel("P(2+ goals)")
    ax[1].set_title("braces: predicted vs observed", loc="left", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower left"), ax[1].legend(fontsize=6.5, loc="upper left")
    for a in ax:
        a.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "fig7_variants.png", dpi=300); plt.close(fig)

    kap = []
    for name, tag in VARIANTS:
        c = OUT / f"xp_component_coefficients_{tag}.csv"
        if c.exists():
            cc = pd.read_csv(c)
            k = cc[(cc.component == "goal_open") & (cc.term == "log_kappa")]
            if len(k):
                kap.append(k.assign(kappa=np.exp(k.beta), variant=name)[["variant", "fold", "kappa", "se"]])
    lines = ["# High-end fixes: variant comparison (development 2019-20..2024-25, starters)\n",
             "Decision rule (fixed before the runs): closest top-10 actual/xP to 1, subject to the overall band and "
             f"PIT coverage within +-2 points; ties to the simpler variant. Band used: {band_txt}.\n",
             f"**Chosen variant: {best}**\n" if best else "**No variant met the constraints.**\n",
             T.round(4).to_markdown(index=False), "", "## Braces by quintile of expected goals (MID/FWD)\n",
             B.round(4).to_markdown(index=False), ""]
    if kap:
        lines += ["## Estimated within-match dependence (kappa, goals after a player's first goal)\n",
                  pd.concat(kap).round(3).to_markdown(index=False), ""]
    (DOCS / "VARIANT_COMPARISON.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
