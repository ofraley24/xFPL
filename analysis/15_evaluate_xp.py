#!/usr/bin/env python3
"""Evaluate the dev-period xP predictions (test seasons 2019-20..2024-25) and draw the figures.

Figures (figures/, 300 dpi):
  fig1_xp_reliability.png        binned xP vs mean actual points, 95% fixture-cluster bootstrap CIs
  fig2_calibration_by_feature.png mean predicted vs mean actual points across bins of 12 features
  fig3_component_reliability.png reliability of each component (goals, assists, clean sheets, ...)
  fig4_uncertainty.png           randomised PIT, interval coverage, spread-skill, season-total z-scores
  fig5_benchmark.png             accuracy against FPL's own xP and a form baseline, per season
Tables: docs/XP_EVALUATION.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
FIG = ROOT / "figures"
DOCS = ROOT / "docs"
SEED = 20261011
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "lines.linewidth": 1.6})
rng = np.random.default_rng(SEED)


def cluster_ci(df, y, n_boot=400):
    """95% CI for mean(y) with resampling of fixtures (players in a match share the match)."""
    g = df.groupby("fx")[y].agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy()
    if len(s) < 5:
        return np.nan, np.nan
    b = rng.integers(0, len(s), size=(n_boot, len(s)))
    m = s[b].sum(1) / c[b].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def binned(df, bin_col, pred, obs, order=None):
    rows = []
    for k, g in df.groupby(bin_col, observed=True):
        lo, hi = cluster_ci(g, obs)
        rows.append({"bin": k, "n": len(g), "pred": g[pred].mean(), "obs": g[obs].mean(), "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    if order is not None:
        t["bin"] = pd.Categorical(t["bin"], categories=order, ordered=True)
        t = t.sort_values("bin")
    return t


def reliability_panel(ax, df, pred, obs, nbins, title, xlabel="predicted", ylabel="observed mean"):
    d = df.copy()
    d["b"] = pd.qcut(d[pred].rank(method="first"), nbins, labels=False)
    t = binned(d, "b", pred, obs)
    lim = [0, max(t.pred.max(), t.hi.max()) * 1.05]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(t.pred, t.obs, yerr=[t.obs - t.lo, t.hi - t.obs], fmt="o", ms=3.5, color=C1, ecolor=C1, elinewidth=0.9,
                capsize=0, zorder=3)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
    return t


def main():
    import sys
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ("sealed" if "--sealed" in sys.argv else "dev")
    SUF = "" if tag == "dev" else f"_{tag}"
    cond = not tag.endswith("_uncond")   # default: xP GIVEN THE PLAYER STARTS (rows = starters only)
    FIG.mkdir(exist_ok=True)
    d = pd.read_parquet(OUT / f"xp_predictions_{tag}.parquet")
    d["fx"] = d.season + "_" + d.fixture.astype(str)
    d["app_pts"] = (d.minutes > 0).astype(int) + (d.minutes >= 60).astype(int)
    d["saves_pts"] = d.saves // 3
    d["played"] = (d.minutes > 0).astype(float)
    # form baseline and lagged points need history from the master (all rows, strictly earlier fixtures)
    m = pd.read_parquet(OUT / "master.parquet", columns=["season", "fixture", "element", "code", "kickoff_time",
                                                           "total_points", "m10", "started"])
    m = m.sort_values(["code", "kickoff_time"])
    m["form5"] = m.groupby("code").total_points.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    m["next_pts"] = m.groupby("code").total_points.shift(-1)   # the player's NEXT fixture: no forecast here can see it
    d = d.merge(m[["season", "fixture", "element", "form5", "next_pts"]], on=["season", "fixture", "element"], how="left")
    extra = []
    if cond:
        # fair baseline for the conditional target: points per start over the player's last 5 starts
        st = m[m.started == 1].copy()
        st["pps5"] = st.groupby("code").total_points.transform(lambda x: x.rolling(5, min_periods=1).mean())
        q = d[["season", "fixture", "element", "code", "kickoff_time"]].copy()
        q["kickoff_time"] = pd.to_datetime(q.kickoff_time)
        st["kickoff_time"] = pd.to_datetime(st.kickoff_time)
        q = pd.merge_asof(q.sort_values("kickoff_time"), st[["code", "kickoff_time", "pps5"]].sort_values("kickoff_time"),
                          on="kickoff_time", by="code", direction="backward", allow_exact_matches=False)
        d = d.merge(q[["season", "fixture", "element", "pps5"]], on=["season", "fixture", "element"], how="left")
        d["pps5"] = d.pps5.fillna(d.pps5.median())
        extra.append(("points per start (last 5)", "pps5"))
        base_file = OUT / f"xp_predictions_{tag}_uncond.parquet"
        if base_file.exists():
            b = pd.read_parquet(base_file, columns=["season", "fixture", "element", "xp"]).rename(columns={"xp": "xp_uncond"})
            d = d.merge(b, on=["season", "fixture", "element"], how="left")
            if d.xp_uncond.notna().mean() > 0.95:
                d["xp_uncond"] = d.xp_uncond.fillna(d.xp_uncond.median())
                extra.append(("unconditional model xP", "xp_uncond"))
    outside = m[(m.season.isin(d.season.unique())) & ~(m.m10.fillna(0) > 0)]
    seasons_txt = ", ".join(sorted(d.season.unique()))
    pdf_ = d.position
    rule_pts = ((d.minutes > 0).astype(int) + (d.minutes >= 60).astype(int)
                + pdf_.map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}) * d.goals_scored + 3 * d.assists
                + pdf_.map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}) * d.clean_sheets
                + pdf_.map({"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0}) * (d.goals_conceded // 2) + d.saves // 3
                + 5 * d.penalties_saved - 2 * d.penalties_missed - d.yellow_cards - 3 * d.red_cards - 2 * d.own_goals
                + d.bonus + 2 * ((d.season >= "2025-26") & (d.position != "GKP")
                                 & (d.defensive_contribution.fillna(0) >= np.where(d.position == "DEF", 10, 12))))
    kind = {"dev": "development 2019-20..2024-25, GIVEN THE PLAYER STARTS",
            "sealed": "2025-26 (opened earlier with the unconditional model), GIVEN THE PLAYER STARTS",
            "live": "2026-27 finished gameweeks (untouched test), GIVEN THE PLAYER STARTS"}.get(tag, tag)
    lines = [f"# xP evaluation ({kind}: {seasons_txt})\n",

             f"Scoring rules (incl. defensive contribution from 2025-26) reproduce actual points on "
             f"{(rule_pts == d.total_points).mean():.4%} of rows.\n",
             (f"Rows evaluated: {len(d):,} player-fixtures in which the player STARTED. The model conditions on the "
              "start; everything else is as of the gameweek deadline.\n" if cond else
              f"Rows evaluated (universe): {len(d):,}. Rows outside the universe (predicted 0): {len(outside):,}, "
              f"which scored {outside.total_points.sum():,} of {outside.total_points.sum() + d.total_points.sum():,} "
              f"points ({outside.total_points.sum() / (outside.total_points.sum() + d.total_points.sum()):.2%}).\n")]

    # ---------------- accuracy and benchmark
    def acc(sub, col):
        e = sub.total_points - sub[col]
        gw = sub.groupby(["season", "round"]).apply(lambda g: spearmanr(g[col], g.total_points).correlation)
        top = sub.groupby(["season", "round"]).apply(lambda g: g.nlargest(10, col).total_points.mean())
        return {"bias": -e.mean(), "RMSE": np.sqrt((e ** 2).mean()), "MAE": e.abs().mean(),
                "within-GW Spearman": gw.mean(), "top-10 by GW: mean actual pts": top.mean()}
    bench = d[d.xP.notna()].copy()
    no_fpl = bench.groupby(["season", "round"]).size().size < 3   # archived FPL xP missing (e.g. the live season)
    if no_fpl:
        bench = d.copy()
    tab = pd.DataFrame({"this model": acc(bench, "xp"), **({} if no_fpl else {"FPL xP (vaastav)": acc(bench, "xP")}),
                        "form: mean of last 5": acc(bench.assign(form5=bench.form5.fillna(0)), "form5"),
                        **{name: acc(bench, col) for name, col in extra}}).T
    per_season = []
    for s, g in bench.groupby("season"):
        for name, col in (("this model", "xp"), *(() if no_fpl else (("FPL xP", "xP"),)), ("form (last 5)", "form5"), *extra):
            gg = g.assign(form5=g.form5.fillna(0))
            nx = gg.dropna(subset=["next_pts"])
            per_season.append({"season": s, "model": name, "RMSE": np.sqrt(((gg.total_points - gg[col]) ** 2).mean()),
                               "Spearman": gg.groupby("round").apply(lambda x: spearmanr(x[col], x.total_points).correlation).mean(),
                               "Spearman next match": nx.groupby("round").apply(lambda x: spearmanr(x[col], x.next_pts).correlation).mean()})
    ps = pd.DataFrame(per_season)
    lines += [f"## Accuracy vs benchmarks (rows where the archived FPL xP exists: {', '.join(sorted(bench.season.unique()))})\n",
              tab.round(4).to_markdown(), ""]
    allacc = pd.DataFrame({"this model": acc(d, "xp"), **{name: acc(d, col) for name, col in extra}}).T
    lines += [f"All evaluated rows ({seasons_txt}):\n", allacc.round(4).to_markdown(), ""]

    # ---------------- Fig 1: overall reliability + histogram
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 5.6), height_ratios=[3, 1], sharex=True)
    t1 = reliability_panel(a1, d, "xp", "total_points", 25, "xP reliability (25 quantile bins, 95% CI)",
                           xlabel="", ylabel="mean actual points")
    a2.hist(d.xp, bins=60, color=C1, alpha=0.85, edgecolor="#fcfcfb", linewidth=0.3)
    a2.set_xlabel("predicted xP"), a2.set_ylabel("rows")
    a2.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / f"fig1_xp_reliability{SUF}.png", dpi=300); plt.close(fig)
    ci_cover = ((t1.lo <= t1.pred) & (t1.pred <= t1.hi)).mean()
    lines += [f"Fig 1: {ci_cover:.0%} of the 25 bins' 95% CIs contain the bin's mean prediction.\n"]

    # ---------------- Fig 2: calibration across feature bins
    q = lambda s, k, lab=None: pd.qcut(s.rank(method="first"), k, labels=lab)
    d["price band (£m)"] = pd.cut(d.value / 10, [0, 4.5, 5.5, 6.5, 8.0, 10.0, 20], labels=["≤4.5", "4.6-5.5", "5.6-6.5", "6.6-8", "8.1-10", ">10"])
    d["team λ quintile"] = q(d.xg_for_pi, 5, ["Q1 weak", "Q2", "Q3", "Q4", "Q5 strong"])
    d["opponent λ quintile"] = q(d.xg_against_pi, 5, ["Q1 weak opp", "Q2", "Q3", "Q4", "Q5 strong opp"])
    d["venue"] = np.where(d.was_home, "home", "away")
    d["P(plays) band"] = pd.cut(d.p_play, [-0.01, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], labels=["≤.2", ".2-.4", ".4-.6", ".6-.8", ".8-.9", ">.9"])
    d["minutes last match"] = pd.cut(d.m1.fillna(0), [-1, 0, 30, 59, 89, 90], labels=["0", "1-30", "31-59", "60-89", "90"])
    d["gameweek"] = pd.cut(d["round"], [0, 5, 13, 26, 33, 50], labels=["1-5", "6-13", "14-26", "27-33", "34+"])
    d["ownership quintile"] = q(d.selected, 5, ["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    d["form (last-5 pts)"] = pd.cut(d.form5.fillna(0), [-5, 1, 2, 3, 4.5, 6, 30], labels=["≤1", "1-2", "2-3", "3-4.5", "4.5-6", ">6"])
    d["fixtures in GW"] = np.where(d.fix_in_event >= 2, "double", "single")
    d["starts in last 10"] = pd.cut(d.st_n10.fillna(0), [-1, 3, 6, 8, 10], labels=["0-3", "4-6", "7-8", "9-10"]) \
        if "st_n10" in d else "n/a"
    feats = ["position", "price band (£m)", "team λ quintile", "opponent λ quintile", "venue",
             "starts in last 10" if cond else "P(plays) band",
             "minutes last match", "gameweek", "ownership quintile", "form (last-5 pts)", "season", "fixtures in GW"]
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.6), sharey=True)
    worst, tabs = [], {}
    for f in feats:
        order = ["GKP", "DEF", "MID", "FWD"] if f == "position" else None
        tabs[f] = binned(d, f, "xp", "total_points", order)
    lo_all = min((t.lo / t.pred).min() for t in tabs.values()); hi_all = max((t.hi / t.pred).max() for t in tabs.values())
    pad = 0.03
    for ax, f in zip(axes.ravel(), feats):
        t = tabs[f]
        x = np.arange(len(t))
        r, rl, rh = t.obs / t.pred, t.lo / t.pred, t.hi / t.pred
        ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.errorbar(x, r, yerr=[r - rl, rh - r], fmt="o", ms=4, color=C1, ecolor=C1, elinewidth=1.0, zorder=3)
        ax.set_xticks(x, [str(b) for b in t["bin"]], rotation=35, ha="right", fontsize=7)
        ax.set_title(f, loc="left", fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(lo_all - pad, hi_all + pad)
        for rr in t.itertuples():
            worst.append({"feature": f, "bin": str(rr.bin), "n": rr.n, "pred": rr.pred, "obs": rr.obs,
                          "ratio obs/pred": rr.obs / rr.pred, "CI contains pred": rr.lo <= rr.pred <= rr.hi})
    fig.supylabel("actual / predicted points (95% CI); 1 = calibrated", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / f"fig2_calibration_by_feature{SUF}.png", dpi=300); plt.close(fig)
    W = pd.DataFrame(worst)
    lines += [f"## Calibration across feature bins (Fig 2)\n\n{W['CI contains pred'].mean():.0%} of {len(W)} bins have "
              "a 95% CI that contains the mean prediction. Bins where it does not:\n",
              W[~W["CI contains pred"]].round(3).to_markdown(index=False), ""]

    # ---------------- Fig 3: component reliability
    comps = [("e_minutes", "minutes", "minutes"), ("e_app", "app_pts", "appearance points"),
             ("e_goals", "goals_scored", "goals"), ("e_assists", "assists", "assists"), ("e_cs", "clean_sheets", "clean sheet (60+ min)"),
             ("e_bonus", "bonus", "bonus points"), ("e_saves_pts", "saves_pts", "save points (GKP)"), ("e_yellow", "yellow_cards", "yellow cards")]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4))
    comp_rows = []
    for ax, (p, o, title) in zip(axes.ravel(), comps):
        sub = d[d.position == "GKP"] if "GKP" in title else d
        reliability_panel(ax, sub, p, o, 15, title)
        comp_rows.append({"component": title, "mean predicted": sub[p].mean(), "mean observed": sub[o].mean(),
                          "ratio": sub[o].mean() / sub[p].mean()})
    fig.tight_layout(); fig.savefig(FIG / f"fig3_component_reliability{SUF}.png", dpi=300); plt.close(fig)
    dcs = d[(d.season >= "2025-26") & (d.position != "GKP")]
    if len(dcs) and "e_dc" in dcs:
        hit = (dcs.defensive_contribution.fillna(0) >= np.where(dcs.position == "DEF", 10, 12)).astype(float)
        comp_rows.append({"component": "defensive contribution (P of +2)", "mean predicted": dcs.e_dc.mean(),
                          "mean observed": hit.mean(), "ratio": hit.mean() / dcs.e_dc.mean()})
    lines += ["## Component calibration in the large (Fig 3)\n", pd.DataFrame(comp_rows).round(4).to_markdown(index=False), ""]

    # ---------------- Fig 4: uncertainty
    u = rng.random(len(d))
    pit = d.cdf_lo + u * (d.cdf_hi - d.cdf_lo)
    sd_tot = np.sqrt(d.var_total)
    d["resid"] = d.total_points - d.xp
    d["sd_bin"] = pd.qcut(sd_tot.rank(method="first"), 15, labels=False)
    ss = d.assign(sd=sd_tot).groupby("sd_bin").agg(pred_sd=("sd", "mean"), rmse=("resid", lambda r: np.sqrt((r ** 2).mean())))
    # season totals per player: aleatoric adds, epistemic is correlated across a player's fixtures
    ps_ = d.groupby(["season", "code"]).agg(n=("xp", "size"), pred=("xp", "sum"), act=("total_points", "sum"),
                                            var_ale=("var_total", lambda v: 0), var_epi_ind=("var_epi", "sum"),
                                            sd_epi_sum=("var_epi", lambda v: np.sqrt(v).sum()))
    ps_["var_ale"] = d.assign(va=d.var_total - d.var_epi).groupby(["season", "code"]).va.sum()
    ps_ = ps_[ps_.n >= 10]
    z_ind = (ps_.act - ps_.pred) / np.sqrt(ps_.var_ale + ps_.var_epi_ind)
    z_cor = (ps_.act - ps_.pred) / np.sqrt(ps_.var_ale + ps_.sd_epi_sum ** 2)
    z_ale = (ps_.act - ps_.pred) / np.sqrt(ps_.var_ale)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].hist(pit, bins=20, range=(0, 1), color=C1, alpha=0.85, edgecolor="#fcfcfb", linewidth=0.4, density=True)
    ax[0].axhline(1, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax[0].set_title("randomised PIT, per player-fixture points", loc="left", fontsize=9)
    ax[0].set_xlabel("PIT"), ax[0].set_ylabel("density (uniform = calibrated)")
    lim = [0, ss.max().max() * 1.05]
    ax[1].plot(lim, lim, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax[1].plot(ss.pred_sd, ss.rmse, "o", ms=3.5, color=C1)
    ax[1].set_xlim(lim), ax[1].set_ylim(lim)
    ax[1].set_title("spread vs skill (15 bins)", loc="left", fontsize=9)
    ax[1].set_xlabel("predicted SD of points"), ax[1].set_ylabel("RMSE of xP")
    xs = np.linspace(-4, 4, 200)
    for z, c, lab in ((z_ale, C3, "aleatoric only"), (z_ind, C1, "+ epistemic, independent"), (z_cor, C2, "+ epistemic, fully correlated")):
        ax[2].hist(z.clip(-4, 4), bins=40, range=(-4, 4), density=True, histtype="step", color=c, lw=1.4,
                   label=f"{lab} (sd {z.std():.2f})")
    ax[2].plot(xs, norm.pdf(xs), color=INK2, lw=0.8, ls=(0, (4, 3)), label="N(0,1)")
    ax[2].set_title("player-season totals: standardised error", loc="left", fontsize=9)
    ax[2].set_xlabel("z = (actual - predicted) / predicted sd"), ax[2].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / f"fig4_uncertainty{SUF}.png", dpi=300); plt.close(fig)
    cov = lambda lo, hi: ((d[lo] <= d.total_points) & (d.total_points <= d[hi])).mean()
    cover = lambda z, a: (np.abs(z) <= norm.ppf(0.5 + a / 2)).mean()
    unc = pd.DataFrame({
        "quantity": ["PIT mean (0.5 ideal)", "PIT sd (0.289 ideal)", "share PIT in central 50%", "share PIT in central 80%",
                     "[q25,q75] covers actual (incl. ties)", "[q10,q90] covers actual (incl. ties)",
                     "season totals sd(z) aleatoric only", "season totals sd(z) +epistemic independent",
                     "season totals sd(z) +epistemic correlated", "season totals 80% coverage (correlated)",
                     "season totals 95% coverage (correlated)", "median epistemic share of variance (row level)"],
        "value": [pit.mean(), pit.std(), ((pit > .25) & (pit < .75)).mean(), ((pit > .1) & (pit < .9)).mean(),
                  cov("q25", "q75"), cov("q10", "q90"), z_ale.std(), z_ind.std(), z_cor.std(), cover(z_cor, .8),
                  cover(z_cor, .95), (d.var_epi / d.var_total.clip(lower=1e-9)).median()]})
    exp_bins = pd.cut(d.L540_time / 90, [-1, 5, 15, 30, 1e9], labels=["<5 full games", "5-15", "15-30", "30+"])
    epi = d.assign(sd_epi=np.sqrt(d.var_epi), share=d.var_epi / d.var_total.clip(lower=1e-9)).groupby(exp_bins).agg(
        n=("xp", "size"), mean_xp=("xp", "mean"), mean_sd_epistemic=("sd_epi", "mean"), mean_epistemic_share=("share", "mean"))
    lines += ["## Uncertainty diagnostics (Fig 4)\n", unc.round(4).to_markdown(index=False), "",
              "Epistemic uncertainty by amount of Understat history (should fall with experience):\n",
              epi.round(4).to_markdown(), ""]

    # ---------------- Fig 5: benchmark per season
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    models5 = [("this model", C1), ("FPL xP", C2), ("form (last 5)", C3)] + \
              [(nm, c) for (nm, _), c in zip(extra, ("#eda100", "#4a3aa7"))]
    for i, (name, c) in enumerate(models5):
        g = ps[ps.model == name]
        x = np.arange(len(g)) + (i - (len(models5) - 1) / 2) * 0.14
        ax[0].plot(x, g.RMSE, "o", color=c, ms=4, label=name)
        ax[1].plot(x, g.Spearman, "o", color=c, ms=4, label=name)
        ax[2].plot(x, g["Spearman next match"], "o", color=c, ms=4, label=name)
    lo_s = min(ps.Spearman.min(), ps["Spearman next match"].min()) - 0.03
    hi_s = max(ps.Spearman.max(), ps["Spearman next match"].max()) + 0.03
    for a, lab in ((ax[0], "RMSE, this match (lower is better)"), (ax[1], "rank correlation, THIS match"),
                   (ax[2], "rank correlation, NEXT match (unseen by all)")):
        a.set_xticks(np.arange(ps.season.nunique()), sorted(ps.season.unique()), rotation=30, ha="right")
        a.set_title(lab, loc="left", fontsize=9), a.grid(axis="x", visible=False)
    ax[1].set_ylim(lo_s, hi_s); ax[2].set_ylim(lo_s, hi_s)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"fig5_benchmark{SUF}.png", dpi=300); plt.close(fig)
    lines += ["## Benchmark validity\n",
              "The archived FPL xP ranks THIS match far better than this model but falls to the same level on the "
              "player's NEXT match, which no forecast can see; a genuine pre-deadline forecast should degrade by a "
              "similar amount to the others. This pattern indicates the archived value contains post-deadline "
              "information about its own gameweek, so it is not a fair same-gameweek benchmark. A clean FPL "
              "benchmark is being collected from deadline snapshots (ep_next) from 2026-10-11.\n"]
    lines += ["## Per-season benchmark (Fig 5)\n", ps.round(4).to_markdown(index=False), ""]
    # ---------------- Fig 6: the high end (players with a genuine prospect of points)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a_, col, thr, title in ((ax[0], "p_ge5", 5, "P(points >= 5)"), (ax[1], "p_ge10", 10, "P(points >= 10)")):
        dd = d.assign(hit=(d.total_points >= thr).astype(float))
        reliability_panel(a_, dd, col, "hit", 12, f"reliability of {title}", xlabel="predicted probability",
                          ylabel="observed frequency")
    tops, top_rows = [], []
    for k in (5, 10, 20, 40):
        sub = d[d.groupby(["season", "round"]).xp.rank(ascending=False, method="first") <= k]
        lo, hi = cluster_ci(sub, "total_points")
        tops.append((k, sub.xp.mean(), sub.total_points.mean(), lo, hi))
        top_rows.append({"top-k by xP per gameweek": k, "rows": len(sub), "mean xP": sub.xp.mean(),
                         "mean actual": sub.total_points.mean(), "actual/xP": sub.total_points.mean() / sub.xp.mean(),
                         "CI low": lo, "CI high": hi, "P(5+) predicted": sub.p_ge5.mean(),
                         "P(5+) observed": (sub.total_points >= 5).mean()})
    T = pd.DataFrame(top_rows)
    x = np.arange(len(T))
    ax[2].plot(x - 0.1, T["mean xP"], "D", color=C1, ms=4, ls="none", label="predicted xP")
    ax[2].errorbar(x + 0.1, T["mean actual"], yerr=[T["mean actual"] - T["CI low"], T["CI high"] - T["mean actual"]],
                   fmt="o", ms=4, color=C2, ecolor=C2, elinewidth=1.0, label="actual (95% CI)")
    ax[2].set_xticks(x, [f"top {k}" for k in T["top-k by xP per gameweek"]])
    ax[2].set_title("each gameweek's top-ranked players", loc="left", fontsize=9)
    ax[2].set_ylabel("mean points"), ax[2].grid(axis="x", visible=False), ax[2].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / f"fig6_high_end{SUF}.png", dpi=300); plt.close(fig)
    prem = d[d.value >= 80]
    hi_rows = []
    for nm, sub in (("price >= £8.0m", prem), ("xP >= 4", d[d.xp >= 4]), ("all evaluated rows", d)):
        lo, hi = cluster_ci(sub, "total_points")
        hi_rows.append({"subset": nm, "rows": len(sub), "mean xP": sub.xp.mean(), "mean actual": sub.total_points.mean(),
                        "actual/xP": sub.total_points.mean() / sub.xp.mean(), "CI low": lo, "CI high": hi})
    gwsp = d.groupby(["season", "round"]).apply(
        lambda g: spearmanr(g.nlargest(50, "xp").xp, g.nlargest(50, "xp").total_points).correlation).mean()
    lines += ["## High end of the distribution (Fig 6)\n", T.round(3).to_markdown(index=False), "",
              pd.DataFrame(hi_rows).round(3).to_markdown(index=False), "",
              f"Rank correlation within each gameweek's top 50 by xP: {gwsp:.3f}\n"]
    (DOCS / f"XP_EVALUATION{SUF}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
