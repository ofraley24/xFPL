#!/usr/bin/env python3
"""Score the horizon backtest against the pre-registered rule (docs/HORIZON_PLAN.md).

Input: data/analysis/xp_horizon.parquet (24_predict_horizons.py). Default: dev seasons 2019-20..2024-25 ->
docs/HORIZON_RESULTS.md. --sealed: 2025-26 only, scored ONCE after the dev result -> docs/HORIZON_RESULTS_sealed.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
H = 8
Z80, Z95 = norm.ppf(0.9), norm.ppf(0.975)


def balanced(d):
    """Target fixtures in gameweek >= H, player-fixtures present at every h = 1..H."""
    b = d[d["round"] >= H]
    n = b.groupby(["season", "element", "fixture"]).horizon.transform("nunique")
    return b[n == H]


def row_metrics(g):
    e = g.total_points - g.xp
    rho = [spearmanr(x.xp, x.total_points)[0] for _, x in g.groupby(["season", "round"]) if len(x) > 20]
    return pd.Series({
        "rows": len(g), "bias": e.mean(), "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
        "rank_corr_gw": np.nanmean(rho), "mean_xP": g.xp.mean(), "mean_actual": g.total_points.mean(),
        "cover_q10_q90": ((g.q10 <= g.total_points) & (g.total_points <= g.q90)).mean(),
        "cover_q25_q75": ((g.q25 <= g.total_points) & (g.total_points <= g.q75)).mean(),
        "P10_pred": g.p_ge10.mean(), "P10_obs": (g.total_points >= 10).mean()})


def windows(d, L):
    """Per (season, deadline t, player): sums over started fixtures in gameweeks t..t+L-1 predicted from t."""
    last = d.groupby("season")["round"].max()
    w = d[(d.horizon <= L) & (d.asof_round + L - 1 <= d.season.map(last))]
    w = w.assign(var_ale=w.var_total - w.var_epi, sd_epi=np.sqrt(w.var_epi.clip(lower=0)))
    s = w.groupby(["season", "asof_round", "element"]).agg(
        act=("total_points", "sum"), pred=("xp", "sum"), va=("var_ale", "sum"), ve=("var_epi", "sum"),
        se=("sd_epi", "sum"), n=("xp", "size"))
    r = s.act - s.pred
    z = {"correlated": r / np.sqrt(s.va + s.se ** 2), "independent": r / np.sqrt(s.va + s.ve),
         "aleatoric": r / np.sqrt(s.va)}
    out = {"windows": len(s), "mean_fixtures": s.n.mean(), "bias_per_window": r.mean()}
    for k, v in z.items():
        out[f"sd_z_{k}"] = v.std()
    zc = z["correlated"].abs()
    out["cover80"], out["cover95"] = (zc <= Z80).mean(), (zc <= Z95).mean()
    return pd.Series(out)


def main():
    sealed = "--sealed" in sys.argv
    d = pd.read_parquet(OUT / "xp_horizon.parquet")
    d = d[d.season == "2025-26"] if sealed else d[d.season.between("2019-20", "2024-25")]
    assert (d.started == 1).all()
    bal = balanced(d)
    rows_bal = bal.groupby("horizon").apply(row_metrics)
    rows_all = d.groupby("horizon").apply(row_metrics)
    win = pd.DataFrame({L: windows(d, L) for L in range(1, H + 1)}).T.rename_axis("L")
    pos = bal.groupby(["horizon", "position"]).apply(lambda g: (g.total_points - g.xp).mean()).unstack()
    dec = bal[bal.horizon.isin([1, 4, 8])].assign(
        dec=lambda x: x.groupby("horizon").xp.transform(lambda v: pd.qcut(v, 10, labels=False, duplicates="drop")))
    cal = dec.groupby(["horizon", "dec"]).agg(xp=("xp", "mean"), actual=("total_points", "mean")).unstack(0)
    per_season = bal.groupby(["season", "horizon"]).apply(lambda g: (g.total_points - g.xp).mean()).unstack()

    rule = pd.DataFrame({"bias": rows_bal.bias, "cover80_window_L=h": win.cover80.reindex(rows_bal.index).to_numpy()},
                        index=rows_bal.index)
    rule["bias_ok"] = rule.bias.abs() <= 0.10
    rule["cover_ok"] = rule["cover80_window_L=h"].between(0.77, 0.83)
    rule["SUPPORTED"] = rule.bias_ok & rule.cover_ok
    sup = 0
    for h in range(1, H + 1):
        if not rule.loc[h, "SUPPORTED"]:
            break
        sup = h
    label = "2025-26 (sealed, scored once)" if sealed else "dev seasons 2019-20..2024-25"
    lines = [f"# Horizon backtest results: {label}\n",
             "Rule and design: `docs/HORIZON_PLAN.md` (fixed before scoring). Script: `analysis/25_score_horizons.py`.\n",
             f"**Largest h with every horizon 1..h supported: {sup}.**\n",
             "## Decision rule (balanced panel for bias; windows of length L = h for coverage)\n",
             rule.round(4).to_markdown(), "",
             f"## Row metrics by horizon, balanced panel (target gameweek >= {H}, present at every h)\n",
             rows_bal.round(4).to_markdown(), "",
             "## Row metrics by horizon, all rows (populations differ by h)\n", rows_all.round(4).to_markdown(), "",
             "## Windows: sums over a player's started fixtures in gameweeks t..t+L-1, predicted from deadline t\n",
             win.round(4).to_markdown(), "",
             "## Bias (actual - xP) by position, balanced panel\n", pos.round(3).to_markdown(), "",
             "## Bias by season, balanced panel\n", per_season.round(3).to_markdown(), "",
             "## Calibration by xP decile (balanced panel; h = 1, 4, 8)\n", cal.round(3).to_markdown(), "",
             "The decomposition listed in the plan (player features vs ratings frozen) was not run.\n"]
    name = "HORIZON_RESULTS_sealed.md" if sealed else "HORIZON_RESULTS.md"
    (ROOT / "docs" / name).write_text("\n".join(lines))
    print("\n".join(lines[:5]))
    print(rule.round(4).to_string())


if __name__ == "__main__":
    main()
