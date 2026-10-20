#!/usr/bin/env python3
"""Full xP model: component models + nested Monte-Carlo assembly with explicit uncertainty.

TARGET: FPL points GIVEN THAT THE PLAYER STARTS (--unconditional: all players with minutes in their previous
<= 10 fixtures). Walk-forward: development folds 2019-20 .. 2024-25, each trained on all earlier seasons
(>= 2016-17); --sealed scores 2025-26 and --live 2026-27, each trained on everything before it.
Options: --slope (linear calibration instead of the spline), --no-cal (no calibration), --reuse-minutes
(reuse the cached minutes-model predictions). Output: data/analysis/xp_predictions_<mode>[_variant].parquet.

COMPONENTS (statsmodels GLMs unless stated; Poisson with offset log(minutes/90); the coefficient on
log(minutes/90) lets rates scale non-linearly with minutes)
  minutes      8-bucket distribution, bagged gradient boosting (B fixture-bootstrap members)
  goals (open) Understat non-penalty goals; team lambda, shot-share and shot-quality FRACTIONS (split-half
               empirical Bayes), opponent lambda, home, price, ownership
  penalties    attempts (taker fraction) x conversion (Beta posterior); misses = attempts - scored
  assists      team lambda, xA fraction (split-half EB), opponent lambda, home, price, ownership
  conceded     goals conceded while on the pitch; clean sheet = 60+ min and 0 conceded
               (one full-match draw per team, shared by team-mates and thinned to their minutes)
  saves, penalty saves (GKP); own goals (all)
  yellow, red  binary GLMs, complementary log-log link (P = 1 - exp(-lambda)); yellow also uses the league
               yellow rate over the previous 365 days
  calibration  walk-forward: log E[goals], log E[assists] = natural cubic spline in the log point-estimate
               index, fitted on earlier out-of-sample seasons only
  bonus        ranked WITHIN the match: Plackett-Luce on the BPS order, scores linear in the simulated match
               events (goals, assists, clean sheet, saves, conceded ...) + the player's lagged BPS rate;
               3/2/1 to the top three in every draw, with BPS ties and substitutes at training rates, so each
               match hands out the right bonus total and team-mates compete for it

UNCERTAINTY
  epistemic: J parameter draws. Each draw takes (i) GLM coefficients ~ MVN(beta_hat, Cov), (ii) the
             player's talent fractions from their EB posterior (sd = sqrt(sigma2/(T+k))), (iii) team
             log-lambda + N(0, s_team^2) (one draw per team and match: a team's attack draw is its opponent's
             defence draw), s_team from over-dispersion of team goals around lambda in training, (iv) a
             minutes-ensemble member, (v) penalty conversion ~ Beta, (vi) bonus-model and calibration
             coefficients ~ MVN. Draws on the log scale are centred (- sd^2 beta^2 / 2) so mean rates are kept
  aleatoric: R outcome draws per parameter draw (minutes, events, bonus)
  Per row: xP = mean of J*R points; var_epi = Var_j(mean_r) - E_j(Var_r)/R; var_total = Var(all).
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis"
sp = importlib.util.spec_from_file_location("eb", ROOT / "analysis" / "08_split_half_eb.py")
EB = importlib.util.module_from_spec(sp)
sp.loader.exec_module(EB)

SEED = 20261011
TEST = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
POS = ["GKP", "DEF", "MID", "FWD"]
EDGES = [0.5, 14.5, 29.5, 44.5, 59.5, 74.5, 89.5]
K = 8
B_MIN = 8          # minutes ensemble members
J, R = 40, 25      # parameter draws x outcome draws
RULES = {"goal": {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}, "cs": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
         "gc_per2": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0}}
MIN_FEATS = ["m1", "m3", "m5", "m10", "f_play5", "f_long5", "f_full5", "ewm3", "absent_streak", "fix_since_app",
             "days_since_app", "std_mpf", "prev_season_mpf", "career_apps", "n_prior_rows", "price_rank",
             "price_gap_top", "own_share_team", "m5_rank_pos", "n_active_pos", "value", "selected_log", "home",
             "rest_days", "opp_rest_days", "next_gap_days", "fix_in_event", "supremacy", "total_goals",
             "season_progress", "L540_log", "L180_log", "us_days_log", "five_subs", "bcd",
             "pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]


# ------------------------------------------------------------------ data
def load(max_season="2024-25"):
    """Development runs read seasons up to 2024-25 only; the sealed run (--sealed) also reads 2025-26."""
    d = pd.read_parquet(OUT / "master.parquet")
    d = d[(d.season >= "2016-17") & (d.season <= max_season)].reset_index(drop=True)
    return prepare(d)


def prepare(d, lg_yel=None):
    """Model columns for rows with the master-table schema. lg_yel: log league yellow rate by deadline
    (Series); by default computed from the rows themselves (which must then be the full history)."""
    d["bucket"] = np.digitize(d.minutes.to_numpy(), EDGES)
    d["home"] = d.was_home.astype(float)
    d["selected_log"] = np.log1p(d.selected)
    d["L540_log"] = np.log1p(d.L540_time / 90)
    d["L180_log"] = np.log1p(d.L180_time / 90)
    d["us_days_log"] = np.log1p(d.us_days_since.fillna(3650))
    for p in POS:
        d[f"pos_{p}"] = (d.position == p).astype(float)
    d["universe"] = d.m10.fillna(0) > 0
    d["e"] = d.minutes / 90
    d["log_lamf"] = np.log(d.xg_for_pi)
    d["log_lama"] = np.log(d.xg_against_pi)
    d["log_price"] = np.log(d.value.clip(lower=35))
    d["log_sel"] = np.log1p(d.selected)
    pm = pd.read_csv(ROOT / "data" / "fpl" / "fpl_player_match.csv", usecols=["season", "fixture", "team_h_score",
                                                                              "team_a_score"]).drop_duplicates(["season", "fixture"])
    d = d.merge(pm, on=["season", "fixture"], how="left")
    d["team_goals"] = np.where(d.was_home, d.team_h_score, d.team_a_score)
    d["log_lg_yel"] = league_yellow_rate(d) if lg_yel is None else d.deadline.map(lg_yel).to_numpy()
    return d


def league_yellow_rate(d, days=365):
    """log league yellow cards per 90 over the `days` before each row's deadline (kickoffs strictly earlier).
    Deadlines with under ~50 matches of history (early 2016-17, training rows only) take the first value
    that has it."""
    k = d.kickoff_time.to_numpy()
    o = np.argsort(k, kind="mergesort")
    ks = k[o]
    cy = np.concatenate([[0.0], np.cumsum(d.yellow_cards.to_numpy(float)[o])])
    cm = np.concatenate([[0.0], np.cumsum(d.minutes.to_numpy(float)[o])])
    dl = np.unique(d.deadline.to_numpy())
    hi = np.searchsorted(ks, dl, side="left")
    lo = np.searchsorted(ks, dl - np.timedelta64(days, "D"), side="left")
    mins = cm[hi] - cm[lo]
    rate = pd.Series(np.where(mins >= 50 * 22 * 90, 90 * (cy[hi] - cy[lo]) / np.maximum(mins, 1), np.nan), index=dl).bfill()
    return np.log(rate.reindex(d.deadline.to_numpy()).to_numpy())


def shrunk_log(num, den, prior, k):
    return np.log(np.clip((num + prior * k) / (den + k), 1e-5, None))


def fixed_features(df, pri):
    """Non-EB player features, shrunk with fixed pseudo-counts and TRAINING priors `pri`."""
    pmap = lambda key: df.position.map(pri[key]).to_numpy()
    F = pd.DataFrame(index=df.index)
    F["log_fr_pen"] = shrunk_log(df.L540_pen_att, df.L540_team_pen_on, pmap("pen"), 1.5)
    F["log_yel90"] = shrunk_log(df.L365_f_yel, df.L365_f_min / 90, pmap("yel"), 10.0)
    F["log_red90"] = shrunk_log(df.L365_f_red, df.L365_f_min / 90, pmap("red"), 20.0)
    F["log_bps90"] = shrunk_log(df.L365_f_bps, df.L365_f_min / 90, pmap("bps"), 5.0)
    F["log_sv90"] = shrunk_log(df.L365_f_saves, df.L365_f_min / 90, pmap("sv"), 5.0)
    return F


def priors(tr):
    a = tr[tr.minutes > 0]
    pri = {}
    for key, num, den, scale in (("pen", "L540_pen_att", "L540_team_pen_on", 1), ("yel", "L365_f_yel", "L365_f_min", 90),
                                 ("red", "L365_f_red", "L365_f_min", 90), ("bps", "L365_f_bps", "L365_f_min", 90),
                                 ("sv", "L365_f_saves", "L365_f_min", 90)):
        g = a.groupby("position")[[num, den]].sum()
        pri[key] = (scale * g[num] / g[den].clip(lower=1e-9)).to_dict()
    return pri


def eb_features(df, kmap):
    """Posterior mean (log) and posterior sd (log scale) of each EB talent fraction."""
    F, S = pd.DataFrame(index=df.index), pd.DataFrame(index=df.index)
    for name, (_, _, nc, dc, feat, _) in EB.STATS.items():
        k = df.position.map({p: kmap[(name, p)][0] for p in POS}).to_numpy()
        m = df.position.map({p: kmap[(name, p)][1] for p in POS}).to_numpy()
        s2 = df.position.map({p: kmap[(name, p)][3] for p in POS}).to_numpy()
        post = (df[nc].to_numpy() + k * m) / (df[dc].to_numpy() + k)
        post = np.clip(post, 1e-5, None)
        var = s2 / (df[dc].to_numpy() + k)
        F[feat] = np.log(post)
        S[feat] = np.clip(np.sqrt(np.maximum(var, 0)) / post, 0, 2.0)   # delta-method sd on log scale
    return F, S


def estimate_k_full(H, smax):
    """As EB.estimate_k but also returns the noise variance sigma2 (needed for posterior sds)."""
    H = H[H.season <= smax]
    out = {}
    for name, (num, den, *_rest, tmin) in EB.STATS.items():
        for p in POS:
            h = H[H.grp == p]
            To, Te = h[f"{den}_o"].to_numpy(), h[f"{den}_e"].to_numpy()
            ok = (To >= tmin) & (Te >= tmin)
            mm = (h[f"{num}_o"].sum() + h[f"{num}_e"].sum()) / max(h[f"{den}_o"].sum() + h[f"{den}_e"].sum(), 1e-9)
            if ok.sum() < 30:
                out[(name, p)] = (1e6, mm, 0, 0.0)
                continue
            To, Te = To[ok], Te[ok]
            ro, re = h[f"{num}_o"].to_numpy()[ok] / To, h[f"{num}_e"].to_numpy()[ok] / Te
            s2 = ((ro - re) ** 2).sum() / (1 / To + 1 / Te).sum()
            W = 1 / (1 / To + 1 / Te)
            t2 = (W * (ro - mm) * (re - mm)).sum() / W.sum()
            out[(name, p)] = ((s2 / t2) if t2 > 0 else 1e6, mm, int(ok.sum()), s2)
    return out


# ------------------------------------------------------------------ GLM components
GLM_SPECS = {
    # goalkeepers almost never score or take penalties: fitting them in causes separation (infinite
    # coefficients); they are excluded and get a small empirical rate instead (gk_rate)
    "goal_open": dict(rows="app", y="tgt_np_goals", exclude_gk=True,
                      x=["pos_MID", "pos_FWD", "log_lamf", "log_fr_shots_540", "log_quality_540", "log_price",
                         "log_sel", "home", "log_lama"]),
    "pen_att": dict(rows="app", y="tgt_pen_att", exclude_gk=True,
                    x=["pos_MID", "pos_FWD", "log_lamf", "log_lama", "log_fr_pen"]),
    "assist": dict(rows="app", y="assists", x=["pos_DEF", "pos_MID", "pos_FWD", "log_lamf", "log_fr_xa_540", "log_price",
                                               "log_sel", "home", "log_lama"]),
    "conceded": dict(rows="app", y="goals_conceded", x=["log_lama", "log_lamf", "home"]),
    "saves": dict(rows="gk", y="saves", x=["log_lama", "log_lamf", "home", "log_sv90"]),
    # keepers nearly always play 90: the minutes term is unidentified (estimate +14), so it is dropped
    "pen_saved": dict(rows="gk", y="penalties_saved", x=["log_lama"], logmin=False),
    # cards are 0/1 per match (a second yellow is a red): binary GLM with cloglog link, P = 1 - exp(-lambda),
    # which is exactly what the simulator draws (Poisson(lambda) > 0)
    "yellow": dict(rows="app", y="yellow_cards", binary=True,
                   x=["pos_DEF", "pos_MID", "pos_FWD", "log_yel90", "log_lama", "home", "log_lg_yel"]),
    # a red card SHORTENS minutes (reverse causality: estimate -1.2), so no minutes elasticity; no position
    # dummies (rare event, separation)
    "red": dict(rows="app", y="red_cards", x=["log_red90", "log_yel90"], logmin=False, binary=True),
    "own_goal": dict(rows="app", y="own_goals", x=["pos_DEF", "pos_MID", "pos_FWD", "log_lama"]),
}
EB_COLS = ["log_fr_shots_540", "log_quality_540", "log_fr_xa_540"]
LAMBDA_COLS = ["log_lamf", "log_lama"]


def fit_glm(df, spec):
    gk = df[df.position == "GKP"]
    if spec.get("exclude_gk"):
        df = df[df.position != "GKP"]
    X = sm.add_constant(df[spec["x"]].astype(float), has_constant="add")
    if spec.get("logmin", True):
        X["log_minutes"] = np.log(df.e)
    y = df[spec["y"]].astype(float)
    if spec.get("binary"):
        fam = sm.families.Binomial(link=sm.families.links.CLogLog())
        y = (y > 0).astype(float)
    else:
        fam = sm.families.Poisson()
    res = sm.GLM(y, X, family=fam, offset=np.log(df.e)).fit()
    out = {"names": list(X.columns), "beta": res.params.to_numpy(), "cov": res.cov_params().to_numpy()}
    if spec.get("exclude_gk"):
        out["gk_rate"] = (gk[spec["y"]].sum() + 0.5) / (gk.e.sum() + 50.0)   # per 90, lightly smoothed
    return out


def expected_hat(fit, X, Pm, mbar):
    """Point-estimate expected count over the minutes distribution (index used by the calibration slope)."""
    names = fit["names"]
    cols = [c for c in names if c != "log_minutes"]
    b = fit["beta"][: len(names)]
    eta = X[cols].to_numpy(float) @ b[[names.index(c) for c in cols]]
    blm = b[names.index("log_minutes")] if "log_minutes" in names else 0.0
    m = np.clip(mbar[1:], 1e-6, None) / 90.0
    e = (Pm[:, 1:] * np.exp(eta[:, None] + (1 + blm) * np.log(m)[None, :])).sum(1)
    if "gk_rate" in fit:
        gk = (X.position == "GKP").to_numpy()
        e[gk] = fit["gk_rate"] * (Pm[gk, 1:] * m[None, :]).sum(1)
    return e


def cal_design(li, kind):
    """Calibration basis on the pool's log index `li`: (B, patsy design matrix or None)."""
    if kind == "linear":
        return sm.add_constant(li), None
    lo, hi = li.min(), li.max()
    kn = tuple(np.quantile(li, [0.5, 0.9, 0.98]))
    D = dmatrix(f"cr(x, knots={kn}, lower_bound={lo}, upper_bound={hi})", {"x": li}, return_type="matrix")
    return np.asarray(D), D


def fit_calibration(pool, comp_col, target, kind="linear"):
    """Walk-forward calibration of a component's point-estimate index on earlier out-of-sample rows:
    log E[observed] = f(log index), f linear (a + b log index) or a natural cubic spline with knots at the
    50/90/98% quantiles of the pool's log index (linear beyond its range). Returns the coefficients, their
    covariance and what calibration_basis needs (the pool's log index rebuilds the spline basis exactly)."""
    p_ = pool[(pool[comp_col] > 0) & pool[target].notna()]
    li = np.log(p_[comp_col].to_numpy())
    B, _ = cal_design(li, kind)
    r = sm.GLM(p_[target].to_numpy(float), B, family=sm.families.Poisson()).fit()
    return {"kind": kind, "params": np.asarray(r.params), "cov": np.asarray(r.cov_params()), "li_pool": li}


def calibration_basis(cal, idx_new):
    """The calibration basis evaluated at new rows' point-estimate index."""
    li_new = np.log(np.clip(idx_new, 1e-6, None))
    if cal["kind"] == "linear":
        return sm.add_constant(li_new, has_constant="add")
    li = cal["li_pool"]
    _, D = cal_design(li, cal["kind"])
    lo, hi = li.min(), li.max()
    # beyond the pool's range: continue linearly in log index from the boundary (slope from the edge)
    xc = np.clip(li_new, lo, hi)
    B_new = np.asarray(build_design_matrices([D.design_info], {"x": xc})[0])
    h = 1e-3 * (hi - lo)
    at = lambda x: np.asarray(build_design_matrices([D.design_info], {"x": np.array([x])})[0])
    d_lo, d_hi = (at(lo + h) - at(lo)) / h, (at(hi) - at(hi - h)) / h
    return B_new + np.minimum(li_new - lo, 0)[:, None] * d_lo + np.maximum(li_new - hi, 0)[:, None] * d_hi


def fit_slope(pool, comp_col, target, idx_new, kind="linear"):
    """fit_calibration + calibration_basis in one call: (coefficients, covariance, basis at the new rows)."""
    cal = fit_calibration(pool, comp_col, target, kind)
    return cal["params"], cal["cov"], calibration_basis(cal, idx_new)


BONUS_X = ["g1", "g2p", "a1", "a2p", "cs_def", "cs_mid", "m60", "sv3", "conc2p", "pensv", "yel", "pos_DEF", "pos_MID",
           "pos_FWD", "log_bps90", "goals_mid", "goals_fwd"]


def bonus_design(goals, assists, cs, m60, sv3, conc, pensv, yel, pos_def, pos_mid, pos_fwd, log_bps90):
    """Event features for the conditional bonus model (vectorised; works on simulation arrays too)."""
    gk_or_def = 1 - pos_mid - pos_fwd
    return np.stack([np.minimum(goals, 1), np.maximum(goals - 1, 0), np.minimum(assists, 1), np.maximum(assists - 1, 0),
                     cs * gk_or_def, cs * pos_mid, m60, sv3, np.maximum(conc - 1, 0) * gk_or_def, pensv, yel,
                     pos_def, pos_mid, pos_fwd, log_bps90,
                     goals * pos_mid, goals * pos_fwd], axis=-1)   # BPS per goal: GKP/DEF 12, MID 18, FWD 24


N_SLOT = 5   # BPS positions tracked for bonus (ties and substitutes can push the third bonus to position 5)


def bonus_rows(a, F):
    """Actual-event design for the ranking model on appearances `a` (rows of training or earlier seasons)."""
    pdf = a.position
    f = lambda c: a[c].to_numpy(float)
    pm, pf = (pdf == "MID").to_numpy(float), (pdf == "FWD").to_numpy(float)
    Xb = bonus_design(f("goals_scored"), f("assists"), f("clean_sheets"), (a.minutes >= 60).to_numpy(float),
                      (a.saves // 3).to_numpy(float), f("goals_conceded"), f("penalties_saved"), f("yellow_cards"),
                      (pdf == "DEF").to_numpy(float), pm, pf, F.loc[a.index, "log_bps90"].to_numpy())
    fx = (a.season + "_" + a.fixture.astype(str)).to_numpy()
    return Xb, fx


def fit_pl(X, fx, bps, offset=None, prior_sd=10.0):
    """Plackett-Luce (rank-ordered logit) for the top three of the BPS order within each fixture; score = X beta
    + offset. Damped Newton with a weak N(0, prior_sd^2) prior (braces nearly always top the BPS)."""
    offset = np.zeros(len(X)) if offset is None else offset
    o = np.lexsort((-bps, fx))
    X, fc, off = X[o], fx[o], offset[o]
    starts = np.flatnonzero(np.r_[True, fc[1:] != fc[:-1]])
    sizes = np.diff(np.r_[starts, len(fc)])
    ok = sizes >= 3
    G, M, p = ok.sum(), sizes.max(), X.shape[1]
    Xg, Og, mask = np.zeros((G, M, p)), np.zeros((G, M)), np.zeros((G, M), bool)
    for g, (s0, n0) in enumerate(zip(starts[ok], sizes[ok])):
        Xg[g, :n0], Og[g, :n0], mask[g, :n0] = X[s0:s0 + n0], off[s0:s0 + n0], True   # rows 0-2 = BPS 1st-3rd
    prior_prec = 1 / prior_sd ** 2

    def pl(beta, deriv=True):
        ll, grad, H = -0.5 * prior_prec * beta @ beta, -prior_prec * beta, prior_prec * np.eye(p)
        avail = mask.copy()
        s = Xg @ beta + Og
        for k in range(3):
            sk = np.where(avail, s, -np.inf)
            mx = sk.max(1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(sk - mx).sum(1))
            ll += (s[:, k] - lse).sum()
            if deriv:
                pk = np.exp(sk - lse[:, None])
                xbar = np.einsum("gm,gmp->gp", pk, Xg)
                grad += (Xg[:, k] - xbar).sum(0)
                H += np.einsum("gm,gmp,gmq->pq", pk, Xg, Xg) - xbar.T @ xbar
            avail[:, k] = False
        return ll, grad, H

    beta = np.zeros(p)
    ll, grad, H = pl(beta)
    for _ in range(100):
        step, t = np.linalg.solve(H, grad), 1.0
        while pl(beta + t * step, False)[0] < ll and t > 1e-6:     # backtracking line search
            t /= 2
        beta = beta + t * step
        ll_new, grad, H = pl(beta)
        if abs(ll_new - ll) < 1e-8:
            break
        ll = ll_new
    return beta, np.linalg.inv(H)


def fit_bonus(a, F, cond_start, rng):
    """Bonus is a ranking WITHIN the match: 3/2/1 to the top three by BPS. Plackett-Luce for the top three of
    the BPS order among the candidates the simulator sees (starters if conditional on starting), with scores
    linear in the match events. Also returns the training rates of BPS ties between consecutive positions and,
    if conditional on starting, of a substitute holding each position; the simulator applies both, so bonus
    totals per match match the data."""
    a = a[a.minutes > 0]
    Xb, fx = bonus_rows(a, F)
    bps = a.bps.to_numpy(float) + rng.random(len(a)) * 1e-3          # random tie-break for the order
    st = a.started.to_numpy() == 1
    o = np.lexsort((-bps, fx))
    fxo, bpo, sto = fx[o], np.floor(bps[o]), st[o]
    first = np.r_[True, fxo[1:] != fxo[:-1]]
    pos_in = np.arange(len(o)) - np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
    tie, sub = np.zeros(N_SLOT - 1), np.zeros(N_SLOT)
    for k in range(N_SLOT):
        at = np.flatnonzero(pos_in == k)
        sub[k] = 0.0 if not cond_start else (~sto[at]).mean()
        if k < N_SLOT - 1:
            nxt = at[(at + 1 < len(o))]
            nxt = nxt[fxo[nxt + 1] == fxo[nxt]]
            tie[k] = (bpo[nxt] == bpo[nxt + 1]).mean()
    keep = st if cond_start else np.ones(len(a), bool)
    beta, cov = fit_pl(Xb[keep], fx[keep], bps[keep])
    return {"beta": beta, "cov": cov, "tie": tie, "sub": sub}


def allocate_bonus(score, fix_codes, played, tie_u, sub_u, tie, sub):
    """3/2/1 by rank within each fixture for every draw. score (n, J, R) with Gumbel noise already added.
    BPS slots 1..N_SLOT: a slot is taken by a substitute with prob sub[k]; consecutive slots tie with prob
    tie[k]; bonus follows FPL's competition ranking (tie for 1st: 3, 3, 1; for 2nd: 3, 2, 2; ...)."""
    bonus = np.zeros(score.shape, dtype=np.int64)
    score = np.where(played, score, -np.inf)
    is_sub = sub_u < sub[:, None, None]                               # (N_SLOT, J, R)
    st_idx = np.cumsum(~is_sub, 0) - 1                                 # which starter fills each slot
    rank = np.ones(is_sub.shape, dtype=np.int64)
    for k in range(1, N_SLOT):
        rank[k] = np.where(tie_u[k - 1] < tie[k - 1], rank[k - 1], k + 1)
    pts = np.select([rank == 1, rank == 2, rank == 3], [3, 2, 1], 0)
    for f in np.unique(fix_codes):
        rows = np.flatnonzero(fix_codes == f)
        top = np.argsort(-score[rows], axis=0)[:N_SLOT]                 # (<=N_SLOT, J, R) starter order
        nt = top.shape[0]
        for k in range(N_SLOT):
            ok = (~is_sub[k]) & (st_idx[k] < nt) & (pts[k] > 0)
            if not ok.any():
                continue
            who = np.take_along_axis(top, np.clip(st_idx[k], 0, nt - 1)[None], 0)[0]   # (J, R)
            jj, rr = np.nonzero(ok)
            alive = np.isfinite(score[rows[who[jj, rr]], jj, rr])
            np.add.at(bonus, (rows[who[jj, rr]][alive], jj[alive], rr[alive]), pts[k][jj, rr][alive])
    return bonus


def team_sigma(tr):
    """Over-dispersion of team goals around lambda -> sd of log(true lambda / lambda_hat)."""
    t = tr.drop_duplicates(["season", "fixture", "was_home"])
    g, lam = t.team_goals.to_numpy(float), t.xg_for_pi.to_numpy(float)
    ok = ~np.isnan(g)
    extra = ((g[ok] - lam[ok]) ** 2 - lam[ok]).sum() / (lam[ok] ** 2).sum()
    return float(np.sqrt(max(extra, 0.0)))


# ------------------------------------------------------------------ defensive contribution (2025-26 rules on)
def dc_design(df, prior):
    pr = df.position.map(prior).fillna(1.0).to_numpy()   # goalkeepers: no DC points; any finite prior
    num = np.where(df.position == "DEF", df.L365_f_cbit, df.L365_f_cbirt)
    return pd.DataFrame({"const": 1.0, "pos_MID": (df.position == "MID").astype(float),
                         "pos_FWD": (df.position == "FWD").astype(float),
                         "log_dcrate": np.log((num + pr * 5) / (df.L365_f_min_dc / 90 + 5)),
                         "log_lamf": df.log_lamf, "log_lama": df.log_lama, "home": df.home}, index=df.index)


def fit_dc(tr):
    """Negative binomial for CBIT (DEF) / CBIRT (MID, FWD) per appearance, on training rows where the counts
    were recorded (2016-17..2018-19 within any training window that precedes 2025-26)."""
    a = tr[(tr.minutes > 0) & tr.position.isin(["DEF", "MID", "FWD"]) & tr.tackles.notna()
           & tr.clearances_blocks_interceptions.notna() & tr.recoveries.notna()]
    cbit = a.clearances_blocks_interceptions + a.tackles
    y = np.where(a.position == "DEF", cbit, cbit + a.recoveries).astype(float)
    prior = (pd.Series(y, index=a.index).groupby(a.position).sum() / a.groupby("position").e.sum()).to_dict()
    X = dc_design(a, prior)
    X["log_minutes"] = np.log(a.e)
    res = sm.NegativeBinomial(y, X, offset=np.log(a.e)).fit(disp=0, maxiter=300)
    return {"beta": res.params.to_numpy(), "cov": res.cov_params().to_numpy(), "prior": prior, "n": len(a),
            "names": list(X.columns) + ["alpha"]}


# ------------------------------------------------------------------ minutes ensemble
START_FEATS = ["st_n10", "st_full5", "st_mean5", "st_since"]


def fit_minutes(tr, rng, feats=None):
    feats = feats or MIN_FEATS
    members = []
    fx = (tr.season + "_" + tr.fixture.astype(str)).to_numpy()
    uniq = np.unique(fx)
    idx_by_fx = pd.Series(np.arange(len(tr))).groupby(fx).apply(np.array).to_dict()
    for b in range(B_MIN):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_fx[f] for f in pick])
        sub = tr.iloc[rows]
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=5, min_samples_leaf=100,
                                           l2_regularization=1.0, random_state=SEED + b)
        m.fit(sub[feats], sub.bucket)
        members.append(m)
    return members


def predict_minutes(members, te, feats=None):
    feats = feats or MIN_FEATS
    P = np.zeros((B_MIN, len(te), K))
    for b, m in enumerate(members):
        P[b][:, m.classes_] = m.predict_proba(te[feats])
    return P


# ------------------------------------------------------------------ simulation
def draw_beta(fit, rng, n):
    return rng.multivariate_normal(fit["beta"], fit["cov"], size=n, method="cholesky")   # (n, p)


def rate_matrix(fit, Xrow, betas, eb_noise, lam_noise):
    """eta (excluding log_minutes and offset) for each row and parameter draw: (n_rows, J)."""
    names = fit["names"]
    base = Xrow[[c for c in names if c != "log_minutes"]].to_numpy(float)          # (n, p-1)
    bidx = [i for i, c in enumerate(names) if c != "log_minutes"]
    eta = base @ betas[:, bidx].T                                                     # (n, J)
    # the GLMs were fitted on posterior means (EB fractions) and point lambdas, so their fitted rates already
    # average over that uncertainty; the draws are centred (- beta^2 sd^2 / 2) to keep the mean rate unchanged
    for c in names:
        j = names.index(c)
        b = betas[:, j][None, :]
        if c in eb_noise:
            eta += eb_noise[c][0] * b - 0.5 * (eb_noise[c][1] * b) ** 2
        if c in lam_noise:
            eta += lam_noise[c][0] * b - 0.5 * (lam_noise[c][1] * b) ** 2
    b_lm = betas[:, names.index("log_minutes")] if "log_minutes" in names else np.zeros(len(betas))
    return np.clip(eta, -20, 3), b_lm


PMF_LO, PMF_HI = -6, 40   # points range of the optional per-row distribution (values outside are clipped)


def simulate_fold(te, F, S, fits, bonus_fit, Pmin, bucket_vals, conv_ab, s_team, season, rng, dc=None,
                  slope=None, ehat=None, want_pmf=False):
    # whole fixtures per chunk (bonus is allocated within the match; team noise is shared within it)
    orig_index = te.index
    order = np.argsort(te.fixture.to_numpy(), kind="stable")
    te, F, S, Pmin = te.iloc[order], F.iloc[order], S.iloc[order], Pmin[:, order]
    ehat = {k: v[order] for k, v in (ehat or {}).items()}
    slope = {k: (v[0], v[1], v[2][order]) for k, v in (slope or {}).items()}
    if dc is not None:
        dc = {**dc, "X": dc["X"][order]}
    n = len(te)
    fixv = te.fixture.to_numpy()
    homev = te.was_home.to_numpy().astype(int)
    pos = te.position.to_numpy()
    out = {k: np.zeros(n) for k in ("xp", "var_total", "var_epi", "q10", "q25", "q50", "q75", "q90", "p_ge2", "p_ge5",
                                     "p_ge10", "cdf_lo", "cdf_hi")}
    # used ONLY to read the predictive CDF at the outcome (for PIT); forecast rows have no outcome
    y_act = te.total_points.to_numpy(float) if "total_points" in te else np.full(n, np.nan)
    comp = {k: np.zeros(n) for k in ("e_app", "e_goals", "e_assists", "e_cs", "e_gc_pts", "e_saves_pts", "e_bonus",
                                      "e_yellow", "e_red", "e_og", "e_pen_miss", "e_pen_saved", "e_minutes", "p_play",
                                      "p60", "e_dc", "p_g1", "p_g2")}
    pmf = np.zeros((n, PMF_HI - PMF_LO + 1), dtype=np.float32) if want_pmf else None
    if want_pmf:   # forecast mode: event probabilities for the website (counted from the same draws; evaluation output unchanged)
        comp.update({k: np.zeros(n) for k in ("p_a1", "p_b1", "p_b3", "p_sv1")})
    dc_on = dc is not None and season >= "2025-26"   # defensive-contribution points exist from 2025-26
    Xall = pd.concat([te, F], axis=1)
    Xall = Xall.loc[:, ~Xall.columns.duplicated()]
    Xall["const"] = 1.0
    CH = 400
    bounds, cur = [], 0
    for b in list(np.flatnonzero(fixv[1:] != fixv[:-1]) + 1) + [n]:
        if b - cur >= CH or b == n:
            bounds.append(slice(cur, b))
            cur = b
    for sl in bounds:
        X = Xall.iloc[sl]
        nn = len(X)
        # ---- epistemic draws shared across components within a parameter draw
        eb_noise = {c: (S[c].to_numpy()[sl][:, None] * rng.standard_normal((nn, J)), S[c].to_numpy()[sl][:, None])
                    for c in EB_COLS}
        # team log-lambda noise: one draw per team and match; a team's attack draw is its opponent's defence draw
        tk, ok_ = fixv[sl] * 2 + homev[sl], fixv[sl] * 2 + 1 - homev[sl]
        keys = np.unique(np.r_[tk, ok_])
        Z = s_team * rng.standard_normal((len(keys), J))
        lam_noise = {"log_lamf": (Z[np.searchsorted(keys, tk)], s_team),
                     "log_lama": (Z[np.searchsorted(keys, ok_)], s_team)}
        betas = {k: draw_beta(f, rng, J) for k, f in fits.items()}
        etas = {k: rate_matrix(fits[k], X, betas[k], eb_noise, lam_noise) for k in fits}
        conv = rng.beta(conv_ab[0], conv_ab[1], size=J)
        gam = rng.multivariate_normal(bonus_fit["beta"], bonus_fit["cov"], size=J, method="cholesky")   # (J, p)
        member = np.arange(J) % B_MIN
        sl_draw = {k: rng.multivariate_normal(v[0], v[1], size=J) for k, v in slope.items()}
        # ---- minutes: (nn, J, R)
        P = Pmin[member][:, sl, :].transpose(1, 0, 2)                                    # (nn, J, K)
        cum = P.cumsum(-1)
        u = rng.random((nn, J, R))
        bkt = (u[..., None] > cum[:, :, None, :]).sum(-1).clip(0, K - 1)
        mins = np.zeros_like(u)
        for k in range(1, K):
            mk = bkt == k
            mins[mk] = rng.choice(bucket_vals[k], size=mk.sum())
        e = mins / 90.0
        played = mins > 0
        loge = np.log(np.where(played, e, 1.0))

        is_gk_ = (pos[sl] == "GKP")[:, None, None]

        def lam(key):
            eta, blm = etas[key]
            v = np.where(played, np.exp(eta[:, :, None] + (1 + blm)[None, :, None] * loge), 0.0)
            if "gk_rate" in fits[key]:
                v = np.where(is_gk_, fits[key]["gk_rate"] * e, v)
            if key in sl_draw:   # walk-forward calibration: rate x exp(f(log index)) / index
                f_ = slope[key][2][sl] @ sl_draw[key].T                                     # (nn, J)
                li = np.log(np.clip(ehat[key][sl], 1e-6, None))[:, None]
                v = v * np.exp(f_ - li)[:, :, None]
            return np.minimum(v, 20.0)

        g_open = rng.poisson(lam("goal_open"))
        pen_att = rng.poisson(lam("pen_att"))
        pen_g = rng.binomial(pen_att, conv[None, :, None])
        pen_miss = pen_att - pen_g
        goals = g_open + pen_g
        assists = rng.poisson(lam("assist"))
        # goals conceded: ONE full-match draw per team, shared by team-mates (so clean sheets are shared, as in
        # reality), thinned binomially to each player's minutes (mean = the fitted on-pitch rate)
        full = np.minimum(np.exp(etas["conceded"][0]), 20.0)                             # (nn, J) 90-minute rate
        _, first, inv = np.unique(tk, return_index=True, return_inverse=True)
        C = rng.poisson(np.broadcast_to(full[first][:, :, None], (len(first), J, R)))[inv]
        conc = rng.binomial(C, np.clip(lam("conceded") / full[:, :, None], 0, 1))
        is_gk = (pos[sl] == "GKP")[:, None, None]
        saves = np.where(is_gk, rng.poisson(lam("saves")), 0)
        pensv = np.where(is_gk, rng.poisson(lam("pen_saved")), 0)
        yel = (rng.poisson(lam("yellow")) > 0).astype(int)
        red = (rng.poisson(lam("red")) > 0).astype(int)
        og = rng.poisson(lam("own_goal"))
        m60 = (mins >= 60).astype(int)
        cs = (m60 * (conc == 0)).astype(int)
        # ---- bonus: ranked within the match, conditional on the simulated events
        pdf = pos[sl]
        Xb = bonus_design(goals, assists, cs, m60, saves // 3, conc, pensv, yel,
                          np.broadcast_to((pdf == "DEF")[:, None, None], goals.shape).astype(float),
                          np.broadcast_to((pdf == "MID")[:, None, None], goals.shape).astype(float),
                          np.broadcast_to((pdf == "FWD")[:, None, None], goals.shape).astype(float),
                          np.broadcast_to(F["log_bps90"].to_numpy()[sl][:, None, None], goals.shape))
        score = np.einsum("njrp,jp->njr", Xb, gam) + rng.gumbel(size=goals.shape)
        bonus = allocate_bonus(score, fixv[sl], played, rng.random((N_SLOT - 1, J, R)), rng.random((N_SLOT, J, R)),
                               bonus_fit["tie"], bonus_fit["sub"])
        # ---- points under the season's rules
        gpts = np.vectorize(RULES["goal"].get)(pdf)[:, None, None]
        if season >= "2026-27":   # rule change published by the API: goalkeeper goal = 10
            gpts = np.where((pdf == "GKP")[:, None, None], 10, gpts)
        cspts = np.vectorize(RULES["cs"].get)(pdf)[:, None, None]
        gcp = np.vectorize(RULES["gc_per2"].get)(pdf)[:, None, None]
        app = played.astype(int) + m60
        dc_hit = np.zeros(goals.shape, dtype=int)
        if dc_on:
            g_ = rng.multivariate_normal(dc["fit"]["beta"], dc["fit"]["cov"], size=J, method="cholesky")
            eta = np.nan_to_num(dc["X"][sl] @ g_[:, :-2].T, nan=-20.0)          # (nn, J)
            nb_n = 1.0 / np.clip(g_[:, -1], 1e-3, None)
            mean = np.where(played, np.exp(np.clip(eta[:, :, None] + (1 + g_[:, -2])[None, :, None] * loge, -20, 5)), 0.0)
            nn_ = np.broadcast_to(nb_n[None, :, None], mean.shape)
            D = rng.negative_binomial(nn_, nn_ / (nn_ + mean))
            thr = np.where(pdf == "DEF", 10, 12)[:, None, None]
            dc_hit = ((D >= thr) & played & (pdf != "GKP")[:, None, None]).astype(int)
        pts = (app + gpts * goals + 3 * assists + cspts * cs + gcp * (conc // 2) + saves // 3 + 5 * pensv
               - 2 * pen_miss - yel - 3 * red - 2 * og + bonus + 2 * dc_hit)
        flat = pts.reshape(nn, -1)
        if want_pmf:   # counting only: the random stream is unchanged
            b = np.clip(flat, PMF_LO, PMF_HI) - PMF_LO
            cnt = np.zeros((nn, PMF_HI - PMF_LO + 1))
            np.add.at(cnt, (np.repeat(np.arange(nn), flat.shape[1]), b.ravel()), 1)
            pmf[sl] = cnt / flat.shape[1]
        out["xp"][sl] = flat.mean(1)
        out["var_total"][sl] = flat.var(1)
        mj = pts.mean(2)
        out["var_epi"][sl] = np.maximum(mj.var(1) - pts.var(2).mean(1) / R, 0)
        for q in (10, 25, 50, 75, 90):
            out[f"q{q}"][sl] = np.percentile(flat, q, axis=1)
        for t in (2, 5, 10):
            out[f"p_ge{t}"][sl] = (flat >= t).mean(1)
        ya = y_act[sl][:, None]
        out["cdf_lo"][sl] = (flat < ya).mean(1)
        out["cdf_hi"][sl] = (flat <= ya).mean(1)
        for key, arr in (("e_app", app), ("e_goals", goals), ("e_assists", assists), ("e_cs", cs),
                         ("e_gc_pts", gcp * (conc // 2)), ("e_saves_pts", saves // 3), ("e_bonus", bonus),
                         ("e_yellow", yel), ("e_red", red), ("e_og", og), ("e_pen_miss", pen_miss),
                         ("e_pen_saved", pensv), ("e_minutes", mins), ("p_play", played), ("p60", m60),
                         ("e_dc", dc_hit), ("p_g1", goals >= 1), ("p_g2", goals >= 2)):
            comp[key][sl] = arr.reshape(nn, -1).mean(1)
        if want_pmf:
            for key, arr in (("p_a1", assists >= 1), ("p_b1", bonus >= 1), ("p_b3", bonus >= 3), ("p_sv1", saves >= 3)):
                comp[key][sl] = arr.reshape(nn, -1).mean(1)
    res = pd.DataFrame({**out, **comp}, index=te.index)
    if want_pmf:
        res["pmf"] = list(pmf)
    return res.loc[orig_index]


# ------------------------------------------------------------------ fit / predict
# A season's model is FITTED ONCE (every component is trained on earlier seasons only) and can then predict any
# rows with the feature schema: the finished fixtures of the season (evaluation) or upcoming ones (forecast).
MODELS = ROOT / "data" / "models"


def frame_hash(df):
    """Content hash of a data frame (values and column names, not the index). `row_id` (a running number that
    shifts when rows are added anywhere) is left out."""
    import hashlib
    df = df.drop(columns=["row_id"], errors="ignore")
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    h.update("|".join(map(str, df.columns)).encode())
    return h.hexdigest()[:16]


def fit_season(tr, ts, cond_start, H, pool=None, cal_kind="spline", fit_minutes_model=True):
    """Fit every component for test season `ts` on the training rows `tr` (seasons before ts).
    pool: earlier out-of-sample predictions for the calibration of the goal / assist index (cal_kind 'spline',
    'linear' or None for no calibration). Returns a picklable dict; predict_rows applies it."""
    # common random numbers: every variant uses the same seed for the same fold
    rng = np.random.default_rng(SEED + 7 * int(ts[:4]) + (10 if cond_start else 0))
    assert tr.season.max() < ts
    pri = priors(tr)
    kmap = estimate_k_full(H, int(ts[:4]) - 1)
    Ftr = pd.concat([fixed_features(tr, pri), eb_features(tr, kmap)[0]], axis=1)
    trF = pd.concat([tr, Ftr], axis=1)
    app = trF[trF.minutes > 0]
    fits, coefs = {}, []
    for name, spec in GLM_SPECS.items():
        rows = app if spec["rows"] == "app" else app[app.position == "GKP"]
        rows = rows[rows[spec["y"]].notna()]
        fits[name] = fit_glm(rows, spec)
        terms = fits[name]["names"]
        coefs.append(pd.DataFrame({"fold": ts, "component": name, "term": terms,
                                   "beta": fits[name]["beta"], "se": np.sqrt(np.diag(fits[name]["cov"]))}))
    bonus_fit = fit_bonus(app, Ftr, cond_start, np.random.default_rng(SEED + int(ts[:4])))
    coefs.append(pd.DataFrame({"fold": ts, "component": "bonus_rank", "term": BONUS_X, "beta": bonus_fit["beta"],
                               "se": np.sqrt(np.diag(bonus_fit["cov"]))}))
    pa, pg = app.tgt_pen_att.sum(), app.tgt_pen_goals.sum()
    if cond_start:   # minutes GIVEN a start: trained on starts only, zero minutes impossible
        utr = tr[tr.started == 1]
        mfeats = MIN_FEATS + START_FEATS
    else:
        utr = tr[tr.universe]
        mfeats = MIN_FEATS
    members = fit_minutes(utr, rng, mfeats) if fit_minutes_model else None
    bucket_vals = {k: utr.minutes[utr.bucket == k].to_numpy(float) for k in range(1, K)}
    dcfit = None
    if ts >= "2025-26":
        dcfit = fit_dc(tr)
        coefs.append(pd.DataFrame({"fold": ts, "component": "defensive_contribution", "term": dcfit["names"],
                                   "beta": dcfit["beta"], "se": np.sqrt(np.diag(dcfit["cov"]))}))
    cal = None
    if cal_kind and pool is not None and len(pool):
        pool = pool[pool.season < ts]
        cal = {"goal_open": fit_calibration(pool, "ehat_goal_open", "tgt_np_goals", cal_kind),
               "assist": fit_calibration(pool, "ehat_assist", "assists", cal_kind)}
        cal["n_pool"] = len(pool)
    return {"season": ts, "cond_start": cond_start, "cal_kind": cal_kind, "pri": pri, "kmap": kmap, "fits": fits,
            "bonus_fit": bonus_fit, "conv_ab": (pg + 1, pa - pg + 1), "conv": pg / pa, "s_team": team_sigma(tr),
            "members": members, "mfeats": mfeats, "bucket_vals": bucket_vals, "dc": dcfit, "cal": cal,
            "coefs": pd.concat(coefs), "n_train": len(tr), "train_hash": frame_hash(tr)}


def predict_rows(M, te, rng, Pmin=None, want_pmf=False):
    """Simulate xP for rows `te` (master-table schema; outcomes optional) with the fitted season model M.
    Pmin: minutes-bucket probabilities (B_MIN, n, K) if already computed. Returns (results, features, Pmin)."""
    Fte_eb, Ste = eb_features(te, M["kmap"])
    Fte = pd.concat([fixed_features(te, M["pri"]), Fte_eb], axis=1)
    if Pmin is None:
        Pmin = predict_minutes(M["members"], te, M["mfeats"])
    dc = None if M["dc"] is None else {"fit": M["dc"], "X": dc_design(te, M["dc"]["prior"]).to_numpy()}
    Xte = pd.concat([te, Fte], axis=1)
    Xte = Xte.loc[:, ~Xte.columns.duplicated()].assign(const=1.0)
    bv = M["bucket_vals"]
    mbar = np.array([0.0] + [bv[k].mean() for k in range(1, K)])
    Pm = Pmin.mean(0)
    ehat = {"goal_open": expected_hat(M["fits"]["goal_open"], Xte, Pm, mbar),
            "assist": expected_hat(M["fits"]["assist"], Xte, Pm, mbar)}
    slope = None
    if M["cal"] is not None:
        slope = {k: (M["cal"][k]["params"], M["cal"][k]["cov"], calibration_basis(M["cal"][k], ehat[k]))
                 for k in ("goal_open", "assist")}
        fx_ = {k: np.exp(v[2] @ v[0]).mean() / ehat[k].mean() for k, v in slope.items()}
        print(f"  {M['season']}: {M['cal_kind']} calibration from {M['cal']['n_pool']:,} earlier out-of-sample rows; "
              f"mean factor goals {fx_['goal_open']:.3f}, assists {fx_['assist']:.3f}", flush=True)
    res = simulate_fold(te, Fte, Ste, M["fits"], M["bonus_fit"], Pmin, bv, M["conv_ab"], M["s_team"], M["season"],
                        rng, dc=dc, slope=slope, ehat=ehat, want_pmf=want_pmf)
    res["ehat_goal_open"], res["ehat_assist"] = ehat["goal_open"], ehat["assist"]
    res["minutes_ens_sd_pplay"] = (1 - Pmin[:, :, 0]).std(0)
    return res, Fte, Pmin


def model_path(tag, ts):
    return MODELS / f"xp_model_{tag}_{ts}.pkl"


def save_model(M, tag):
    import pickle
    import subprocess
    MODELS.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    M = {**M, "tag": tag, "code_commit": git.stdout.strip(),
         "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    with open(model_path(tag, M["season"]), "wb") as f:
        pickle.dump(M, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(tag, ts):
    import pickle
    with open(model_path(tag, ts), "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ main
def main():
    import sys
    sealed = "--sealed" in sys.argv
    live = "--live" in sys.argv
    cond_start = "--unconditional" not in sys.argv   # default: xP GIVEN THE PLAYER STARTS
    mode = "sealed" if sealed else "live" if live else "dev"
    folds = ["2025-26"] if sealed else ["2026-27"] if live else TEST
    # calibration of the goal / assist index: spline (default), --slope (linear) or --no-cal (none)
    use_slope = "--no-cal" not in sys.argv
    curve = use_slope and "--slope" not in sys.argv
    cal = "" if curve else "_slope" if use_slope else "_nocal"
    cal_kind = None if not use_slope else "spline" if curve else "linear"
    tag = mode + ("" if cond_start else "_uncond") + cal
    variant = ("" if cond_start else "_uncond") + cal   # calibration pool = earlier predictions of the SAME target
    d = load("2025-26" if sealed else "2026-27" if live else "2024-25")
    print(f"RUN {tag}: folds {folds}; conditional on starting: {cond_start}", flush=True)
    print(f"rows {len(d):,}; universe share {d.universe.mean():.3f}", flush=True)
    a_chk = d[d.minutes >= 60]
    print("clean_sheets == (60+ and 0 conceded):",
          f"{((a_chk.clean_sheets == 1) == (a_chk.goals_conceded == 0)).mean():.4f}", flush=True)
    H = EB.split_half_table(EB.player_seasons())
    results, coefs = [], []
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    for ts in folds:
        if cond_start:
            tr, te = d[d.season < ts], d[(d.season == ts) & (d.started == 1)]
        else:
            tr, te = d[d.season < ts], d[(d.season == ts) & d.universe]
        # calibration pool: this run's earlier folds + earlier OOS predictions of the same variant
        prior = [r_ for r_ in results]
        if mode != "dev":
            for f_ in ([f"dev{variant}"] + ([f"sealed{variant}"] if live else [])):
                fp = OUT / f"xp_predictions_{f_}.parquet"
                if fp.exists():
                    prior.append(pd.read_parquet(fp))
        pool = pd.concat(prior) if prior else None
        cfile = cache / f"pmin_{'start' if cond_start else 'uncond'}_{ts}.npy"
        reuse = "--reuse-minutes" in sys.argv and cfile.exists()
        M = fit_season(tr, ts, cond_start, H, pool, cal_kind, fit_minutes_model=not reuse)
        coefs.append(M["coefs"])
        Pmin = None
        if reuse:
            Pmin = np.load(cfile)
            assert Pmin.shape[1] == len(te), "minutes cache does not match the test rows"
        else:
            save_model(M, tag)
        rng_sim = np.random.default_rng(SEED + 7 * int(ts[:4]) + 1000)   # same simulation stream for every variant
        res, Fte, Pmin = predict_rows(M, te, rng_sim, Pmin)
        np.save(cfile, Pmin)
        results.append(pd.concat([te, Fte.add_prefix("feat_")], axis=1).join(res))
        print(f"  fold {ts}: s_team={M['s_team']:.3f} conv={M['conv']:.3f} rows={len(te):,} "
              f"mean xP={res.xp.mean():.3f} actual={te.total_points.mean():.3f}", flush=True)
    R_ = pd.concat(results)
    R_.to_parquet(OUT / f"xp_predictions_{tag}.parquet", index=False)
    pd.concat(coefs).to_csv(OUT / f"xp_component_coefficients_{tag}.csv", index=False)
    print("saved", len(R_))


if __name__ == "__main__":
    main()
