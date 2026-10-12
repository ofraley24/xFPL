#!/usr/bin/env python3
"""Full xP model: component models + nested Monte-Carlo assembly with explicit uncertainty.

Walk-forward: test seasons 2019-20 .. 2024-25, each fold trained on all earlier seasons (>= 2016-17).
Sealed 2025-26 and live 2026-27 are not read. Evaluation universe: players with minutes in their
previous <= 10 fixtures (97% of real appearances); every other row is predicted as 0 points.

COMPONENTS (statsmodels GLMs unless stated; Poisson with offset log(minutes/90); the coefficient on
log(minutes/90) lets rates scale non-linearly with minutes)
  minutes      8-bucket distribution, bagged gradient boosting (B fixture-bootstrap members)
  goals (open) Understat non-penalty goals; team lambda, shot-share and shot-quality FRACTIONS (split-half
               empirical Bayes), opponent lambda, home, price, ownership
  penalties    attempts (taker fraction) x conversion (Beta posterior); misses = attempts - scored
  assists      team lambda, xA fraction (split-half EB), opponent lambda, home, price, ownership
  conceded     goals conceded while on the pitch; clean sheet = 60+ min and 0 conceded
  saves, penalty saves (GKP); yellow, red, own goals (all)
  bonus        multinomial logit for 0/1/2/3 CONDITIONAL on the match events (goals, assists, clean
               sheet, saves, conceded ...) + the player's lagged BPS rate; sampled AFTER the events in
               each simulation, so bonus correlates with goals/clean sheets as it does in reality

UNCERTAINTY
  epistemic: J parameter draws. Each draw takes (i) GLM coefficients ~ MVN(beta_hat, Cov), (ii) the
             player's talent fractions from their EB posterior (sd = sqrt(sigma2/(T+k))), (iii) team
             log-lambda + N(0, s_team^2), s_team estimated from over-dispersion of team goals around
             lambda in training, (iv) a minutes-ensemble member, (v) penalty conversion ~ Beta, (vi)
             bonus-model coefficients ~ MVN
  aleatoric: R outcome draws per parameter draw (minutes, events, bonus)
  Per row: xP = mean of J*R points; var_epi = Var_j(mean_r) - E_j(Var_r)/R; var_total = Var(all).
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
    return d


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
    "yellow": dict(rows="app", y="yellow_cards", x=["pos_DEF", "pos_MID", "pos_FWD", "log_yel90", "log_lama", "home"]),
    # a red card SHORTENS minutes (reverse causality: estimate -1.2), so no minutes elasticity; no position
    # dummies (rare event, separation)
    "red": dict(rows="app", y="red_cards", x=["log_red90", "log_yel90"], logmin=False),
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
    res = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(df.e)).fit()
    out = {"names": list(X.columns), "beta": res.params.to_numpy(), "cov": res.cov_params().to_numpy()}
    if spec.get("exclude_gk"):
        out["gk_rate"] = (gk[spec["y"]].sum() + 0.5) / (gk.e.sum() + 50.0)   # per 90, lightly smoothed
    return out


BONUS_X = ["g1", "g2p", "a1", "a2p", "cs_def", "cs_mid", "m60", "sv3", "conc2p", "pensv", "yel", "pos_DEF", "pos_MID",
           "pos_FWD", "log_bps90"]


def bonus_design(goals, assists, cs, m60, sv3, conc, pensv, yel, pos_def, pos_mid, pos_fwd, log_bps90):
    """Event features for the conditional bonus model (vectorised; works on simulation arrays too)."""
    gk_or_def = 1 - pos_mid - pos_fwd
    return np.stack([np.minimum(goals, 1), np.maximum(goals - 1, 0), np.minimum(assists, 1), np.maximum(assists - 1, 0),
                     cs * gk_or_def, cs * pos_mid, m60, sv3, np.maximum(conc - 1, 0) * gk_or_def, pensv, yel,
                     pos_def, pos_mid, pos_fwd, log_bps90], axis=-1)


def fit_bonus(a, F):
    pdf = a.position
    Xb = bonus_design(a.goals_scored.to_numpy(float), a.assists.to_numpy(float), a.clean_sheets.to_numpy(float),
                      (a.minutes >= 60).to_numpy(float), (a.saves // 3).to_numpy(float), a.goals_conceded.to_numpy(float),
                      a.penalties_saved.to_numpy(float), a.yellow_cards.to_numpy(float), (pdf == "DEF").to_numpy(float),
                      (pdf == "MID").to_numpy(float), (pdf == "FWD").to_numpy(float), F.loc[a.index, "log_bps90"].to_numpy())
    X = sm.add_constant(Xb, has_constant="add")
    y = a.bonus.clip(0, 3).astype(int).to_numpy()
    res = sm.MNLogit(y, X).fit(method="newton", maxiter=100, disp=0)
    # params: (p, 3) for classes 1..3 vs 0; cov_params is ordered as params.ravel(order="F")
    return {"beta": np.asarray(res.params), "cov": np.asarray(res.cov_params())}


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
    for c in names:
        j = names.index(c)
        if c in eb_noise:
            eta += eb_noise[c] * betas[:, j][None, :]
        if c in lam_noise:
            eta += lam_noise[c] * betas[:, j][None, :]
    b_lm = betas[:, names.index("log_minutes")] if "log_minutes" in names else np.zeros(len(betas))
    return np.clip(eta, -20, 3), b_lm


def simulate_fold(te, F, S, fits, bonus_fit, Pmin, bucket_vals, conv_ab, s_team, season, rng, dc=None):
    n = len(te)
    pos = te.position.to_numpy()
    out = {k: np.zeros(n) for k in ("xp", "var_total", "var_epi", "q10", "q25", "q50", "q75", "q90", "p_ge2", "p_ge5",
                                     "p_ge10", "cdf_lo", "cdf_hi")}
    y_act = te.total_points.to_numpy(float)   # used ONLY to read the predictive CDF at the outcome (for PIT)
    comp = {k: np.zeros(n) for k in ("e_app", "e_goals", "e_assists", "e_cs", "e_gc_pts", "e_saves_pts", "e_bonus",
                                      "e_yellow", "e_red", "e_og", "e_pen_miss", "e_pen_saved", "e_minutes", "p_play",
                                      "p60", "e_dc")}
    dc_on = dc is not None and season >= "2025-26"   # defensive-contribution points exist from 2025-26
    Xall = pd.concat([te, F], axis=1)
    Xall = Xall.loc[:, ~Xall.columns.duplicated()]
    Xall["const"] = 1.0
    CH = 400
    for s in range(0, n, CH):
        sl = slice(s, min(s + CH, n))
        X = Xall.iloc[sl]
        nn = len(X)
        # ---- epistemic draws shared across components within a parameter draw
        eb_noise = {c: S[c].to_numpy()[sl][:, None] * rng.standard_normal((nn, J)) for c in EB_COLS}
        lam_noise = {c: s_team * rng.standard_normal((nn, J)) for c in LAMBDA_COLS}
        betas = {k: draw_beta(f, rng, J) for k, f in fits.items()}
        etas = {k: rate_matrix(fits[k], X, betas[k], eb_noise, lam_noise) for k in fits}
        conv = rng.beta(conv_ab[0], conv_ab[1], size=J)
        bb = rng.multivariate_normal(bonus_fit["beta"].ravel(order="F"), bonus_fit["cov"],
                                     size=J, method="cholesky").reshape(J, -1, 3, order="F")   # (J, p, 3)
        member = np.arange(J) % B_MIN
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
            return np.minimum(v, 20.0)

        g_open = rng.poisson(lam("goal_open"))
        pen_att = rng.poisson(lam("pen_att"))
        pen_g = rng.binomial(pen_att, conv[None, :, None])
        goals = g_open + pen_g
        assists = rng.poisson(lam("assist"))
        conc = rng.poisson(lam("conceded"))
        is_gk = (pos[sl] == "GKP")[:, None, None]
        saves = np.where(is_gk, rng.poisson(lam("saves")), 0)
        pensv = np.where(is_gk, rng.poisson(lam("pen_saved")), 0)
        yel = (rng.poisson(lam("yellow")) > 0).astype(int)
        red = (rng.poisson(lam("red")) > 0).astype(int)
        og = rng.poisson(lam("own_goal"))
        m60 = (mins >= 60).astype(int)
        cs = (m60 * (conc == 0)).astype(int)
        # ---- bonus, conditional on the simulated events
        pdf = pos[sl]
        Xb = bonus_design(goals, assists, cs, m60, saves // 3, conc, pensv, yel,
                          np.broadcast_to((pdf == "DEF")[:, None, None], goals.shape).astype(float),
                          np.broadcast_to((pdf == "MID")[:, None, None], goals.shape).astype(float),
                          np.broadcast_to((pdf == "FWD")[:, None, None], goals.shape).astype(float),
                          np.broadcast_to(F["log_bps90"].to_numpy()[sl][:, None, None], goals.shape))
        Xb = np.concatenate([np.ones(goals.shape + (1,)), Xb], -1)                       # (nn,J,R,p)
        logits = np.einsum("njrp,jpc->njrc", Xb, bb)
        logits = np.concatenate([np.zeros(goals.shape + (1,)), logits], -1)
        pb = np.exp(logits - logits.max(-1, keepdims=True))
        pb /= pb.sum(-1, keepdims=True)
        bonus = (rng.random(goals.shape)[..., None] > pb.cumsum(-1)).sum(-1).clip(0, 3)
        bonus = np.where(played, bonus, 0)
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
               - 2 * (pen_att - pen_g) - yel - 3 * red - 2 * og + bonus + 2 * dc_hit)
        flat = pts.reshape(nn, -1)
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
                         ("e_yellow", yel), ("e_red", red), ("e_og", og), ("e_pen_miss", pen_att - pen_g),
                         ("e_pen_saved", pensv), ("e_minutes", mins), ("p_play", played), ("p60", m60),
                         ("e_dc", dc_hit)):
            comp[key][sl] = arr.reshape(nn, -1).mean(1)
    return pd.DataFrame({**out, **comp}, index=te.index)


# ------------------------------------------------------------------ main
def main():
    import sys
    sealed = "--sealed" in sys.argv
    live = "--live" in sys.argv
    cond_start = "--unconditional" not in sys.argv   # default: xP GIVEN THE PLAYER STARTS
    mode = "sealed" if sealed else "live" if live else "dev"
    folds = ["2025-26"] if sealed else ["2026-27"] if live else TEST
    tag = mode + ("" if cond_start else "_uncond")
    rng = np.random.default_rng(SEED + (1 if sealed else 2 if live else 0) + (10 if cond_start else 0))
    d = load("2025-26" if sealed else "2026-27" if live else "2024-25")
    print(f"RUN {tag}: folds {folds}; conditional on starting: {cond_start}", flush=True)
    print(f"rows {len(d):,}; universe share {d.universe.mean():.3f}", flush=True)
    a_chk = d[d.minutes >= 60]
    print("clean_sheets == (60+ and 0 conceded):",
          f"{((a_chk.clean_sheets == 1) == (a_chk.goals_conceded == 0)).mean():.4f}", flush=True)
    H = EB.split_half_table(EB.player_seasons())
    results, coefs = [], []
    for ts in folds:
        if cond_start:
            tr, te = d[d.season < ts], d[(d.season == ts) & (d.started == 1)]
        else:
            tr, te = d[d.season < ts], d[(d.season == ts) & d.universe]
        assert tr.season.max() < ts
        pri = priors(tr)
        kmap = estimate_k_full(H, int(ts[:4]) - 1)
        Ftr = pd.concat([fixed_features(tr, pri), eb_features(tr, kmap)[0]], axis=1)
        Fte_fixed = fixed_features(te, pri)
        Fte_eb, Ste = eb_features(te, kmap)
        Fte = pd.concat([Fte_fixed, Fte_eb], axis=1)
        trF = pd.concat([tr, Ftr], axis=1)
        app = trF[trF.minutes > 0]
        fits = {}
        for name, spec in GLM_SPECS.items():
            rows = app if spec["rows"] == "app" else app[app.position == "GKP"]
            rows = rows[rows[spec["y"]].notna()]
            fits[name] = fit_glm(rows, spec)
            coefs.append(pd.DataFrame({"fold": ts, "component": name, "term": fits[name]["names"],
                                       "beta": fits[name]["beta"], "se": np.sqrt(np.diag(fits[name]["cov"]))}))
        bonus_fit = fit_bonus(app, Ftr)
        pa, pg = app.tgt_pen_att.sum(), app.tgt_pen_goals.sum()
        conv_ab = (pg + 1, pa - pg + 1)
        s_team = team_sigma(tr)
        if cond_start:   # minutes GIVEN a start: trained on starts only, zero minutes impossible
            utr = tr[tr.started == 1]
            mfeats = MIN_FEATS + START_FEATS
        else:
            utr = tr[tr.universe]
            mfeats = MIN_FEATS
        members = fit_minutes(utr, rng, mfeats)
        Pmin = predict_minutes(members, te, mfeats)
        bucket_vals = {k: utr.minutes[utr.bucket == k].to_numpy(float) for k in range(1, K)}
        dc = None
        if ts >= "2025-26":
            dcfit = fit_dc(tr)
            dc = {"fit": dcfit, "X": dc_design(te, dcfit["prior"]).to_numpy()}
            coefs.append(pd.DataFrame({"fold": ts, "component": "defensive_contribution", "term": dcfit["names"],
                                       "beta": dcfit["beta"], "se": np.sqrt(np.diag(dcfit["cov"]))}))
        res = simulate_fold(te, Fte, Ste, fits, bonus_fit, Pmin, bucket_vals, conv_ab, s_team, ts, rng, dc=dc)
        res["minutes_ens_sd_pplay"] = (1 - Pmin[:, :, 0]).std(0)
        results.append(pd.concat([te, Fte.add_prefix("feat_")], axis=1).join(res))
        print(f"  fold {ts}: s_team={s_team:.3f} conv={pg/pa:.3f} rows={len(te):,} "
              f"mean xP={res.xp.mean():.3f} actual={te.total_points.mean():.3f}", flush=True)
    R_ = pd.concat(results)
    R_.to_parquet(OUT / f"xp_predictions_{tag}.parquet", index=False)
    pd.concat(coefs).to_csv(OUT / f"xp_component_coefficients_{tag}.csv", index=False)
    print("saved", len(R_))


if __name__ == "__main__":
    main()
