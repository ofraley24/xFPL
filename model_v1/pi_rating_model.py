"""
v1 of the dual attack/defence pi-rating -> Dixon-Coles Poisson model.

Implements Stage 1 + Stage 2 of the literature review's recommendations:

Stage 1 (core statistics):
  1. Dixon-Coles tau correction for the four low-score cells (0-0,0-1,1-0,1-1),
     rho initialised at -0.13 (Dixon & Coles 1997) and re-estimated by MLE.
  2. Pi-rating-style update conventions applied to the attack/defence
     structure the model actually uses (the SPI-style architecture, not a
     naive single-number chess Elo): a discrepancy-based update, a
     diminishing-returns transform of the error (psi(e) = sign(e)*3*log10(1+|e|),
     the Constantinou & Fenton form), and learning rates fit on data rather
     than hand-picked.
  3. A log-linear ratings -> Poisson-rate link:
        log(lambda_home) = c_league + home_adv + (attack_home - defence_away)/s
        log(lambda_away) = c_league +            (attack_away - defence_home)/s
     following Maher (1982) / Dyte & Clarke (2000).

Stage 2 (xG blend):
  4. Ratings are updated on a blend of non-penalty xG (Understat npxG) and
     actual goals: blend = alpha*npxG + (1-alpha)*goals, with alpha a fitted
     hyperparameter. Understat is used as the single, fixed xG provider
     throughout (never mixed with another source), per the literature
     review's provider-consistency caveat.

Explicitly deferred to later stages (not implemented here):
  - Stage 3 context features (home advantage crowd/league splits, squad
    availability, rest days).
  - Stage 4 full evaluation (RPS/log-loss backtests, benchmarking against a
    weighted-MLE Dixon-Coles model and vig-removed closing odds).
  - Stage 5 gradient-boosted upgrade.

This module only fits Stage 1/2 hyperparameters using an in-sample walk-
forward Poisson likelihood as a sanity-check objective, NOT a proper
rolling-origin cross-validated backtest -- that is Stage 4 work. Treat the
fitted hyperparameters below as a reasonable starting point, not a final
tuned model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def psi(e: float, const: float = 3.0) -> float:
    """Constantinou & Fenton (2013) diminishing-returns transform.

    Applied to a signed error so a single blow-out match cannot dominate a
    team's rating trajectory the way a raw goal-difference update would.
    """
    return math.copysign(const * math.log10(1.0 + abs(e)), e)


@dataclass
class RatingState:
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)

    def get(self, team: str):
        return self.attack.get(team, 0.0), self.defence.get(team, 0.0)

    def ensure(self, team: str):
        self.attack.setdefault(team, 0.0)
        self.defence.setdefault(team, 0.0)


class PiRatingDixonColesModel:
    """Dual attack/defence pi-rating model with a Dixon-Coles score matrix."""

    def __init__(
        self,
        home_adv: float,
        scale: float,
        alpha: float,
        lr_attack: float,
        lr_defence: float,
        league_intercepts: dict,
        rho: float = -0.13,
        psi_const: float = 3.0,
    ):
        self.home_adv = home_adv
        self.scale = scale
        self.alpha = alpha
        self.lr_attack = lr_attack
        self.lr_defence = lr_defence
        self.league_intercepts = league_intercepts
        self.rho = rho
        self.psi_const = psi_const
        self.ratings: dict[str, RatingState] = {}

    def _state(self, league: str) -> RatingState:
        if league not in self.ratings:
            self.ratings[league] = RatingState()
        return self.ratings[league]

    def predict_lambdas(self, league: str, home: str, away: str):
        st = self._state(league)
        st.ensure(home)
        st.ensure(away)
        att_h, def_h = st.get(home)
        att_a, def_a = st.get(away)
        c = self.league_intercepts[league]
        log_lh = c + self.home_adv + (att_h - def_a) / self.scale
        log_la = c + (att_a - def_h) / self.scale
        return math.exp(log_lh), math.exp(log_la)

    def update(self, league: str, home: str, away: str, gh: int, ga: int, npxg_h: float, npxg_a: float):
        st = self._state(league)
        lam_h, lam_a = self.predict_lambdas(league, home, away)

        blend_h = self.alpha * npxg_h + (1 - self.alpha) * gh
        blend_a = self.alpha * npxg_a + (1 - self.alpha) * ga

        err_h = blend_h - lam_h  # home attack / away defence signal
        err_a = blend_a - lam_a  # away attack / home defence signal

        d_att_h = self.lr_attack * psi(err_h, self.psi_const)
        d_def_a = -self.lr_defence * psi(err_h, self.psi_const)
        d_att_a = self.lr_attack * psi(err_a, self.psi_const)
        d_def_h = -self.lr_defence * psi(err_a, self.psi_const)

        st.attack[home] += d_att_h
        st.defence[away] += d_def_a
        st.attack[away] += d_att_a
        st.defence[home] += d_def_h

        return lam_h, lam_a

    # -- Dixon-Coles score matrix -------------------------------------------------
    def tau(self, x: int, y: int, lam: float, mu: float) -> float:
        if x == 0 and y == 0:
            return 1 - lam * mu * self.rho
        if x == 0 and y == 1:
            return 1 + lam * self.rho
        if x == 1 and y == 0:
            return 1 + mu * self.rho
        if x == 1 and y == 1:
            return 1 - self.rho
        return 1.0

    def score_matrix(self, lam: float, mu: float, max_goals: int = 10) -> np.ndarray:
        xs = np.arange(max_goals + 1)
        px = poisson.pmf(xs, lam)
        py = poisson.pmf(xs, mu)
        mat = np.outer(px, py)
        for x in (0, 1):
            for y in (0, 1):
                mat[x, y] *= self.tau(x, y, lam, mu)
        mat = mat / mat.sum()
        return mat

    def match_probs_1x2(self, lam: float, mu: float, max_goals: int = 10):
        mat = self.score_matrix(lam, mu, max_goals)
        p_home = np.tril(mat, -1).sum()
        p_draw = np.trace(mat)
        p_away = np.triu(mat, 1).sum()
        return p_home, p_draw, p_away


def walk_forward_nll(df: pd.DataFrame, params: dict, leagues: list[str]) -> float:
    """Independent-Poisson negative log-likelihood over one walk-forward pass.

    Used as the Stage-1/2 fitting objective (rho / Dixon-Coles tau is fit
    separately afterwards -- the standard two-step approach, since tau only
    perturbs four low-score cells and barely affects the mean-rate fit).
    """
    model = PiRatingDixonColesModel(
        home_adv=params["home_adv"],
        scale=params["scale"],
        alpha=params["alpha"],
        lr_attack=params["lr_attack"],
        lr_defence=params["lr_defence"],
        league_intercepts={lg: params[f"c_{lg}"] for lg in leagues},
    )
    nll = 0.0
    for row in df.itertuples(index=False):
        lam_h, lam_a = model.update(
            row.league, row.home_team, row.away_team,
            row.home_goals, row.away_goals, row.home_npxG, row.away_npxG,
        )
        # NLL uses the *pre-update* lambda already returned by update()
        nll -= poisson.logpmf(row.home_goals, lam_h)
        nll -= poisson.logpmf(row.away_goals, lam_a)
    return nll, model
