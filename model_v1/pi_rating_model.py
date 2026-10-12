"""
Dual attack/defence pi-rating model for team scoring rates.

Each team carries an attack and a defence rating per league. Expected goals for the home and away
side are log-linear in the rating gaps (Maher 1982; Dyte & Clarke 2000):

    log(lambda_home) = c_league + home_adv + (attack_home - defence_away) / scale
    log(lambda_away) = c_league +            (attack_away - defence_home) / scale

After each match the ratings move by a pi-rating style update (Constantinou & Fenton 2013): the error
between a blend of non-penalty xG and goals, alpha*npxG + (1-alpha)*goals, and the predicted rate,
passed through the diminishing-returns transform psi(e) = sign(e) * 3 * log10(1 + |e|), times a
learning rate. Hyperparameters are fitted by walk-forward Poisson likelihood (walk_forward_nll).

The model supplies the expected goals for (lambda) and against (mu) that the player-level xP
components use as covariates; no score-matrix layer is needed downstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
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


class PiRatingModel:
    """Dual attack/defence pi-rating model producing expected goals for both sides of a fixture."""

    def __init__(
        self,
        home_adv: float,
        scale: float,
        alpha: float,
        lr_attack: float,
        lr_defence: float,
        league_intercepts: dict,
        psi_const: float = 3.0,
    ):
        self.home_adv = home_adv
        self.scale = scale
        self.alpha = alpha
        self.lr_attack = lr_attack
        self.lr_defence = lr_defence
        self.league_intercepts = league_intercepts
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
        """Return the PRE-match (lambda_home, lambda_away), then update the ratings with the result."""
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


def walk_forward_nll(df: pd.DataFrame, params: dict, leagues: list[str]):
    """Poisson negative log-likelihood of home and away goals over one walk-forward pass
    (each match scored with the rates predicted before it). Returns (nll, model)."""
    model = PiRatingModel(
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
        nll -= poisson.logpmf(row.home_goals, lam_h)
        nll -= poisson.logpmf(row.away_goals, lam_a)
    return nll, model
