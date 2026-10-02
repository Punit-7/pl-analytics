"""Dixon-Coles match model, written from scratch."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

log = logging.getLogger(__name__)


def tau(x, y, lam, mu, rho):
    """Low-score correction: changes only 0-0, 0-1, 1-0 and 1-1."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


@dataclass
class DixonColes:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    max_goals: int = 10
    _index: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {t: i for i, t in enumerate(self.teams)}
        # Prior for a team with no data (a newly promoted side before its first match):
        # the average rating of the three weakest teams in the fit.
        weakest = np.argsort(self.attack - self.defence)[:3]
        self.promoted_attack = float(self.attack[weakest].mean())
        self.promoted_defence = float(self.defence[weakest].mean())

    def rating(self, team: str) -> tuple[float, float]:
        if team in self._index:
            i = self._index[team]
            return float(self.attack[i]), float(self.defence[i])
        return self.promoted_attack, self.promoted_defence

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        a_h, d_h = self.rating(home)
        a_a, d_a = self.rating(away)
        return float(np.exp(a_h + d_a + self.home_adv)), float(np.exp(a_a + d_h))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """Rows = home goals 0..max, columns = away goals 0..max; sums to 1."""
        lam, mu = self.expected_goals(home, away)
        goals = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        return m / m.sum()

    def outcome_probs(self, home: str, away: str) -> tuple[float, float, float]:
        m = self.score_matrix(home, away)
        return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def fit_dixon_coles(matches: pd.DataFrame, as_of, xi: float, history_days: int | None = None,
                    max_goals: int = 10, fit_rho: bool = True) -> DixonColes:
    """Fit on matches strictly before `as_of`. Columns: home_team, away_team,
    home_goals, away_goals, match_date."""
    as_of = pd.Timestamp(as_of)
    dates = pd.to_datetime(matches["match_date"])
    keep = dates < as_of
    if history_days is not None:
        keep &= dates >= as_of - pd.Timedelta(days=history_days)
    df = matches.loc[keep]
    if df.empty:
        raise ValueError(f"no matches before {as_of.date()}")

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    h = df["home_team"].map(idx).to_numpy()
    a = df["away_team"].map(idx).to_numpy()
    x = df["home_goals"].to_numpy(dtype=int)
    y = df["away_goals"].to_numpy(dtype=int)
    w = np.exp(-xi * (as_of - pd.to_datetime(df["match_date"])).dt.days.to_numpy())
    n = len(teams)

    def unpack(p):
        attack = p[:n] - p[:n].mean()  # identifiability: attack ratings average zero
        return attack, p[n:2 * n], p[2 * n], p[2 * n + 1]

    def neg_log_lik(p):
        attack, defence, home, rho = unpack(p)
        lam = np.exp(attack[h] + defence[a] + home)
        mu = np.exp(attack[a] + defence[h])
        t = np.clip(tau(x, y, lam, mu, rho), 1e-10, None)
        ll = np.log(t) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu)
        return -np.sum(w * ll)

    p0 = np.concatenate([np.zeros(2 * n), [0.25, 0.0]])
    rho_bounds = (-0.2, 0.2) if fit_rho else (0.0, 0.0)
    bounds = [(None, None)] * (2 * n + 1) + [rho_bounds]
    res = minimize(neg_log_lik, p0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        log.warning("Dixon-Coles fit as of %s did not fully converge: %s", as_of.date(), res.message)
    attack, defence, home, rho = unpack(res.x)
    return DixonColes(teams, attack, defence, float(home), float(rho), max_goals)