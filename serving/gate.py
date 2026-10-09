"""Champion/challenger gate: compare two configurations on matches neither has trained on."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from modelling.match.dixon_coles import DixonColes, fit_dixon_coles
from modelling.metrics import rps

log = logging.getLogger(__name__)


def outcome_index(home_goals, away_goals):
    """0 = home win, 1 = draw, 2 = away win. The same rule as in P2's backtest."""
    return np.select([home_goals > away_goals, home_goals == away_goals], [0, 1], default=2)


def fit(results: pd.DataFrame, config: dict, as_of) -> DixonColes:
    """Fit P2's Dixon-Coles model with one configuration, on matches before `as_of`."""
    return fit_dixon_coles(
        results,
        as_of,
        config["dc_xi"],
        config["dc_history_days"],
        config["dc_max_goals"],
        fit_rho=config["fit_rho"],
    )


def walk_forward(results: pd.DataFrame, config: dict, end, n_matches: int) -> pd.DataFrame:
    """Refit at the start of each week and predict that week's matches, over the most recent
    `n_matches` matches before `end`. No model ever sees a match it is tested on."""
    end = pd.Timestamp(end).normalize()
    past = results[results["match_date"] < end]
    start = past["match_date"].tail(n_matches).min()
    test = past[past["match_date"] >= start].copy()
    test["week"] = test["match_date"] - pd.to_timedelta(test["match_date"].dt.weekday, unit="D")
    rows = []
    for week, games in test.groupby("week"):
        model = fit(results, config, week)  # sees only matches before this week
        for g in games.itertuples():
            p = model.outcome_probs(g.home_team, g.away_team)
            rows.append(
                {
                    "season": g.season,
                    "home_team": g.home_team,
                    "away_team": g.away_team,
                    "outcome": int(outcome_index(g.home_goals, g.away_goals)),
                    "p_home": p[0],
                    "p_draw": p[1],
                    "p_away": p[2],
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        probs = out[["p_home", "p_draw", "p_away"]].to_numpy()
        out["rps"] = rps(probs, out["outcome"].to_numpy())
    return out


def paired_bootstrap(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the mean of per-match differences, by resampling matches."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    low, high = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


@dataclass
class Decision:
    promoted: bool
    reason: str
    n_matches: int = 0
    champion_rps: float | None = None
    challenger_rps: float | None = None
    gain: float | None = None
    gain_low_95: float | None = None
    gain_high_95: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def decide(champion: pd.DataFrame, challenger: pd.DataFrame, cfg: dict) -> Decision:
    """Promote only if the challenger's RPS is lower by enough, on enough matches."""
    n = len(champion)
    if n < cfg["gate_min_matches"]:
        return Decision(False, f"only {n} test matches; {cfg['gate_min_matches']} needed", n)
    gain_per_match = champion["rps"].to_numpy() - challenger["rps"].to_numpy()  # positive = better
    gain = float(gain_per_match.mean())
    low, high = paired_bootstrap(gain_per_match, seed=cfg["random_seed"])
    d = Decision(False, "", n, float(champion["rps"].mean()), float(challenger["rps"].mean()),
                 gain, low, high)  # fmt: skip
    if gain < cfg["gate_min_improvement"]:
        d.reason = f"gain {gain:.4f} is below the required {cfg['gate_min_improvement']}"
    elif cfg["gate_require_significance"] and low <= 0:
        d.reason = f"gain {gain:.4f} is not clear of noise (95% interval {low:.4f} to {high:.4f})"
    else:
        d.promoted, d.reason = True, f"gain {gain:.4f} with 95% interval {low:.4f} to {high:.4f}"
    return d


def run_gate(results: pd.DataFrame, champion: dict, challenger: dict, end, cfg: dict) -> Decision:
    a = walk_forward(results, champion, end, cfg["gate_matches"])
    b = walk_forward(results, challenger, end, cfg["gate_matches"])
    decision = decide(a, b, cfg)
    log.info("Gate: promoted=%s (%s)", decision.promoted, decision.reason)
    return decision
