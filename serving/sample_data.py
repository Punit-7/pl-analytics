"""Invented match results, used by the tests. Never used by the weekly job."""

from datetime import date, timedelta
from itertools import permutations

import numpy as np
import pandas as pd

TEAMS = [f"Team {chr(65 + i)}" for i in range(20)]


def league(seasons=(2023, 2024, 2025), last_day=date(2026, 5, 20), seed=0) -> pd.DataFrame:
    """Full double round-robins with random scores. Team A is strongest, Team T weakest."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, np.linspace(0.5, -0.5, 20), strict=True))
    rows = []
    for year in seasons:
        pairs = list(permutations(TEAMS, 2))
        rng.shuffle(pairs)
        for i, (h, a) in enumerate(pairs):
            day = date(year, 8, 10) + timedelta(days=7 * (i // 10))
            if day > last_day:
                break
            rows.append(
                {
                    "season": f"{year}/{(year + 1) % 100:02d}",
                    "match_date": pd.Timestamp(day),
                    "home_team": h,
                    "away_team": a,
                    "home_goals": int(rng.poisson(np.exp(0.25 + strength[h] - strength[a]))),
                    "away_goals": int(rng.poisson(np.exp(strength[a] - strength[h]))),
                    "avg_close_home_odds": 2.1,
                    "avg_close_draw_odds": 3.4,
                    "avg_close_away_odds": 3.6,
                }
            )
    return pd.DataFrame(rows)
