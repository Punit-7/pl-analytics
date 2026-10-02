import numpy as np
import pandas as pd

from modelling.match.dixon_coles import fit_dixon_coles


def synthetic_league(seed: int = 0):
    """10 teams, 10 double round-robins, known ratings, rho = 0."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    attack = np.linspace(-0.4, 0.4, 10)
    defence = np.linspace(0.3, -0.3, 10)  # stronger teams concede less
    rows, day = [], pd.Timestamp("2020-01-01")
    for _ in range(10):
        for i in range(10):
            for j in range(10):
                if i != j:
                    lam = np.exp(attack[i] + defence[j] + 0.3)
                    mu = np.exp(attack[j] + defence[i])
                    rows.append((teams[i], teams[j], rng.poisson(lam), rng.poisson(mu), day))
                    day += pd.Timedelta(days=1)
    cols = ["home_team", "away_team", "home_goals", "away_goals", "match_date"]
    return pd.DataFrame(rows, columns=cols), attack


def test_recovers_known_ratings():
    df, true_attack = synthetic_league()
    model = fit_dixon_coles(df, df["match_date"].max() + pd.Timedelta(days=1), xi=0.0)
    fitted = np.array([model.rating(f"T{i}")[0] for i in range(10)])
    assert abs(model.home_adv - 0.3) < 0.1
    assert np.corrcoef(fitted, true_attack)[0, 1] > 0.9


def test_probabilities_sum_to_one():
    df, _ = synthetic_league()
    model = fit_dixon_coles(df, df["match_date"].max() + pd.Timedelta(days=1), xi=0.0)
    assert abs(sum(model.outcome_probs("T9", "T0")) - 1) < 1e-9
    assert abs(model.score_matrix("T0", "T9").sum() - 1) < 1e-9


def test_future_matches_are_ignored():
    df, _ = synthetic_league()
    cutoff = df["match_date"].iloc[450]
    with_future = fit_dixon_coles(df, cutoff, xi=0.0)
    past_only = fit_dixon_coles(df[df["match_date"] < cutoff], cutoff, xi=0.0)
    assert np.allclose(with_future.attack, past_only.attack)