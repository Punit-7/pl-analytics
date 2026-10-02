import numpy as np
import pandas as pd

from modelling.match.dixon_coles import DixonColes
from modelling.match.simulate import simulate


def test_simulated_probabilities_add_up():
    teams = [f"T{i}" for i in range(6)]
    model = DixonColes(teams, np.linspace(-0.3, 0.3, 6), np.linspace(0.2, -0.2, 6), 0.25, -0.05)
    fixtures = pd.DataFrame([(h, a) for h in teams for a in teams if h != a],
                            columns=["home_team", "away_team"])
    table = pd.DataFrame(columns=["team", "pts", "gf", "ga"])
    res = simulate(model, table, fixtures, n_sims=2000, seed=1)
    assert abs(res["p_title"].sum() - 1) < 1e-9
    assert abs(res["p_top4"].sum() - 4) < 1e-9
    assert abs(res["p_relegation"].sum() - 3) < 1e-9
    assert res.loc[res["p_title"].idxmax(), "team"] == "T5"