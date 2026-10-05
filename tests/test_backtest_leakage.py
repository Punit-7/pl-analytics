import numpy as np
import pandas as pd
import pytest

from data.common.config import ROOT

PATH = ROOT / "modelling" / "reports" / "backtest_predictions.csv"


@pytest.fixture(scope="module")
def backtest():
    if not PATH.exists():
        pytest.skip("run python -m modelling.match.backtest first")
    return pd.read_csv(PATH, parse_dates=["match_date", "train_max_date"])


def test_training_data_is_strictly_before_each_prediction(backtest):
    leaks = backtest[backtest["train_max_date"] >= backtest["match_date"]]
    assert leaks.empty, f"{len(leaks)} predictions trained on same-day or later matches"


def test_probabilities_are_valid(backtest):
    for name in ["dc", "poisson", "freq", "eqpois"]:
        p = backtest[[f"{name}_h", f"{name}_d", f"{name}_a"]].to_numpy()
        assert ((p >= 0) & (p <= 1)).all()
        assert np.allclose(p.sum(axis=1), 1, atol=1e-6)
