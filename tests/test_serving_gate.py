import numpy as np
import pandas as pd
import pytest

import serving.gate as gate
from serving.gate import decide, outcome_index, walk_forward
from serving.sample_data import league

CFG = {"gate_min_matches": 150, "gate_min_improvement": 0.001, "gate_require_significance": True,
       "random_seed": 1}  # fmt: skip
CONFIG = {"dc_xi": 0.0019, "dc_history_days": 1095, "dc_max_goals": 10, "fit_rho": True}


def frame(rps_values):
    return pd.DataFrame({"rps": rps_values})


def test_outcome_index():
    assert list(outcome_index(np.array([2, 1, 0]), np.array([0, 1, 3]))) == [0, 1, 2]


def test_clear_gain_promotes():
    rng = np.random.default_rng(0)
    champion = rng.uniform(0.15, 0.30, 300)
    d = decide(frame(champion), frame(champion - 0.01), CFG)
    assert d.promoted and d.gain == pytest.approx(0.01) and d.gain_low_95 > 0


def test_too_few_matches_never_promotes():
    d = decide(frame(np.full(100, 0.25)), frame(np.full(100, 0.10)), CFG)
    assert not d.promoted and "only 100" in d.reason


def test_small_gain_is_refused():
    champion = np.full(300, 0.2)
    d = decide(frame(champion), frame(champion - 0.0005), CFG)
    assert not d.promoted and "below the required" in d.reason


def test_noisy_gain_is_refused():
    rng = np.random.default_rng(3)
    champion = rng.uniform(0.0, 0.5, 300)
    challenger = champion - 0.002 + rng.normal(0, 0.2, 300)  # better on average, mostly noise
    d = decide(frame(champion), frame(challenger), {**CFG, "gate_min_improvement": -1})
    assert not d.promoted and "noise" in d.reason


def test_walk_forward_never_trains_on_a_match_it_tests(monkeypatch):
    results = league()
    seen = []

    class Dummy:
        def outcome_probs(self, home, away):
            return 0.45, 0.27, 0.28

    def fake_fit(matches, as_of, *args, **kwargs):
        seen.append(pd.Timestamp(as_of))
        return Dummy()

    monkeypatch.setattr(gate, "fit_dixon_coles", fake_fit)
    out = walk_forward(results, CONFIG, "2026-05-21", 300)
    assert len(out) >= 300 and out["rps"].between(0, 1).all()
    tested_from = results["match_date"].sort_values().tail(len(out)).min()
    assert min(seen) <= tested_from  # every fit date is at or before the week it predicts
    assert len(seen) == out.shape[0] // 10  # one fit per week of ten matches
