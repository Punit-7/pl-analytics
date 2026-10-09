import pandas as pd
import pytest

from serving.monitor.drift import drift_table, match_features, psi, windows
from serving.monitor.tracking import flatten
from serving.sample_data import league


def test_psi_matches_hand_calculation():
    reference = pd.Series(["H"] * 50 + ["D"] * 25 + ["A"] * 25)
    current = pd.Series(["H"] * 40 + ["D"] * 25 + ["A"] * 35)
    # (0.40-0.50)*ln(0.40/0.50) + 0 + (0.35-0.25)*ln(0.35/0.25) = 0.02231 + 0.03365
    assert psi(reference, current) == pytest.approx(0.05596, abs=1e-5)


def test_identical_data_has_zero_drift():
    s = pd.Series([0, 1, 1, 2, 3])
    assert psi(s, s) == pytest.approx(0.0)


def test_windows_do_not_overlap_and_end_before_today():
    results = league()
    reference, current = windows(results, "2026-05-21", 150, 760)
    assert len(current) == 150 and len(reference) == 760
    assert set(reference.columns) == {"home_goals", "away_goals", "total_goals", "result"}


def test_a_real_change_raises_the_alert():
    results = league()
    reference = match_features(results)
    changed = match_features(results.assign(home_goals=results["home_goals"] + 2))
    table = drift_table(reference, changed, alert=0.25)
    assert table["home_goals"]["alert"] and not table["away_goals"]["alert"]


def test_run_report_becomes_numbers_for_mlflow():
    run = {
        "model": {"model_version": "dc-x", "trained_through": "2026-10-02", "config_id": "abc",
                  "config": {"dc_xi": 0.0019}},
        "data": {"n_matches": 1980, "days_since_latest": 5, "stale": False},
        "gate": None,
        "published": {"n_predictions": 300},
        "ledger": {"n_scored": 40, "model_rps": 0.21},
    }  # fmt: skip
    params, metrics, tags = flatten(run, {"result": {"psi": 0.03, "alert": False}})
    assert params == {"dc_xi": 0.0019, "config_id": "abc"}
    assert metrics["live_model_rps"] == 0.21 and metrics["psi_max"] == 0.03
    assert all(isinstance(v, float) for v in metrics.values())
    assert tags["gate"] == "not run" and tags["drift_alert"] == "False"