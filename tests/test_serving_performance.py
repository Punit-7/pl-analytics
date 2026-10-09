import numpy as np
import pandas as pd
import pytest

from serving.monitor.performance import add_baseline, alert_text, check, escalate

RULES = {
    "window_matches": 100,
    "min_matches": 60,
    "baseline_history": 760,
    "max_gap_to_book": 0.02,
    "max_calibration_error": 0.08,
    "amber_runs_to_red": 3,
}


def results(n=40):
    """History: 50% home wins, 25% draws, 25% away wins, all before October."""
    goals = [(1, 0), (1, 0), (0, 0), (0, 1)] * (n // 4)
    return pd.DataFrame(
        {
            "match_date": pd.date_range("2026-08-01", periods=n, freq="D"),
            "home_goals": [g[0] for g in goals],
            "away_goals": [g[1] for g in goals],
        }
    )


def ledger(n, p_home, outcome, book_rps=0.20):
    """n scored matches, all with the same forecast and the same result."""
    return pd.DataFrame(
        {
            "match_date": pd.date_range("2026-10-01", periods=n, freq="D"),
            "p_home": p_home[0],
            "p_draw": p_home[1],
            "p_away": p_home[2],
            "outcome": outcome,
            "book_rps": book_rps,
        }
    )


def test_baseline_uses_only_earlier_matches():
    lg = ledger(1, (0.5, 0.25, 0.25), 0)
    out = add_baseline(lg, results(), 760)
    assert out.loc[0, ["base_home", "base_draw", "base_away"]].tolist() == [0.5, 0.25, 0.25]
    # Home win. Cumulative forecast 0.50, 0.75; outcome 1, 1.
    assert out.loc[0, "base_rps"] == pytest.approx(((0.5 - 1) ** 2 + (0.75 - 1) ** 2) / 2)


def test_too_few_matches_gives_no_verdict():
    lg = add_baseline(ledger(30, (0.6, 0.2, 0.2), 0), results(), 760)
    lg["model_rps"] = 0.1
    assert check(lg, RULES)["level"] == "waiting" and "needed" in check(lg, RULES)["note"]


def test_model_clearly_worse_than_baseline_is_red():
    lg = add_baseline(ledger(100, (0.1, 0.2, 0.7), 0), results(), 760)  # backs away; home wins
    rng = np.random.default_rng(0)
    lg["model_rps"] = lg["base_rps"] + 0.05 + rng.normal(0, 0.01, len(lg))
    result = check(lg, RULES)
    assert result["level"] == "red"
    assert result["checks"][0]["low_95"] > 0


def test_model_better_than_baseline_and_close_to_book_is_green():
    lg = add_baseline(ledger(100, (0.5, 0.25, 0.25), [0, 0, 1, 2] * 25), results(), 760)
    lg["model_rps"] = lg["base_rps"] - 0.01
    lg["book_rps"] = lg["model_rps"] - 0.005
    result = check(lg, RULES)
    names = {c["name"]: c["level"] for c in result["checks"]}
    assert names["vs_baseline"] == "green" and names["vs_bookmaker"] == "green"
    assert names["calibration"] == "green" and result["level"] == "green"


def test_large_gap_to_book_is_amber():
    lg = add_baseline(ledger(100, (0.5, 0.25, 0.25), [0, 0, 1, 2] * 25), results(), 760)
    lg["model_rps"] = lg["base_rps"] - 0.01
    lg["book_rps"] = lg["model_rps"] - 0.03
    assert check(lg, RULES)["level"] == "amber"


def test_three_amber_runs_with_new_matches_become_red():
    amber = {"level": "amber", "n_matches": 100, "checks": []}
    history = [{"n_scored": 100, "level": "amber"}, {"n_scored": 110, "level": "amber"}]
    assert escalate(amber, history, 120, 3)["level"] == "red"
    assert escalate(amber, history, 110, 3)["level"] == "amber"  # no new matches this week
    assert escalate(amber, history[:1], 120, 3)["level"] == "amber"  # only two in a row
    history[0]["level"] = "green"
    assert escalate(amber, history, 120, 3)["level"] == "amber"


def test_alert_text_names_every_check():
    result = {
        "level": "red",
        "n_matches": 100,
        "first_match": "a",
        "last_match": "b",
        "checks": [{"name": "vs_baseline", "level": "red", "reason": "worse"}],
    }
    text = alert_text(result, "2026-12-03T06:17:00+00:00")
    assert "**RED**" in text and "| vs_baseline | red | worse |" in text
