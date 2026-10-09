import pandas as pd
import pytest

from serving.ledger import build_ledger, implied_probs, load_predictions, summarise

RESULTS = pd.DataFrame(
    {
        "season": ["2026/27"] * 2,
        "match_date": pd.to_datetime(["2026-10-10", "2026-10-11"]),
        "home_team": ["Arsenal", "Leeds"],
        "away_team": ["Chelsea", "Everton"],
        "home_goals": [2, 0],
        "away_goals": [0, 0],
        "avg_close_home_odds": [2.10, None],
        "avg_close_draw_odds": [3.40, None],
        "avg_close_away_odds": [3.60, None],
    }
)


def prediction(created_at, home, away, p_home, p_draw, p_away, pick):
    return {"season": "2026/27", "home_team": home, "away_team": away, "created_at": created_at,
            "model_version": "v", "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "predicted_result": pick}  # fmt: skip


PREDS = pd.DataFrame(
    [
        prediction("2026-10-01 06:00", "Arsenal", "Chelsea", 0.40, 0.30, 0.30, "home"),
        prediction("2026-10-08 06:00", "Arsenal", "Chelsea", 0.50, 0.30, 0.20, "home"),
        prediction("2026-10-10 06:00", "Arsenal", "Chelsea", 0.90, 0.05, 0.05, "home"),  # match day
        prediction("2026-10-08 06:00", "Leeds", "Everton", 0.50, 0.25, 0.25, "home"),
        prediction("2026-10-08 06:00", "Fulham", "Wolves", 0.40, 0.30, 0.30, "home"),  # not played
    ]
)
PREDS["created_at"] = pd.to_datetime(PREDS["created_at"])


def test_latest_prediction_before_the_match_day_is_the_one_scored():
    ledger = build_ledger(PREDS, RESULTS)
    assert len(ledger) == 2
    arsenal = ledger[ledger["home_team"] == "Arsenal"].iloc[0]
    assert arsenal["p_home"] == 0.50 and arsenal["days_ahead"] == 2  # not 0.40, and not 0.90


def test_rps_matches_hand_calculation():
    ledger = build_ledger(PREDS, RESULTS).set_index("home_team")
    # Arsenal won. Cumulative forecast 0.5, 0.8; cumulative outcome 1, 1.
    assert ledger.loc["Arsenal", "model_rps"] == pytest.approx(
        ((0.5 - 1) ** 2 + (0.8 - 1) ** 2) / 2
    )
    # Leeds drew. Cumulative forecast 0.5, 0.75; cumulative outcome 0, 1.
    assert ledger.loc["Leeds", "model_rps"] == pytest.approx((0.5**2 + (0.75 - 1) ** 2) / 2)
    assert bool(ledger.loc["Arsenal", "pick_correct"]) and not bool(
        ledger.loc["Leeds", "pick_correct"]
    )


def test_bookmaker_score_only_where_odds_exist():
    ledger = build_ledger(PREDS, RESULTS).set_index("home_team")
    assert pd.isna(ledger.loc["Leeds", "book_rps"])
    odds = RESULTS.loc[[0], ["avg_close_home_odds", "avg_close_draw_odds", "avg_close_away_odds"]]
    book = implied_probs(odds.to_numpy(dtype=float))
    assert book.sum() == pytest.approx(1.0) and book[0, 0] == pytest.approx(0.4543, abs=1e-4)
    summary = summarise(build_ledger(PREDS, RESULTS))
    assert summary["n_scored"] == 2 and summary["n_with_odds"] == 1
    assert summary["pick_accuracy"] == 0.5


def test_empty_ledger_is_handled():
    assert summarise(build_ledger(PREDS.head(0), RESULTS)) == {"n_scored": 0}


def test_blank_prediction_file_is_skipped(tmp_path):
    season = tmp_path / "2026-27"
    season.mkdir()
    (season / "a_matches.csv").write_text("\n")  # what an empty week's publish wrote
    pd.DataFrame([prediction("2026-10-05", "Arsenal", "Chelsea", 0.5, 0.3, 0.2, "home")]).to_csv(
        season / "b_matches.csv", index=False
    )
    assert len(load_predictions([tmp_path])) == 1
