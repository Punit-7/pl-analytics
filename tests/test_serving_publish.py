from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from modelling.match.dixon_coles import DixonColes
from serving.publish import NotReady, current_table, match_predictions, publish, remaining_pairs
from serving.sample_data import TEAMS, league

RESULTS = league(last_day=date(2025, 12, 1))  # 2025/26 is part-played
MODEL = DixonColes(TEAMS, np.linspace(0.5, -0.5, 20), np.linspace(-0.5, 0.5, 20), 0.25, -0.05)
NOW = datetime(2025, 12, 4, 6, 0, tzinfo=UTC)


def test_remaining_pairs_are_all_pairings_minus_played():
    played = int((RESULTS["season"] == "2025/26").sum())
    pairs = remaining_pairs(RESULTS, "2025/26")
    assert len(pairs) == 380 - played
    done = set(zip(RESULTS["home_team"], RESULTS["away_team"], RESULTS["season"], strict=True))
    assert not any((h, a, "2025/26") in done for h, a in pairs.itertuples(index=False))


def test_a_season_without_twenty_teams_is_not_ready():
    with pytest.raises(NotReady):
        remaining_pairs(RESULTS[RESULTS["home_team"] != "Team A"].head(30), "2025/26")


def test_current_table_matches_hand_count():
    r = pd.DataFrame(
        {
            "season": ["s"] * 3,
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "C", "C"],
            "home_goals": [2, 1, 0],
            "away_goals": [0, 1, 3],
        }
    )
    table = current_table(r, "s").set_index("team")
    assert table.loc["A"].tolist() == [3, 2, 3]  # one win, one loss: 3 points, 2 for, 3 against
    assert table.loc["B"].tolist() == [1, 1, 3]
    assert table.loc["C"].tolist() == [4, 4, 1]


def test_predictions_are_valid_probabilities():
    preds = match_predictions(MODEL, remaining_pairs(RESULTS, "2025/26"), "2025/26", NOW, "v1")
    total = preds["p_home"] + preds["p_draw"] + preds["p_away"]
    assert total.between(0.999, 1.001).all() and preds["prediction_id"].is_unique
    assert set(preds["predicted_result"]) <= {"home", "draw", "away"}


def test_publish_writes_files_and_a_simulation_that_adds_up(tmp_path):
    cfg = {"n_sims": 200, "random_seed": 1}
    info = publish(MODEL, RESULTS, "2025/26", NOW, "v1", cfg, out=tmp_path)
    folder = tmp_path / "2025-26"
    sim = pd.read_csv(folder / f"{info['stamp']}_season_sim.csv")
    assert len(sim) == 20 and sim["p_title"].sum() == pytest.approx(1.0)
    assert sim["p_relegation"].sum() == pytest.approx(3.0)
    assert (tmp_path / "latest.json").exists()
