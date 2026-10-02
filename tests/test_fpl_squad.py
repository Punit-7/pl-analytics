import numpy as np
import pandas as pd
import pytest

from data.common.io_utils import DataValidationError
from data.ingest.fpl import validate
from modelling.match.fpl_squad import optimise, project

RULES = {
    "squad": 15,
    "starters": 11,
    "club_limit": 3,
    "budget": 100.0,
    "positions": {"GKP": (2, 1, 1), "DEF": (5, 3, 5), "MID": (5, 2, 5), "FWD": (3, 1, 3)},
}
SCORING = {
    "long_play": 2,
    "assists": 3,
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
}


def fake_players(seed: int = 0) -> pd.DataFrame:
    """8 clubs, each with 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards."""
    rng = np.random.default_rng(seed)
    rows = [
        {"team": f"C{club}", "position": position}
        for club in range(8)
        for position, (count, _, _) in RULES["positions"].items()
        for _ in range(count)
    ]
    df = pd.DataFrame(rows)
    df["price"] = rng.uniform(4.0, 13.0, len(df)).round(1)
    df["points"] = (df["price"] * 2 + rng.normal(0, 3, len(df))).clip(lower=0)  # dearer = better
    return df


def test_squad_obeys_every_rule():
    squad = optimise(fake_players(), RULES)
    starters = squad[squad["start"]]
    assert len(squad) == 15 and len(starters) == 11
    assert squad["price"].sum() <= RULES["budget"] + 1e-9
    assert squad["team"].value_counts().max() <= RULES["club_limit"]
    for position, (select, min_play, max_play) in RULES["positions"].items():
        assert (squad["position"] == position).sum() == select
        assert min_play <= (starters["position"] == position).sum() <= max_play
    # one captain: the starter with the most projected points
    assert squad["captain"].sum() == 1
    assert squad.loc[squad["captain"], "points"].iloc[0] == starters["points"].max()


def test_squad_is_the_best_one_when_nothing_binds():
    players = fake_players()
    easy = {**RULES, "budget": 1000.0, "club_limit": 15}
    starters = optimise(players, easy).query("start")
    # with no budget or club limit, the XI holds the best goalkeeper and the best forward
    for position in ("GKP", "FWD"):
        best = players.loc[players["position"] == position, "points"].max()
        assert best in starters["points"].to_numpy()


def test_projection_matches_hand_calculation():
    players = pd.DataFrame(
        {
            "team": ["A", "A"],
            "position": ["MID", "DEF"],
            "play": [0.8, 1.0],
            "chance": [1.0, 1.0],
            "xg_share": [0.25, 0.0],
            "xa_share": [0.20, 0.0],
        }
    )
    outlook = pd.DataFrame(
        {"games": [5], "goals_for": [8.0], "clean_sheets": [1.5], "conceded_pairs": [2.0]},
        index=["A"],
    )
    points = project(players, outlook, SCORING, assists_per_goal=0.7)
    assert points[0] == pytest.approx(8.0 + 10.0 + 3.36 + 1.2)  # midfielder: 22.56
    assert points[1] == pytest.approx(10.0 + 6.0 - 2.0)  # defender: 14.0


def test_validate_rejects_wrong_json():
    with pytest.raises(DataValidationError, match="missing keys"):
        validate(b'{"elements": []}', "bootstrap-static")
    with pytest.raises(DataValidationError, match="not readable"):
        validate(b"<html>Not found</html>", "fixtures")
