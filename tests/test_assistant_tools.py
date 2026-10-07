import numpy as np
import pytest

from assistant.tools import MatchTools, ToolError, most_likely_score, resolve_team
from data.common.config import load_settings
from data.common.db import make_reader_engine
from modelling.match.dixon_coles import DixonColes

LOOKUP = {"man united": "Man United", "manchester united": "Man United", "arsenal": "Arsenal",
          "nott'm forest": "Nott'm Forest", "nottingham forest": "Nott'm Forest"}  # fmt: skip


def test_resolve_team_handles_full_names_suffixes_and_typos():
    assert resolve_team("Manchester United", LOOKUP) == "Man United"
    assert resolve_team("Arsenal F.C.", LOOKUP) == "Arsenal"
    assert resolve_team("Arsenall", LOOKUP) == "Arsenal"
    assert resolve_team("Real Madrid", LOOKUP) is None


def test_most_likely_score_is_the_largest_cell():
    model = DixonColes(["A", "B", "C", "D"], np.array([0.4, 0.0, -0.2, -0.2]),
                       np.array([-0.3, 0.0, 0.1, 0.2]), home_adv=0.25, rho=-0.05)  # fmt: skip
    score, p = most_likely_score(model, "A", "D")
    matrix = model.score_matrix("A", "D")
    home, away = (int(x) for x in score.split("-"))
    assert p == pytest.approx(matrix.max()) and matrix[home, away] == matrix.max()


@pytest.fixture(scope="module")
def tools():
    try:
        return MatchTools(make_reader_engine(), load_settings())
    except Exception as err:
        pytest.skip(f"no read-only database connection: {err}")


def test_prediction_probabilities_add_up_to_one(tools):
    p = tools.predict_match("Arsenal", "Chelsea")
    assert p["home_team"] == "Arsenal" and p["away_team"] == "Chelsea"
    assert p["p_home_win"] + p["p_draw"] + p["p_away_win"] == pytest.approx(1.0, abs=0.002)


def test_unknown_team_is_a_tool_error(tools):
    with pytest.raises(ToolError):
        tools.predict_match("Arsenal", "Real Madrid")


def test_ratings_cover_this_season_and_are_ranked(tools):
    ratings = tools.team_ratings()["ratings"]
    assert len(ratings) == 20 and ratings[0]["strength"] >= ratings[-1]["strength"]
    assert tools.team_ratings("Arsenal")["ratings"][0]["team"] == "Arsenal"