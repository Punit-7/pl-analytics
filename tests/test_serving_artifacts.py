"""Checks on the files the weekly job commits. The API container is built from these files."""

import json

import pytest

from serving.api.service import LATEST, MODEL, ModelService

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run python -m serving.weekly first")


def test_committed_model_loads_and_gives_valid_probabilities():
    service = ModelService()
    teams = service.model.teams
    p = service.predict_match(teams[0], teams[1])
    assert p["p_home_win"] + p["p_draw"] + p["p_away_win"] == pytest.approx(1.0, abs=1e-3)
    assert service.health()["model_version"].startswith("dc-")


def test_published_predictions_come_from_the_committed_model():
    if not LATEST.exists():
        pytest.skip("nothing published yet")
    latest = json.loads(LATEST.read_text(encoding="utf-8"))
    service = ModelService()
    assert latest["model_version"] == service.meta["model_version"]
    for m in latest["matches"]:
        assert m["p_home"] + m["p_draw"] + m["p_away"] == pytest.approx(1.0, abs=2e-4)
    if latest["season_sim"]:
        assert sum(t["p_title"] for t in latest["season_sim"]) == pytest.approx(1.0, abs=1e-3)