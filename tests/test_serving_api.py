import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelling.match.dixon_coles import DixonColes
from modelling.xg.features import build_matrix
from serving.api import main
from serving.api.service import ModelService
from serving.model_io import save_model

TEAMS = ["Arsenal", "Chelsea", "Leeds", "Everton"]


def tiny_xg_model(path):
    """A small model trained on invented shots, with the same columns as P2's."""
    rng = np.random.default_rng(0)
    n = 400
    shots = pd.DataFrame(
        {
            "x": rng.uniform(85, 119, n),
            "y": rng.uniform(20, 60, n),
            "body_part": rng.choice(["Right Foot", "Left Foot", "Head"], n),
            "shot_type": rng.choice(["Open Play", "Free Kick"], n),
            "play_pattern": rng.choice(["Regular Play", "From Corner"], n),
            "technique": rng.choice(["Normal", "Volley"], n),
            "under_pressure": rng.random(n) < 0.3,
            "first_time": rng.random(n) < 0.3,
        }
    )
    X = build_matrix(shots)
    y = (rng.random(n) < np.clip(0.9 - X["distance"] / 30, 0.02, 0.9)).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X, y)
    joblib.dump(model, path)


@pytest.fixture
def client(tmp_path):
    model = DixonColes(TEAMS, np.array([0.4, 0.1, -0.2, -0.3]), np.array([-0.3, 0.0, 0.1, 0.2]),
                       home_adv=0.25, rho=-0.05)  # fmt: skip
    meta = {"model_version": "dc-test", "fitted_on": "2026-10-08", "trained_through": "2026-10-05"}
    save_model(tmp_path / "model.json", model, meta)
    latest = {
        "created_at": "2026-10-08T06:00:00",
        "model_version": "dc-test",
        "season": "2026/27",
        "matches": [
            {"home_team": "Arsenal", "away_team": "Chelsea", "p_home": 0.5},
            {"home_team": "Leeds", "away_team": "Everton", "p_home": 0.4},
        ],
        "season_sim": [{"team": t, "p_title": 0.25} for t in TEAMS],
    }
    (tmp_path / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    tiny_xg_model(tmp_path / "xg.joblib")
    main.state["service"] = ModelService(
        tmp_path / "model.json", tmp_path / "latest.json", tmp_path / "xg.joblib"
    )
    yield TestClient(main.app)
    main.state.clear()


def test_health_reports_the_model_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"] == "dc-test"
    assert body["teams"] == 4 and body["xg_model_loaded"] is True


def test_match_prediction_is_a_valid_forecast(client):
    r = client.post("/predict/match", json={"home_team": "arsenal", "away_team": "Chelsea"})
    body = r.json()
    assert r.status_code == 200 and body["home_team"] == "Arsenal"
    assert body["p_home_win"] + body["p_draw"] + body["p_away_win"] == pytest.approx(1.0, abs=1e-3)
    assert body["p_home_win"] > body["p_away_win"]


def test_unknown_team_is_404_with_the_valid_names(client):
    r = client.post("/predict/match", json={"home_team": "Arsenal", "away_team": "Real Madrid"})
    assert r.status_code == 404 and "Chelsea" in r.json()["detail"]["teams"]


def test_same_team_twice_and_missing_field_are_422(client):
    same = client.post("/predict/match", json={"home_team": "Arsenal", "away_team": "Arsenal"})
    missing = client.post("/predict/match", json={"home_team": "Arsenal"})
    assert same.status_code == 422 and missing.status_code == 422


def test_teams_are_ranked_by_strength(client):
    ranked = client.get("/teams").json()
    assert [t["rank"] for t in ranked] == [1, 2, 3, 4] and ranked[0]["team"] == "Arsenal"


def test_published_predictions_can_be_filtered_by_team(client):
    assert len(client.get("/predictions/latest").json()["matches"]) == 2
    leeds = client.get("/predictions/latest", params={"team": "Leeds"}).json()
    assert [m["home_team"] for m in leeds["matches"]] == ["Leeds"]
    assert client.get("/season").json()["season_sim"][0]["team"] == "Arsenal"


def test_xg_is_higher_close_to_goal(client):
    close = client.post("/predict/xg", json={"x": 114, "y": 40}).json()
    far = client.post("/predict/xg", json={"x": 90, "y": 40}).json()
    assert 0 < far["xg"] < close["xg"] < 1 and close["distance"] == 6.0


def test_xg_rejects_impossible_shots(client):
    off_pitch = client.post("/predict/xg", json={"x": 130, "y": 40})
    bad_value = client.post("/predict/xg", json={"x": 100, "y": 40, "body_part": "Elbow"})
    assert off_pitch.status_code == 422 and bad_value.status_code == 422