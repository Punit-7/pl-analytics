import json

import numpy as np
import pytest

from modelling.match.dixon_coles import DixonColes
from serving.model_io import dict_to_model, load_model, model_to_dict, save_model

MODEL = DixonColes(["A", "B", "C", "D"], np.array([0.4, 0.1, -0.2, -0.3]),
                   np.array([-0.3, 0.0, 0.1, 0.2]), home_adv=0.25, rho=-0.05)  # fmt: skip


def test_saved_model_gives_the_same_probabilities(tmp_path):
    save_model(tmp_path / "m.json", MODEL, {"model_version": "test"})
    loaded, meta = load_model(tmp_path / "m.json")
    assert meta["model_version"] == "test"
    assert loaded.outcome_probs("A", "D") == pytest.approx(MODEL.outcome_probs("A", "D"), abs=1e-5)


def test_file_is_plain_readable_json(tmp_path):
    save_model(tmp_path / "m.json", MODEL, {})
    data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert data["teams"] == ["A", "B", "C", "D"] and data["format"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"attack": [0.1, 0.2]},  # wrong length
        {"rho": float("nan")},  # not a finite number
        {"teams": ["A", "A", "C", "D"]},  # a team twice
        {"format": 99},
    ],
)
def test_damaged_files_are_refused(change):
    with pytest.raises(ValueError):
        dict_to_model({**model_to_dict(MODEL, {}), **change})