"""Save and load the fitted Dixon-Coles model as a small JSON file."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from modelling.match.dixon_coles import DixonColes

FORMAT = 1


def model_to_dict(model: DixonColes, meta: dict) -> dict:
    return {
        "format": FORMAT,
        "teams": list(model.teams),
        "attack": [round(float(x), 6) for x in model.attack],
        "defence": [round(float(x), 6) for x in model.defence],
        "home_adv": round(float(model.home_adv), 6),
        "rho": round(float(model.rho), 6),
        "max_goals": int(model.max_goals),
        "meta": meta,
    }


def dict_to_model(data: dict) -> tuple[DixonColes, dict]:
    """Rebuild the model, refusing a file that is incomplete or contains bad numbers."""
    if data.get("format") != FORMAT:
        raise ValueError(f"unsupported model file format: {data.get('format')}")
    teams, attack, defence = data["teams"], data["attack"], data["defence"]
    if not (len(teams) == len(attack) == len(defence) >= 4) or len(set(teams)) != len(teams):
        raise ValueError("teams, attack and defence must have the same length, without repeats")
    numbers = [*attack, *defence, data["home_adv"], data["rho"]]
    if not all(isinstance(x, int | float) and math.isfinite(x) for x in numbers):
        raise ValueError("the model file contains a value that is not a finite number")
    model = DixonColes(
        teams,
        np.array(attack, dtype=float),
        np.array(defence, dtype=float),
        float(data["home_adv"]),
        float(data["rho"]),
        int(data["max_goals"]),
    )
    return model, data.get("meta", {})


def save_model(path: Path, model: DixonColes, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_to_dict(model, meta), indent=1), encoding="utf-8")


def load_model(path: Path) -> tuple[DixonColes, dict]:
    return dict_to_model(json.loads(path.read_text(encoding="utf-8")))
