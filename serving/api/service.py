"""Everything the API computes. No web code here, so it can be tested without a server."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from modelling.xg.features import CATEGORICAL, build_matrix
from serving.model_io import load_model

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "serving" / "models" / "match_model.json"
XG = ROOT / "serving" / "models" / "xg_logistic.joblib"
LATEST = ROOT / "serving" / "predictions" / "latest.json"
SERVICE_VERSION = "api-1.0"


class UnknownTeam(LookupError):
    """The name is not a team in the fitted model."""


class ModelService:
    def __init__(self, model_path: Path = MODEL, latest_path: Path = LATEST, xg_path: Path = XG):
        self.model, self.meta = load_model(model_path)
        self.names = {t.casefold(): t for t in self.model.teams}
        self.latest = {"created_at": None, "matches": [], "season_sim": []}
        if latest_path.exists():
            self.latest = json.loads(latest_path.read_text(encoding="utf-8"))
        self.xg_model, self.xg_columns, self.xg_values = None, [], {}
        if xg_path.exists():
            import joblib  # only load a model file that you created yourself

            self.xg_model = joblib.load(xg_path)
            self.xg_columns = list(self.xg_model.feature_names_in_)
            for field in CATEGORICAL:  # the category values the model saw in training
                prefix = f"{field}_"
                self.xg_values[field] = sorted(
                    c.removeprefix(prefix) for c in self.xg_columns if c.startswith(prefix)
                )

    def team(self, name: str) -> str:
        found = self.names.get(name.strip().casefold())
        if found is None:
            raise UnknownTeam(name)
        return found

    def health(self) -> dict:
        return {
            "status": "ok",
            "service_version": SERVICE_VERSION,
            "model_version": self.meta.get("model_version", "unknown"),
            "fitted_on": self.meta.get("fitted_on", "unknown"),
            "trained_through": self.meta.get("trained_through", "unknown"),
            "teams": len(self.model.teams),
            "xg_model_loaded": self.xg_model is not None,
            "predictions_published_at": self.latest.get("created_at"),
        }

    def ratings(self) -> list[dict]:
        current = {t["team"] for t in self.latest["season_sim"]}  # this season's 20 teams
        rows = []
        for t in self.model.teams:
            if current and t not in current:
                continue  # a team from an earlier season that is still in the three-year fit
            a, d = self.model.rating(t)
            rows.append({"team": t, "attack": round(a, 3), "defence": round(d, 3),
                         "strength": round(a - d, 3)})  # fmt: skip
        rows.sort(key=lambda r: -r["strength"])
        return [{"rank": i, **r} for i, r in enumerate(rows, start=1)]

    def predict_match(self, home_team: str, away_team: str) -> dict:
        home, away = self.team(home_team), self.team(away_team)
        if home == away:
            raise ValueError("home_team and away_team must be different")
        matrix = self.model.score_matrix(home, away)
        p_home, p_draw, p_away = self.model.outcome_probs(home, away)
        lam, mu = self.model.expected_goals(home, away)
        h, a = divmod(int(matrix.argmax()), matrix.shape[1])
        return {
            "home_team": home,
            "away_team": away,
            "p_home_win": round(p_home, 4),
            "p_draw": round(p_draw, 4),
            "p_away_win": round(p_away, 4),
            "expected_home_goals": round(lam, 3),
            "expected_away_goals": round(mu, 3),
            "most_likely_score": f"{h}-{a}",
            "most_likely_score_probability": round(float(matrix[h, a]), 4),
            "model_version": self.meta.get("model_version", "unknown"),
            "fitted_on": self.meta.get("fitted_on", "unknown"),
        }

    def published(self, team: str | None = None) -> dict:
        """The predictions the weekly job published, for all teams or for one."""
        matches = self.latest["matches"]
        if team:
            wanted = self.team(team)
            matches = [m for m in matches if wanted in (m["home_team"], m["away_team"])]
        return {**{k: v for k, v in self.latest.items() if k != "season_sim"}, "matches": matches}

    def season(self) -> dict:
        return {k: v for k, v in self.latest.items() if k != "matches"}

    def predict_xg(self, shot: dict) -> dict:
        if self.xg_model is None:
            raise RuntimeError("the xG model file is not in this image")
        for field, allowed in self.xg_values.items():
            if shot[field] not in allowed:
                raise ValueError(f"{field} must be one of {allowed}")
        X = build_matrix(pd.DataFrame([shot]), columns=self.xg_columns)  # the training code
        p = float(self.xg_model.predict_proba(X)[0, 1])
        return {
            "xg": round(p, 4),
            "distance": round(float(X["distance"].iloc[0]), 2),
            "angle_degrees": round(math.degrees(float(X["angle"].iloc[0])), 1),
            "model": "xg_logistic",
        }
