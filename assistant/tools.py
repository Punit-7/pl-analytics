"""Tools into P2: team ratings, a match prediction and the stored season simulation."""

from __future__ import annotations

import difflib
import logging
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from data.common.config import Settings
from data.common.seasons import current_season_start, season_label
from modelling.match.data import load_fixtures, load_results
from modelling.match.dixon_coles import DixonColes, fit_dixon_coles

log = logging.getLogger(__name__)


class ToolError(ValueError):
    """A problem the model can fix by calling the tool again with other arguments."""


def clean_name(name: str) -> str:
    words = [w for w in name.casefold().replace(".", "").split() if w not in ("fc", "afc")]
    return " ".join(words)


def resolve_team(name: str, lookup: dict[str, str]) -> str | None:
    """Map what the user wrote to the warehouse team name, or None if nothing is close."""
    key = clean_name(name or "")
    if key in lookup:
        return lookup[key]
    close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=0.8)
    return lookup[close[0]] if close else None


def most_likely_score(model: DixonColes, home: str, away: str) -> tuple[str, float]:
    matrix = model.score_matrix(home, away)
    h, a = divmod(int(matrix.argmax()), matrix.shape[1])
    return f"{h}-{a}", float(matrix[h, a])


class MatchTools:
    def __init__(self, engine: Engine, settings: Settings):
        self.engine, self.s = engine, settings
        self.season = season_label(current_season_start(start_month=settings.season_start_month))
        self._model: DixonColes | None = None
        self._fitted_on: date | None = None
        teams = pd.read_sql(text("SELECT DISTINCT team FROM mart.fact_team_match"), engine)["team"]
        self.lookup = {clean_name(t): t for t in teams}
        aliases = settings.reference / "team_aliases.csv"  # written in P3 Stage 3
        if aliases.exists():
            for row in pd.read_csv(aliases).itertuples():
                if row.team in set(teams):
                    self.lookup.setdefault(clean_name(row.alias), row.team)

    def team(self, name: str) -> str:
        found = resolve_team(name, self.lookup)
        if found is None:
            raise ToolError(f"Unknown team '{name}'. Use a Premier League team name.")
        return found

    def model(self) -> DixonColes:
        """Fit P2's Dixon-Coles model once per day and reuse it."""
        today = date.today()
        if self._model is None or self._fitted_on != today:
            m = self.s.modelling
            self._model = fit_dixon_coles(
                load_results(self.engine),
                today,
                m["dc_xi"],
                m["dc_history_days"],
                m["dc_max_goals"],
            )
            self._fitted_on = today
            log.info("Fitted Dixon-Coles on results before %s", today)
        return self._model

    def team_ratings(self, team: str = "") -> dict:
        """Attack, defence and strength of this season's teams, strongest first."""
        model = self.model()
        fixtures = load_fixtures(self.engine, self.season)
        rows = []
        for t in sorted(set(fixtures["home_team"]) | set(fixtures["away_team"])):
            a, d = model.rating(t)
            rows.append(
                {
                    "team": t,
                    "attack": round(a, 3),
                    "defence": round(d, 3),
                    "strength": round(a - d, 3),
                }
            )
        rows.sort(key=lambda r: -r["strength"])
        for rank, r in enumerate(rows, start=1):
            r["rank"] = rank
        if team:
            wanted = self.team(team)
            rows = [r for r in rows if r["team"] == wanted]
            if not rows:
                raise ToolError(f"{wanted} is not in the {self.season} Premier League.")
        return {
            "season": self.season,
            "fitted_on": str(self._fitted_on),
            "note": "A higher attack and a lower defence are better. strength = attack - defence.",
            "ratings": rows,
        }

    def predict_match(self, home_team: str, away_team: str) -> dict:
        """Result probabilities for one match from P2's Dixon-Coles model."""
        home, away = self.team(home_team), self.team(away_team)
        if home == away:
            raise ToolError("The two teams must be different.")
        model = self.model()
        p_home, p_draw, p_away = model.outcome_probs(home, away)
        lam, mu = model.expected_goals(home, away)
        score, p_score = most_likely_score(model, home, away)
        return {
            "home_team": home,
            "away_team": away,
            "p_home_win": round(p_home, 3),
            "p_draw": round(p_draw, 3),
            "p_away_win": round(p_away, 3),
            "expected_home_goals": round(lam, 2),
            "expected_away_goals": round(mu, 2),
            "most_likely_score": score,
            "most_likely_score_probability": round(p_score, 3),
            "model": self.s.modelling["model_version"],
            "fitted_on": str(self._fitted_on),
        }

    def season_outlook(self, team: str = "") -> dict:
        """Title, top-four and relegation probabilities from the latest stored simulation."""
        query = text("""
            SELECT created_at, model_version, team, exp_points, p_title, p_top4, p_relegation
            FROM predictions.season_sim
            WHERE season = :season AND created_at = (
                SELECT MAX(created_at) FROM predictions.season_sim WHERE season = :season)
            ORDER BY exp_points DESC""")
        try:
            df = pd.read_sql(query, self.engine, params={"season": self.season})
        except Exception as e:  # the table does not exist until P2's predict script has run
            raise ToolError("No season simulation is stored. Run P2's predict script.") from e
        if df.empty:
            raise ToolError("No season simulation is stored. Run P2's predict script.")
        made, version = str(df["created_at"].iloc[0]), df["model_version"].iloc[0]
        df = df.drop(columns=["created_at", "model_version"]).round(3)
        df.insert(0, "expected_rank", range(1, len(df) + 1))
        if team:
            df = df[df["team"] == self.team(team)]
        return {
            "season": self.season,
            "simulated_at": made,
            "model": version,
            "teams": df.to_dict(orient="records"),
        }
