"""The shapes of the API's requests and replies. FastAPI checks every request against them."""

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    home_team: str = Field(min_length=2, max_length=40, examples=["Arsenal"])
    away_team: str = Field(min_length=2, max_length=40, examples=["Chelsea"])


class MatchPrediction(BaseModel):
    home_team: str
    away_team: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    expected_home_goals: float
    expected_away_goals: float
    most_likely_score: str
    most_likely_score_probability: float
    model_version: str
    fitted_on: str


class Shot(BaseModel):
    """One non-penalty shot, in StatsBomb pitch coordinates: the goal is at x = 120, y = 40."""

    x: float = Field(ge=0, le=120, examples=[108])
    y: float = Field(ge=0, le=80, examples=[36])
    body_part: str = "Right Foot"
    shot_type: str = "Open Play"
    play_pattern: str = "Regular Play"
    technique: str = "Normal"
    under_pressure: bool = False
    first_time: bool = False


class XgPrediction(BaseModel):
    xg: float
    distance: float
    angle_degrees: float
    model: str


class TeamRating(BaseModel):
    rank: int
    team: str
    attack: float
    defence: float
    strength: float


class Health(BaseModel):
    status: str
    service_version: str
    model_version: str
    fitted_on: str
    trained_through: str
    teams: int
    xg_model_loaded: bool
    predictions_published_at: str | None