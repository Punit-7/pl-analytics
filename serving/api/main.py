"""The web API. Run locally: uvicorn serving.api.main:app --reload --port 10000"""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from serving.api.schemas import (
    Health,
    MatchPrediction,
    MatchRequest,
    Shot,
    TeamRating,
    XgPrediction,
)
from serving.api.service import SERVICE_VERSION, ModelService, UnknownTeam

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("serving.api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["service"] = ModelService()  # load the model files once, when the server starts
    yield
    state.clear()


app = FastAPI(
    title="Premier League match model",
    description="Dixon-Coles match predictions, refitted every week, and an xG model. "
    "A portfolio project. Results data: football-data.co.uk. xG training data: StatsBomb.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One JSON log line per request: what was asked, the status, and how long it took."""
    start = time.perf_counter()
    response = await call_next(request)
    log.info(
        json.dumps(
            {
                "path": request.url.path,
                "status": response.status_code,
                "ms": round(1000 * (time.perf_counter() - start), 1),
                "model_version": state["service"].meta.get("model_version") if state else None,
            }
        )
    )
    return response


def service() -> ModelService:
    return state["service"]


def team_not_found(e: UnknownTeam) -> HTTPException:
    detail = {"message": f"Unknown team '{e}'", "teams": sorted(service().model.teams)}
    return HTTPException(status_code=404, detail=detail)


@app.get("/")
def index() -> dict:
    return {"service": "Premier League match model", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=Health)
def health() -> dict:
    return service().health()


@app.get("/teams", response_model=list[TeamRating])
def teams() -> list[dict]:
    return service().ratings()


@app.post("/predict/match", response_model=MatchPrediction)
def predict_match(body: MatchRequest) -> dict:
    try:
        return service().predict_match(body.home_team, body.away_team)
    except UnknownTeam as e:
        raise team_not_found(e) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/predictions/latest")
def predictions(team: str | None = None) -> dict:
    try:
        return service().published(team)
    except UnknownTeam as e:
        raise team_not_found(e) from e


@app.get("/season")
def season() -> dict:
    return service().season()


@app.post("/predict/xg", response_model=XgPrediction)
def predict_xg(shot: Shot) -> dict:
    try:
        return service().predict_xg(shot.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
