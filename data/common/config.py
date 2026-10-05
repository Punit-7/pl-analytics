from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
CONFIG_FILE = ROOT / "config.toml"


@dataclass(frozen=True)
class Settings:
    first_season: int
    understat_first: int
    season_start_month: int
    raw: Path
    reference: Path
    marts: Path
    logs: Path
    staging_schema: str
    mart_schema: str
    ops_schema: str
    reader_role: str
    timeout: int
    max_retries: int
    backoff_factor: float
    delay: float
    user_agent: str
    understat_enabled: bool
    modelling: dict
    nlp: dict


def load_settings(path: Path = CONFIG_FILE) -> Settings:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    seasons, paths, db, http = c["seasons"], c["paths"], c["database"], c["http"]
    return Settings(
        first_season=seasons["first"],
        understat_first=seasons["understat_first"],
        season_start_month=seasons["season_start_month"],
        raw=ROOT / paths["raw"],
        reference=ROOT / paths["reference"],
        marts=ROOT / paths["marts"],
        logs=ROOT / paths["logs"],
        staging_schema=db["staging_schema"],
        mart_schema=db["mart_schema"],
        ops_schema=db["ops_schema"],
        reader_role=db["reader_role"],
        timeout=http["timeout_seconds"],
        max_retries=http["max_retries"],
        backoff_factor=http["backoff_factor"],
        delay=http["delay_seconds"],
        user_agent=http["user_agent"],
        understat_enabled=c["sources"]["understat_enabled"],
        modelling=c["modelling"],
        nlp=c["nlp"],
    )
