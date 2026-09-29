from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config.toml"

@dataclass
class Settings:
    first_season: int
    understat_first: int
    season_start_month: int
    raw: Path
    reference: Path
    warehouse: Path
    marts: Path
    logs: Path
    timeout: int
    max_retries: int
    backoff_factor: float
    delay: float
    user_agent: str
    understat_enabled: bool
    
def load_settings(path: Path = CONFIG_FILE) -> Settings:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    seasons, paths, http = c["seasons"], c["paths"], c["http"]
    return Settings(
        first_season=seasons["first"],
        understat_first=seasons["understat_first"],
        season_start_month=seasons["season_start_month"],
        raw=ROOT / paths["raw"],
        reference=ROOT / paths["reference"],
        warehouse=ROOT / paths["warehouse"],
        marts=ROOT / paths["marts"],
        logs=ROOT / paths["logs"],
        timeout=http["timeout_seconds"],
        max_retries=http["max_retries"],
        backoff_factor=http["backoff_factor"],
        delay=http["delay_seconds"],
        user_agent=http["user_agent"],
        understat_enabled=c["sources"]["understat_enabled"],
    )