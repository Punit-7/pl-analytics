import logging
from datetime import date

import soccerdata as sd

from data.common.config import Settings
from data.common.io_utils import DataValidationError, write_atomic
from data.common.seasons import current_season_start

log = logging.getLogger(__name__)

KEEP = ["season", "date", "home_team", "away_team",
        "home_goals", "away_goals", "home_xg", "away_xg", "is_result"]

def run(s: Settings) -> dict:
    if not s.understat_enabled:
        log.warning("Understat disabled in config.toml; skipping")
        return {"rows": 0}

    out_dir = s.raw / "understat"
    out_dir.mkdir(parents=True, exist_ok=True)
    current = current_season_start(start_month=s.season_start_month)
    seasons = [f"{y}/{y + 1}" for y in range(s.understat_first, current + 1)]
    log.info("Reading Understat schedule for %d seasons", len(seasons))

    us = sd.Understat(leagues="ENG-Premier League", seasons=seasons)
    schedule = us.read_schedule().reset_index()

    missing = set(KEEP) - set(schedule.columns)
    if missing:
        raise DataValidationError(f"Understat: missing columns {sorted(missing)}")

    csv = schedule[KEEP].to_csv(index=False).encode("utf-8")
    write_atomic(out_dir / "schedule.csv", csv)
    snap = out_dir / "snapshots" / f"schedule_{date.today():%Y-%m-%d}.csv"
    snap.parent.mkdir(exist_ok=True)
    write_atomic(snap, csv)                # dated copy for reproducibility
    played = int(schedule["is_result"].sum())
    log.info("Saved %d Understat fixtures (%d played)", len(schedule), played)
    return {"rows": len(schedule)}


if __name__ == "__main__":
    from data.common.config import load_settings
    from data.common.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings.logs)
    run(settings)