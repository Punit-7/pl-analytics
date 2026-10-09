"""Match results for the weekly job, read from the downloaded CSV files. No database."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date

import pandas as pd

from data.build_warehouse import load_matches
from data.common.config import Settings
from data.common.io_utils import DataValidationError
from data.common.seasons import current_season_start, season_label
from data.ingest import football_data

log = logging.getLogger(__name__)
NEEDED = ["season", "match_date", "home_team", "away_team", "home_goals", "away_goals"]
ODDS = ["avg_close_home_odds", "avg_close_draw_odds", "avg_close_away_odds"]
OFF_SEASON_MONTHS = (6, 7)  # no league matches, so old results are normal


def current_season(s: Settings, today: date | None = None) -> str:
    return season_label(current_season_start(today, s.season_start_month))


def load_results(s: Settings, ingest: bool = True, today: date | None = None) -> pd.DataFrame:
    """Download the recent season files (optional), then load every file on disk."""
    if ingest:
        first = current_season_start(today, s.season_start_month) - s.serving["history_seasons"]
        football_data.run(replace(s, first_season=first))
    m = load_matches(s).reindex(columns=NEEDED + ODDS)
    m["match_date"] = pd.to_datetime(m["match_date"])
    return m.sort_values(["match_date", "home_team"]).reset_index(drop=True)


def validate(results: pd.DataFrame, s: Settings, today: date | None = None) -> dict:
    """Stop the job on broken data. Return facts about the data for the run report."""
    today = today or date.today()
    season = current_season(s, today)
    problems = []
    if results.empty:
        raise DataValidationError("no matches loaded")
    if results[NEEDED].isna().any().any():
        problems.append("missing values in a required column")
    else:
        goals = results[["home_goals", "away_goals"]]
        if ((goals % 1 != 0) | (goals < 0) | (goals > 15)).any().any():
            problems.append("goals that are not whole numbers from 0 to 15")
    if results.duplicated(["season", "home_team", "away_team"]).any():
        problems.append("the same fixture appears twice in one season")
    if (results["match_date"].dt.date > today).any():
        problems.append("matches dated in the future")
    per_season = results.groupby("season").size()
    finished = per_season[(per_season.index != season) & (per_season.index >= "1995/96")]
    if (finished != 380).any():
        problems.append(
            f"finished seasons without 380 matches: {list(finished[finished != 380].index)}"
        )
    if per_season.get(season, 0) > 380:
        problems.append("more than 380 matches in the current season")
    if problems:
        raise DataValidationError("; ".join(problems))

    latest = results["match_date"].max().date()
    age = (today - latest).days
    stale = age > s.serving["freshness_days"] and today.month not in OFF_SEASON_MONTHS
    if stale:
        log.warning("Latest result is %d days old (%s)", age, latest)
    return {
        "season": season,
        "n_matches": int(len(results)),
        "n_current_season": int(per_season.get(season, 0)),
        "latest_match_date": str(latest),
        "days_since_latest": int(age),
        "stale": bool(stale),
    }
