from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from data.common.config import load_settings
from data.common.io_utils import DataValidationError
from serving.data import validate
from serving.sample_data import league

S = load_settings()
TODAY = date(2026, 5, 1)


def test_clean_data_passes_and_reports_freshness():
    stats = validate(league(), S, TODAY)
    assert stats["n_matches"] == 3 * 380 and stats["season"] == "2025/26"
    assert stats["days_since_latest"] >= 0 and stats["stale"] is False


def test_old_data_is_flagged_as_stale_but_not_refused():
    results = league(last_day=date(2026, 3, 1))
    assert validate(results, S, TODAY)["stale"] is True


@pytest.mark.parametrize(
    "damage",
    [
        lambda df: pd.concat([df, df.tail(1)]),  # the same fixture twice
        lambda df: df.assign(home_goals=df["home_goals"].where(df.index != 5, -1)),
        lambda df: df.assign(
            match_date=df["match_date"].where(df.index != 5, pd.Timestamp(2030, 1, 1))
        ),
        lambda df: df.assign(home_team=df["home_team"].where(df.index != 5)),  # a missing name
        lambda df: df.drop(index=7),  # a finished season with 379 matches
    ],
)
def test_broken_data_stops_the_job(damage):
    with pytest.raises(DataValidationError):
        validate(damage(league()), S, TODAY)


def test_settings_can_be_narrowed_to_recent_seasons():
    assert replace(S, first_season=2021).first_season == 2021
