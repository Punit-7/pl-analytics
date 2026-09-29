from datetime import date

import pytest

from data.common.seasons import (
    current_season_start,
    season_code,
    season_label,
    start_year_from_code,
)


@pytest.mark.parametrize("year, code", [(1993, "9394"), (1999, "9900"), (2026, "2627")])
def test_season_code_round_trip(year, code):
    assert season_code(year) == code
    assert start_year_from_code(code) == year


def test_season_label():
    assert season_label(2026) == "2026/27"
    assert season_label(1999) == "1999/00"


@pytest.mark.parametrize(
    "today, expected",
    [
        (date(2026, 9, 28), 2026),
        (date(2027, 3, 10), 2026),
        (date(2026, 7, 31), 2025),
        (date(2026, 8, 1), 2026),
    ],
)
def test_current_season_start(today, expected):
    assert current_season_start(today) == expected
