import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from data.common.config import load_settings
from data.common.db import make_engine
from data.common.seasons import current_season_start, season_label

S = load_settings()
SCHEMA = os.getenv("PL_TEST_SCHEMA", S.mart_schema)  # the pipeline sets "staging"
CURRENT = season_label(current_season_start(start_month=S.season_start_month))
FM = f"{SCHEMA}.fact_match"
FTM = f"{SCHEMA}.fact_team_match"


@pytest.fixture(scope="module")
def con():
    engine = make_engine()
    try:
        c = engine.connect()
    except OperationalError as exc:
        pytest.fail(f"PostgreSQL not reachable; is the service running? {exc}")
    yield c
    c.close()
    engine.dispose()


def rows(con, sql):
    return con.execute(text(sql)).fetchall()


def test_finished_seasons_are_complete(con):
    # Every team plays every other team home and away: n teams -> n x (n - 1) matches.
    for season, matches, teams in rows(
        con,
        f"""
            SELECT season, COUNT(*), COUNT(DISTINCT home_team) FROM {FM}
            WHERE season <> '{CURRENT}' GROUP BY season""",
    ):
        assert matches == teams * (teams - 1), f"{season}: {matches} matches, {teams} teams"


def test_no_duplicate_fixtures(con):
    assert (
        rows(
            con,
            f"""
        SELECT season, home_team, away_team FROM {FM}
        GROUP BY season, home_team, away_team HAVING COUNT(*) > 1""",
        )
        == []
    )


def test_each_match_has_two_team_rows(con):
    assert (
        rows(
            con,
            f"""
        SELECT match_id FROM {FTM}
        GROUP BY match_id HAVING COUNT(*) <> 2""",
        )
        == []
    )


def test_points_add_up(con):
    # A win hands out 3 points in total; a draw hands out 2.
    assert (
        rows(
            con,
            f"""
        WITH a AS (
            SELECT season,
                   SUM(CASE WHEN home_goals = away_goals THEN 2 ELSE 3 END) AS expected
            FROM {FM} GROUP BY season),
        b AS (
            SELECT season, SUM(points) AS actual
            FROM {FTM} GROUP BY season)
        SELECT season, expected, actual FROM a JOIN b USING (season)
        WHERE expected <> actual""",
        )
        == []
    )


def test_leicester_2015_16(con):
    pts = rows(
        con,
        f"""
        SELECT SUM(points) FROM {FTM}
        WHERE season = '2015/16' AND team = 'Leicester'""",
    )[0][0]
    assert pts == 81


def test_xg_joined_for_understat_seasons(con):
    if not S.understat_enabled:
        pytest.skip("Understat disabled in config.toml")
    missing = rows(
        con,
        f"""
        SELECT season, home_team, away_team FROM {FM}
        WHERE CAST(LEFT(season, 4) AS INTEGER) >= {S.understat_first}
          AND home_xg IS NULL""",
    )
    assert missing == [], f"check team_names.csv; first gaps: {missing[:5]}"


def test_key_columns_not_null(con):
    assert (
        rows(
            con,
            f"""
        SELECT match_id FROM {FM}
        WHERE season IS NULL OR match_date IS NULL
           OR home_team IS NULL OR away_team IS NULL
           OR home_goals IS NULL OR away_goals IS NULL""",
        )
        == []
    )


def test_values_in_valid_ranges(con):
    # NULL stats (early seasons) make each comparison NULL, so they pass.
    bad = rows(
        con,
        f"""
        SELECT match_id, season, home_team, away_team FROM {FM}
        WHERE home_goals < 0 OR away_goals < 0
           OR home_goals > 15 OR away_goals > 15
           OR home_shots < 0 OR away_shots < 0
           OR home_sot > home_shots OR away_sot > away_shots
           OR home_xg < 0 OR away_xg < 0""",
    )
    assert bad == [], f"out-of-range values: {bad[:5]}"


def test_match_dates_inside_season(con):
    # A season runs from August of its start year to July of the next year.
    bad = rows(
        con,
        f"""
        SELECT match_id, season, match_date FROM {FM}
        WHERE match_date < make_date(CAST(LEFT(season, 4) AS INTEGER), 7, 1)
           OR match_date > make_date(CAST(LEFT(season, 4) AS INTEGER) + 1, 7, 31)""",
    )
    assert bad == [], f"dates outside their season: {bad[:5]}"

def test_odds_one_row_per_match(con):
    n_match = rows(con, f"SELECT COUNT(*) FROM {FM}")[0][0]
    n_odds = rows(con, f"SELECT COUNT(*) FROM {SCHEMA}.fact_match_odds")[0][0]
    assert n_odds == n_match


def test_closing_overround_plausible(con):
    bad = rows(con, f"""
        SELECT match_id FROM {SCHEMA}.fact_match_odds
        WHERE avg_close_home_odds IS NOT NULL
          AND 1 / avg_close_home_odds + 1 / avg_close_draw_odds + 1 / avg_close_away_odds
              NOT BETWEEN 0.98 AND 1.15""")
    assert bad == [], f"implausible overround: {bad[:5]}"
    
def test_sb_shots_valid(con):
    n = rows(con, f"SELECT COUNT(*) FROM {SCHEMA}.sb_shot")[0][0]
    if n == 0:
        pytest.skip("StatsBomb shots not downloaded")
    # Direct corner shots are logged just past the line (x = 120.2), so allow 1 unit of slack.
    bad = rows(con, f"""
        SELECT event_id FROM {SCHEMA}.sb_shot
        WHERE x NOT BETWEEN -1 AND 121 OR y NOT BETWEEN -1 AND 81
           OR statsbomb_xg NOT BETWEEN 0 AND 1 OR outcome IS NULL""")
    assert bad == [], f"invalid shots: {bad[:5]}"