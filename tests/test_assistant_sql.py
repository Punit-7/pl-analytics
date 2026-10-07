import time

import pytest

from assistant.sql_runner import SQLFailed, SQLRejected, check_sql, run_sql
from data.common.db import make_reader_engine


def test_check_sql_accepts_select_and_with():
    assert check_sql("SELECT 1;") == "SELECT 1"
    assert check_sql("-- note\nWITH a AS (SELECT 1) SELECT * FROM a").startswith("WITH")


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "DROP TABLE mart.fact_match",
        "UPDATE mart.fact_match SET home_goals = 9",
        "SELECT 1; DELETE FROM mart.fact_match",
        "/* hidden */ INSERT INTO mart.dim_team VALUES ('x')",
    ],
)
def test_check_sql_rejects_everything_else(sql):
    with pytest.raises(SQLRejected):
        check_sql(sql)


@pytest.fixture(scope="module")
def engine():
    try:
        e = make_reader_engine()
        run_sql(e, "SELECT 1", 2000, 1)
    except Exception as err:
        pytest.skip(f"no read-only database connection: {err}")
    return e


def test_select_returns_columns_and_rows(engine):
    r = run_sql(engine, "SELECT 1 AS one, 'a' AS letter", 2000, 10)
    assert r.columns == ["one", "letter"] and r.rows == [(1, "a")] and not r.truncated


def test_database_refuses_a_write_hidden_in_a_select(engine):
    sql = "WITH gone AS (DELETE FROM mart.fact_match RETURNING 1) SELECT COUNT(*) FROM gone"
    with pytest.raises(SQLFailed):
        run_sql(engine, sql, 2000, 10)
    assert run_sql(engine, "SELECT COUNT(*) FROM mart.fact_match", 2000, 1).rows[0][0] > 0


def test_slow_query_is_cancelled(engine):
    start = time.perf_counter()
    with pytest.raises(SQLFailed):
        run_sql(engine, "SELECT pg_sleep(5)", 300, 10)
    assert time.perf_counter() - start < 3


def test_rows_are_capped(engine):
    r = run_sql(engine, "SELECT generate_series(1, 50)", 2000, 10)
    assert len(r.rows) == 10 and r.truncated