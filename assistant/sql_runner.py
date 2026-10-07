"""Run model-written SQL safely: one SELECT, read-only role, time limit, row limit."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from sqlalchemy.engine import Engine

COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


class SQLRejected(ValueError):
    """The text was refused before it reached the database."""


class SQLFailed(RuntimeError):
    """The database refused or could not run the query."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    seconds: float


def check_sql(sql: str) -> str:
    """Return one cleaned SELECT statement, or raise SQLRejected."""
    cleaned = COMMENTS.sub(" ", sql or "").strip().rstrip(";").strip()
    if not cleaned:
        raise SQLRejected("The query is empty")
    if ";" in cleaned:
        raise SQLRejected("Only one statement is allowed")
    first_word = cleaned.split(None, 1)[0].lower()
    if first_word not in ("select", "with"):
        raise SQLRejected("Only SELECT queries are allowed")
    return cleaned


def run_sql(engine: Engine, sql: str, timeout_ms: int, max_rows: int) -> QueryResult:
    """Run one SELECT inside a read-only transaction and return at most max_rows rows."""
    cleaned = check_sql(sql)
    start = time.perf_counter()
    raw = engine.raw_connection()  # the database driver's own connection
    try:
        with raw.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
            cur.execute(cleaned)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(max_rows + 1)
    except Exception as e:  # any database error: syntax, missing column, timeout, write attempt
        raise SQLFailed(str(e).strip().splitlines()[0]) from e
    finally:
        raw.rollback()
        raw.close()
    return QueryResult(
        columns=columns,
        rows=[tuple(r) for r in rows[:max_rows]],
        truncated=len(rows) > max_rows,
        seconds=time.perf_counter() - start,
    )