"""P1 data pipeline: ingest -> build staging -> test -> publish -> export."""

import argparse
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.engine import Engine

from data import build_warehouse
from data.common.config import Settings, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from data.ingest import football_data, understat

log = logging.getLogger("data.pipeline")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 data pipeline")
    p.add_argument(
        "--full-refresh",
        action="store_true",
        help="re-download every season, not only the current one",
    )
    p.add_argument(
        "--skip-ingest", action="store_true", help="rebuild from the raw files already on disk"
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def run_tests(s: Settings) -> None:
    env = {**os.environ, "PL_TEST_SCHEMA": s.staging_schema}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, env=env
    )
    summary = result.stdout.strip().splitlines()[-1] if result.stdout else "no output"
    if result.returncode != 0:
        log.error("Test output:\n%s", result.stdout)
        raise RuntimeError(f"Tests failed ({summary}); nothing published")
    log.info("Tests: %s", summary)


def record_run(
    s: Settings,
    engine: Engine,
    run_id: str,
    started: datetime,
    status: str,
    error: str | None,
    counts: dict,
) -> None:
    ops = s.ops_schema
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {ops}"))
        conn.execute(
            text(f"""
            CREATE TABLE IF NOT EXISTS {ops}.pipeline_runs (
                run_id TEXT PRIMARY KEY, started_at TIMESTAMP, finished_at TIMESTAMP,
                status TEXT, error TEXT, fact_match_rows BIGINT)""")
        )
        conn.execute(
            text(
                f"INSERT INTO {ops}.pipeline_runs VALUES "
                "(:run_id, :started, :finished, :status, :error, :rows)"
            ),
            {
                "run_id": run_id,
                "started": started,
                "finished": utc_now(),
                "status": status,
                "error": error,
                "rows": counts.get("fact_match"),
            },
        )
        conn.execute(text(f"GRANT USAGE ON SCHEMA {ops} TO {s.reader_role}"))
        conn.execute(text(f"GRANT SELECT ON {ops}.pipeline_runs TO {s.reader_role}"))


def main(argv=None) -> int:
    args = parse_args(argv)
    s = load_settings()
    setup_logging(s.logs, args.log_level)
    run_id = uuid.uuid4().hex[:8]
    started, t0 = utc_now(), time.perf_counter()
    status, error, counts, engine = "failed", None, {}, None
    log.info(
        "Run %s started (full_refresh=%s, skip_ingest=%s)",
        run_id,
        args.full_refresh,
        args.skip_ingest,
    )
    try:
        engine = make_engine()
        if not args.skip_ingest:
            football_data.run(s, full_refresh=args.full_refresh)
            understat.run(s)
        counts = build_warehouse.build(s, engine)
        run_tests(s)
        build_warehouse.publish(s, engine)
        build_warehouse.export_marts(s, engine)
        status = "success"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.exception("Run %s failed", run_id)
    finally:
        if engine is not None:
            try:
                record_run(s, engine, run_id, started, status, error, counts)
                build_warehouse.export_table(s, engine, s.ops_schema, "pipeline_runs")
            except Exception:
                log.exception("Could not record run %s", run_id)
            engine.dispose()
        log.info("Run %s finished: %s in %.1f s", run_id, status, time.perf_counter() - t0)
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
