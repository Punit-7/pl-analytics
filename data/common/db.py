import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

from data.common.config import ROOT

REQUIRED = ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD")


def make_engine() -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL from the .env file."""
    load_dotenv(ROOT / ".env")
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing in .env: {missing}")
    url = URL.create(
        "postgresql+psycopg",
        username=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        host=os.environ["PGHOST"],
        port=int(os.environ["PGPORT"]),
        database=os.environ["PGDATABASE"],
    )
    return create_engine(url, pool_pre_ping=True)


def make_reader_engine() -> Engine:
    """Engine for the read-only role pl_reader. The assistant uses only this engine."""
    load_dotenv(ROOT / ".env")
    needed = ("PGHOST", "PGPORT", "PGDATABASE", "PGREADER_USER", "PGREADER_PASSWORD")
    missing = [k for k in needed if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing in .env: {missing}")
    url = URL.create(
        "postgresql+psycopg",
        username=os.environ["PGREADER_USER"],
        password=os.environ["PGREADER_PASSWORD"],
        host=os.environ["PGHOST"],
        port=int(os.environ["PGPORT"]),
        database=os.environ["PGDATABASE"],
    )
    # Every transaction on this engine starts read-only, whatever the code does later.
    options = "-c default_transaction_read_only=on"
    return create_engine(url, pool_pre_ping=True, connect_args={"options": options})
