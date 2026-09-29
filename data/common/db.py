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
