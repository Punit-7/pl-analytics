import logging
import time
from datetime import date
from io import BytesIO

import pandas as pd

from data.common.config import Settings
from data.common.http import make_session
from data.common.io_utils import DataValidationError, write_atomic
from data.common.seasons import current_season_start, season_code

log = logging.getLogger(__name__)



log = logging.getLogger(__name__)

BASE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
REQUIRED = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}

def validate(content: bytes, code: str) -> int:
    """Check the download is a results CSV. Returns the number of matches."""
    try:
        df = pd.read_csv(BytesIO(content), encoding="latin-1", on_bad_lines="skip")
    except Exception as exc:
        raise DataValidationError(f"{code}: file is not a readable CSV") from exc
    missing = REQUIRED - set(df.columns)
    if missing:
        raise DataValidationError(f"{code}: missing columns {sorted(missing)}")
    return int(df["HomeTeam"].notna().sum())

def run(s: Settings, full_refresh: bool = False) -> dict:
    out_dir = s.raw / "football_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    current = current_season_start(start_month=s.season_start_month)
    session = make_session(s)
    stats = {"downloaded": 0, "skipped": 0}

    for year in range(s.first_season, current + 1):
        code = season_code(year)
        out = out_dir / f"E0_{code}.csv"
        if out.exists() and year < current and not full_refresh:
            stats["skipped"] += 1          # finished seasons do not change
            continue

        url = BASE.format(code=code)
        log.debug("GET %s", url)
        r = session.get(url, timeout=s.timeout)
        if r.status_code == 404 and year == current:
            log.warning("No file yet for current season %s; skipping", code)
            continue
        r.raise_for_status()

        matches = validate(r.content, code)
        write_atomic(out, r.content)
        if year == current:                # keep a dated copy of each weekly download
            snap = out_dir / "snapshots" / f"E0_{code}_{date.today():%Y-%m-%d}.csv"
            snap.parent.mkdir(exist_ok=True)
            write_atomic(snap, r.content)
        log.info("Saved %s (%d matches)", out.name, matches)
        stats["downloaded"] += 1
        time.sleep(s.delay)                # be polite to the site

    log.info("football-data: %d downloaded, %d skipped",
             stats["downloaded"], stats["skipped"])
    return stats

if __name__ == "__main__":
    from data.common.config import load_settings
    from data.common.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings.logs)
    run(settings)