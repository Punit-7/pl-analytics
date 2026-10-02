"""FPL API: players, prices, game rules and fixtures. Run before each squad optimisation."""

import json
import logging
import time

from data.common.config import Settings
from data.common.http import make_session
from data.common.io_utils import DataValidationError, write_atomic

log = logging.getLogger(__name__)

BASE = "https://fantasy.premierleague.com/api/{name}/"
# file name -> keys that must be present (in the object, or in the first item of a list)
REQUIRED = {
    "bootstrap-static": {"elements", "teams", "events", "element_types", "game_settings"},
    "fixtures": {"event", "team_h", "team_a", "finished"},
}


def validate(content: bytes, name: str) -> int:
    """Check the download is the expected JSON. Returns the number of top-level items."""
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise DataValidationError(f"{name}: not readable JSON") from exc
    first = data[0] if isinstance(data, list) and data else data
    missing = REQUIRED[name] - set(first)
    if missing:
        raise DataValidationError(f"{name}: missing keys {sorted(missing)}")
    return len(data)


def run(s: Settings) -> dict:
    out_dir = s.raw / "fpl"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(s)
    for name in REQUIRED:
        r = session.get(BASE.format(name=name), timeout=s.timeout)
        r.raise_for_status()
        validate(r.content, name)
        write_atomic(out_dir / f"{name}.json", r.content)
        log.info("Saved %s.json (%d bytes)", name, len(r.content))
        time.sleep(s.delay)  # be polite to the site
    return {"files": len(REQUIRED)}


if __name__ == "__main__":
    from data.common.config import load_settings
    from data.common.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings.logs)
    run(settings)
