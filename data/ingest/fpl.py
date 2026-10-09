"""FPL API: players, prices, rules, fixtures, and (optionally) your own team.

python -m data.ingest.fpl
"""

import json
import logging
import time

from data.common.config import Settings
from data.common.http import make_session
from data.common.io_utils import DataValidationError, write_atomic

log = logging.getLogger(__name__)
BASE = "https://fantasy.premierleague.com/api/{path}/"
# saved file name -> keys that must be present (in the object, or in the first item of a list)
REQUIRED = {
    "bootstrap-static": {"elements", "teams", "events", "element_types", "game_settings", "chips"},
    "fixtures": {"event", "team_h", "team_a", "finished"},
    "entry_history": {"current", "chips"},
    "entry_transfers": set(),  # an empty list is valid: no transfers made yet
    "entry_picks": {"picks", "entry_history", "active_chip"},
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


def last_played_event(boot: dict) -> int | None:
    """The latest gameweek whose deadline has passed, or None before the season starts."""
    started = [e["id"] for e in boot["events"] if e["is_current"] or e["finished"]]
    return max(started) if started else None


def run(s: Settings) -> dict:
    out_dir = s.raw / "fpl"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(s)

    def fetch(path: str, name: str) -> bytes:
        r = session.get(BASE.format(path=path), timeout=s.timeout)
        r.raise_for_status()
        validate(r.content, name)
        write_atomic(out_dir / f"{name}.json", r.content)
        log.info("Saved %s.json (%d bytes)", name, len(r.content))
        time.sleep(s.delay)  # be polite to the site
        return r.content

    boot = json.loads(fetch("bootstrap-static", "bootstrap-static"))
    fetch("fixtures", "fixtures")
    files = 2

    team_id = int(s.modelling.get("fpl", {}).get("team_id", 0))
    event = last_played_event(boot)
    if team_id and event:
        fetch(f"entry/{team_id}/history", "entry_history")
        fetch(f"entry/{team_id}/transfers", "entry_transfers")
        picks = json.loads(fetch(f"entry/{team_id}/event/{event}/picks", "entry_picks"))
        files += 3
        if picks["active_chip"] == "freehit" and event > 1:
            # A Free Hit squad lasts one week; the squad you own is the one before it.
            fetch(f"entry/{team_id}/event/{event - 1}/picks", "entry_picks")
            log.info("Gameweek %d was a Free Hit: using gameweek %d's squad", event, event - 1)
    elif team_id:
        log.info("The season has not started: no team to download yet")
    return {"files": files}


if __name__ == "__main__":
    from data.common.config import load_settings
    from data.common.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings.logs)
    run(settings)
