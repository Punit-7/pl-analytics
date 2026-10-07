"""Download Premier League club season articles from English Wikipedia (CC BY-SA 4.0)."""
import json
import logging
import time
from datetime import UTC, datetime

import pandas as pd
import requests
from sqlalchemy import text

from data.common.config import Settings
from data.common.db import make_engine
from data.common.io_utils import write_atomic

log = logging.getLogger(__name__)
API = "https://en.wikipedia.org/w/api.php"


def season_title(season: str, wiki_name: str) -> str:
    """'2024/25' + 'Arsenal F.C.' -> '2024–25 Arsenal F.C. season' (with an en dash)."""
    return f"{season[:4]}\u2013{season[5:7]} {wiki_name} season"


def club_seasons(engine, first_season: int) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT DISTINCT season, team FROM mart.fact_team_match
        WHERE CAST(LEFT(season, 4) AS INTEGER) >= :first ORDER BY season, team"""),
        engine, params={"first": first_season})


def fetch(session: requests.Session, title: str) -> dict | None:
    params = {"action": "query", "prop": "extracts|revisions", "explaintext": 1,
              "rvprop": "ids", "titles": title, "redirects": 1,
              "format": "json", "formatversion": 2}
    r = session.get(API, params=params, timeout=30)
    r.raise_for_status()
    page = r.json()["query"]["pages"][0]
    if page.get("missing"):
        return None
    return {"title": page["title"], "revid": page["revisions"][0]["revid"],
            "text": page.get("extract", "")}


def run(s: Settings) -> dict:
    out = s.raw / "wikipedia"
    out.mkdir(parents=True, exist_ok=True)
    clubs = pd.read_csv(s.reference / "wiki_clubs.csv")
    wiki = dict(zip(clubs["team"], clubs["wiki_name"], strict=True))
    session = requests.Session()
    session.headers["User-Agent"] = s.nlp["wiki_user_agent"]
    counts = {"saved": 0, "missing": 0, "skipped": 0, "no_mapping": 0}

    for row in club_seasons(make_engine(), s.nlp["first_season"]).itertuples():
        name = wiki.get(row.team)
        if name is None:
            log.warning("No wiki_clubs.csv row for %s", row.team)
            counts["no_mapping"] += 1
            continue
        path = out / f"{row.season.replace('/', '-')}_{row.team.replace(' ', '_')}.json"
        if path.exists():
            counts["skipped"] += 1
            continue
        title = season_title(row.season, name)
        page = fetch(session, title)
        time.sleep(s.nlp["wiki_delay_seconds"])  # one request per second
        if page is None:
            log.info("No article: %s", title)
            counts["missing"] += 1
            continue
        page.update(season=row.season, team=row.team, licence="CC BY-SA 4.0",
                    url=f"https://en.wikipedia.org/w/index.php?oldid={page['revid']}",
                    fetched_at=datetime.now(UTC).isoformat())
        write_atomic(path, json.dumps(page, ensure_ascii=False).encode("utf-8"))
        counts["saved"] += 1
    log.info("Wikipedia: %s", counts)
    return counts


if __name__ == "__main__":
    from data.common.config import load_settings
    from data.common.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings.logs)
    run(settings)