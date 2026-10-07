"""Split Wikipedia articles into paragraphs and store them in nlp.document."""

import json
import logging
import re

import pandas as pd
from sqlalchemy import text

from data.common.config import load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging

log = logging.getLogger("nlp.corpus")
HEADING = re.compile(r"^(=+)\s*(.+?)\s*\1$")
SKIP_SECTIONS = {
    "References",
    "External links",
    "See also",
    "Notes",
    "Bibliography",
    "Further reading",
    "Footnotes",
}


def paragraphs(article: dict, min_chars: int) -> list[dict]:
    rows, section, idx = [], "Introduction", 0
    for line in article["text"].split("\n"):
        line = line.strip()
        if not line:
            continue
        heading = HEADING.match(line)
        if heading:
            section = heading.group(2)
            continue
        if section in SKIP_SECTIONS or len(line) < min_chars:
            continue
        rows.append(
            {
                "doc_id": f"{article['revid']}-{idx}",
                "article_title": article["title"],
                "revid": article["revid"],
                "url": article["url"],
                "season": article["season"],
                "team": article["team"],
                "section": section,
                "para_index": idx,
                "text": line,
                "n_chars": len(line),
            }
        )
        idx += 1
    return rows


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    files = sorted((s.raw / "wikipedia").glob("*.json"))
    rows = []
    for f in files:
        rows.extend(
            paragraphs(json.loads(f.read_text(encoding="utf-8")), s.nlp["min_paragraph_chars"])
        )
    docs = pd.DataFrame(rows)
    engine = make_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS nlp"))
        docs.to_sql("document", conn, schema="nlp", if_exists="replace", index=False)
        conn.execute(text("ALTER TABLE nlp.document ADD PRIMARY KEY (doc_id)"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA nlp TO {s.reader_role}"))
        conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA nlp TO {s.reader_role}"))
    log.info("%d paragraphs from %d articles", len(docs), len(files))


if __name__ == "__main__":
    main()
