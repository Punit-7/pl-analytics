"""Write nlp/data/SOURCES.csv: every Wikipedia revision used, with its licence."""

import pandas as pd
from sqlalchemy import text

from data.common.config import ROOT
from data.common.db import make_engine

sources = pd.read_sql(
    text("""
    SELECT DISTINCT article_title, revid, url FROM nlp.document ORDER BY article_title"""),
    make_engine(),
)
sources["licence"] = "CC BY-SA 4.0"
sources.to_csv(ROOT / "nlp" / "data" / "SOURCES.csv", index=False, encoding="utf-8")
print(len(sources), "sources written")
