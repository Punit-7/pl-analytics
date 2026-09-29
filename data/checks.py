import sys

import pandas as pd
from sqlalchemy import text

from data.common.db import make_engine

schema = sys.argv[1] if len(sys.argv) > 1 else "mart"
engine = make_engine()
print(
    pd.read_sql(
        text(f"""
    SELECT team, SUM(points) AS pts FROM {schema}.fact_team_match
    WHERE season = '2015/16' GROUP BY team ORDER BY pts DESC LIMIT 3"""),
        engine,
    )
)
print(
    pd.read_sql(
        text(f"""
    SELECT MIN(season) AS first_season_with_shots
    FROM {schema}.fact_match WHERE home_shots IS NOT NULL"""),
        engine,
    )
)
