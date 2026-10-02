import logging

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from data.common.config import Settings
from data.common.io_utils import DataValidationError
from data.common.seasons import season_label, start_year_from_code

log = logging.getLogger(__name__)

# football-data.co.uk column -> warehouse column
RENAME = {
    "Date": "match_date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HTHG": "home_ht_goals",
    "HTAG": "away_ht_goals",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

ODDS = {
    "AvgH": "avg_home_odds", "AvgD": "avg_draw_odds", "AvgA": "avg_away_odds",
    "AvgCH": "avg_close_home_odds", "AvgCD": "avg_close_draw_odds", "AvgCA": "avg_close_away_odds",
    "B365CH": "b365_close_home_odds", "B365CD": "b365_close_draw_odds", "B365CA": "b365_close_away_odds",
}

RENAME = {**RENAME, **ODDS}

TABLES = ["fact_match", "fact_team_match", "dim_team", "dim_season","fact_match_odds"]
EXTRA = ["sb_shot", "fixture"]  # counted, not exported to CSV

SB_COLS = ["event_id", "match_id", "competition_id", "period", "minute", "team", "player",
           "x", "y", "under_pressure", "shot_first_time", "play_pattern", "shot_type",
           "shot_body_part", "shot_technique", "shot_outcome", "shot_statsbomb_xg"]


def name_lookup(s: Settings, source: str) -> dict:
    names = pd.read_csv(s.reference / "team_names.csv")
    names = names[names["source"] == source]
    return dict(zip(names["source_name"], names["team"], strict=True))


def load_matches(s: Settings) -> pd.DataFrame:
    files = sorted((s.raw / "football_data").glob("E0_*.csv"))
    if not files:
        raise DataValidationError("No football-data files found; run ingest first")
    frames = []
    for f in files:
        # Some seasons add odds columns mid-season: trim long rows, never drop them
        width = len(pd.read_csv(f, encoding="latin-1", nrows=0).columns)
        df = pd.read_csv(
            f, encoding="latin-1", engine="python", on_bad_lines=lambda row, w=width: row[:w]
        )
        df = df.dropna(subset=["HomeTeam", "FTHG"])  # drop blank rows
        df = df.reindex(columns=list(RENAME))  # old seasons lack stats: NaN
        df["season"] = season_label(start_year_from_code(f.stem.split("_")[1]))
        log.debug("%s: %d matches", f.name, len(df))
        frames.append(df)
    m = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    m["match_date"] = pd.to_datetime(m["match_date"], dayfirst=True, format="mixed")
    for c in ("home_team", "away_team"):
        m[c] = m[c].str.strip()
    for side in ("home", "away"):  # source typos: more shots on target than shots
        bad = m[f"{side}_sot"] > m[f"{side}_shots"]
        if bad.any():
            log.warning(
                "%d %s shot counts impossible (on target > total); set to NULL", bad.sum(), side
            )
            m.loc[bad, [f"{side}_shots", f"{side}_sot"]] = None
    log.info("Loaded %d matches from %d season files", len(m), len(files))
    return m


def load_xg(s: Settings) -> pd.DataFrame:
    path = s.raw / "understat" / "schedule.csv"
    cols = ["season", "home_team", "away_team", "home_xg", "away_xg"]
    if not s.understat_enabled or not path.exists():
        log.warning("No Understat data; xG columns will be empty")
        return pd.DataFrame(columns=cols).astype({"home_xg": float, "away_xg": float})
    u = pd.read_csv(path)
    u = u[u["is_result"].eq(True)].copy()
    lookup = name_lookup(s, "understat")
    for c in ("home_team", "away_team"):
        u[c] = u[c].map(lookup).fillna(u[c])
    code = u["season"].astype(str).str.zfill(4)  # '1516'
    u["season"] = "20" + code.str[:2] + "/" + code.str[2:]  # '2015/16'
    log.info("Loaded %d played Understat matches", len(u))
    return u[cols]


def build_sql(st: str) -> str:
    """SQL that turns the raw tables in schema `st` into the warehouse tables."""
    return f"""
CREATE TABLE {st}.fact_match AS
SELECT
    CAST(ROW_NUMBER() OVER (ORDER BY m.match_date, m.home_team) AS INTEGER) AS match_id,
    m.season,
    CAST(m.match_date AS DATE) AS match_date,
    m.home_team,
    m.away_team,
    CAST(m.home_goals AS INTEGER) AS home_goals,
    CAST(m.away_goals AS INTEGER) AS away_goals,
    CAST(m.home_ht_goals AS INTEGER) AS home_ht_goals,
    CAST(m.away_ht_goals AS INTEGER) AS away_ht_goals,
    m.referee,
    CAST(m.home_shots AS INTEGER) AS home_shots,
    CAST(m.away_shots AS INTEGER) AS away_shots,
    CAST(m.home_sot AS INTEGER) AS home_sot,
    CAST(m.away_sot AS INTEGER) AS away_sot,
    CAST(m.home_fouls AS INTEGER) AS home_fouls,
    CAST(m.away_fouls AS INTEGER) AS away_fouls,
    CAST(m.home_corners AS INTEGER) AS home_corners,
    CAST(m.away_corners AS INTEGER) AS away_corners,
    CAST(m.home_yellows AS INTEGER) AS home_yellows,
    CAST(m.away_yellows AS INTEGER) AS away_yellows,
    CAST(m.home_reds AS INTEGER) AS home_reds,
    CAST(m.away_reds AS INTEGER) AS away_reds,
    CAST(x.home_xg AS DOUBLE PRECISION) AS home_xg,
    CAST(x.away_xg AS DOUBLE PRECISION) AS away_xg,
    CONCAT(m.home_team, ' ', CAST(m.home_goals AS INTEGER), '-',
           CAST(m.away_goals AS INTEGER), ' ', m.away_team,
           ' (', CAST(m.match_date AS DATE), ')') AS match_label
FROM {st}.raw_matches m
LEFT JOIN {st}.raw_xg x
  ON x.season = m.season AND x.home_team = m.home_team AND x.away_team = m.away_team;

ALTER TABLE {st}.fact_match ADD PRIMARY KEY (match_id);

CREATE TABLE {st}.fact_team_match AS
WITH sides AS (
    SELECT match_id, season, match_date,
           home_team AS team, away_team AS opponent, 'Home' AS venue,
           home_goals AS goals_for, away_goals AS goals_against,
           home_shots AS shots, home_sot AS shots_on_target,
           home_corners AS corners, home_fouls AS fouls,
           home_yellows AS yellows, home_reds AS reds,
           home_xg AS xg_for, away_xg AS xg_against
    FROM {st}.fact_match
    UNION ALL
    SELECT match_id, season, match_date,
           away_team, home_team, 'Away',
           away_goals, home_goals,
           away_shots, away_sot,
           away_corners, away_fouls,
           away_yellows, away_reds,
           away_xg, home_xg
    FROM {st}.fact_match
)
SELECT *,
    CASE WHEN goals_for > goals_against THEN 3
         WHEN goals_for = goals_against THEN 1 ELSE 0 END AS points,
    CASE WHEN goals_for > goals_against THEN 'W'
         WHEN goals_for = goals_against THEN 'D' ELSE 'L' END AS result,
    CAST(ROW_NUMBER() OVER (PARTITION BY season, team ORDER BY match_date) AS INTEGER)
        AS game_no
FROM sides;

ALTER TABLE {st}.fact_team_match ADD PRIMARY KEY (match_id, team);

CREATE TABLE {st}.dim_team AS
SELECT DISTINCT team FROM {st}.fact_team_match ORDER BY team;

ALTER TABLE {st}.dim_team ADD PRIMARY KEY (team);

CREATE TABLE {st}.dim_season AS
SELECT DISTINCT season, CAST(LEFT(season, 4) AS INTEGER) AS start_year
FROM {st}.fact_match ORDER BY start_year;

ALTER TABLE {st}.dim_season ADD PRIMARY KEY (season);

CREATE TABLE {st}.fact_match_odds AS
SELECT f.match_id,
       CAST(m.avg_home_odds AS DOUBLE PRECISION) AS avg_home_odds,
       CAST(m.avg_draw_odds AS DOUBLE PRECISION) AS avg_draw_odds,
       CAST(m.avg_away_odds AS DOUBLE PRECISION) AS avg_away_odds,
       CAST(m.avg_close_home_odds AS DOUBLE PRECISION) AS avg_close_home_odds,
       CAST(m.avg_close_draw_odds AS DOUBLE PRECISION) AS avg_close_draw_odds,
       CAST(m.avg_close_away_odds AS DOUBLE PRECISION) AS avg_close_away_odds,
       CAST(m.b365_close_home_odds AS DOUBLE PRECISION) AS b365_close_home_odds,
       CAST(m.b365_close_draw_odds AS DOUBLE PRECISION) AS b365_close_draw_odds,
       CAST(m.b365_close_away_odds AS DOUBLE PRECISION) AS b365_close_away_odds
FROM {st}.fact_match f
JOIN {st}.raw_matches m
  ON m.season = f.season AND m.home_team = f.home_team AND m.away_team = f.away_team;

ALTER TABLE {st}.fact_match_odds ADD PRIMARY KEY (match_id);

CREATE TABLE {st}.sb_shot AS
SELECT CAST(event_id AS TEXT) AS event_id,
       CAST(match_id AS INTEGER) AS match_id,
       CAST(competition_id AS INTEGER) AS competition_id,
       CAST(period AS INTEGER) AS period,
       CAST(minute AS INTEGER) AS minute,
       CAST(team AS TEXT) AS team,
       CAST(player AS TEXT) AS player,
       CAST(x AS DOUBLE PRECISION) AS x,
       CAST(y AS DOUBLE PRECISION) AS y,
       under_pressure,
       shot_first_time AS first_time,
       CAST(play_pattern AS TEXT) AS play_pattern,
       CAST(shot_type AS TEXT) AS shot_type,
       CAST(shot_body_part AS TEXT) AS body_part,
       CAST(shot_technique AS TEXT) AS technique,
       CAST(shot_outcome AS TEXT) AS outcome,
       CAST(shot_statsbomb_xg AS DOUBLE PRECISION) AS statsbomb_xg
FROM {st}.raw_sb_shots;

ALTER TABLE {st}.sb_shot ADD PRIMARY KEY (event_id);

CREATE TABLE {st}.fixture AS
SELECT season,
       CAST(match_date AS DATE) AS match_date,
       home_team,
       away_team,
       is_result AS is_played
FROM {st}.raw_fixtures;

ALTER TABLE {st}.fixture ADD PRIMARY KEY (season, home_team, away_team);

DROP TABLE {st}.raw_matches, {st}.raw_xg, {st}.raw_sb_shots, {st}.raw_fixtures;
"""


def build(s: Settings, engine: Engine) -> dict:
    """Rebuild every table in the staging schema. Never touches mart."""
    st = s.staging_schema
    matches, xg = load_matches(s), load_xg(s)
    shots = load_sb_shots(s)
    fixtures = load_fixtures(s)
    
    with engine.begin() as conn:  # one transaction
        conn.execute(text(f"DROP SCHEMA IF EXISTS {st} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {st}"))
        matches.to_sql("raw_matches", conn, schema=st, index=False, method="multi", chunksize=1000)
        xg.to_sql("raw_xg", conn, schema=st, index=False, method="multi", chunksize=1000)
        shots.to_sql("raw_sb_shots", conn, schema=st, index=False,
                     method="multi", chunksize=1000)
        fixtures.to_sql("raw_fixtures", conn, schema=st, index=False,
                        method="multi", chunksize=1000)
        for statement in build_sql(st).split(";"):
            if statement.strip():
                conn.execute(text(statement))
        counts = {
            t: conn.execute(text(f"SELECT COUNT(*) FROM {st}.{t}")).scalar_one()
            for t in TABLES + EXTRA
        }
    for t, n in counts.items():
        log.info("%s.%s: %s rows", st, t, f"{n:,}")
    return counts


def publish(s: Settings, engine: Engine) -> None:
    """Swap staging into mart in one transaction; leave an empty staging."""
    st, mart, reader = s.staging_schema, s.mart_schema, s.reader_role
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {mart}"))
        conn.execute(text(f"DROP SCHEMA IF EXISTS {mart}_old CASCADE"))
        conn.execute(text(f"ALTER SCHEMA {mart} RENAME TO {mart}_old"))
        conn.execute(text(f"ALTER SCHEMA {st} RENAME TO {mart}"))
        conn.execute(text(f"DROP SCHEMA {mart}_old CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {st}"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA {mart} TO {reader}"))
        conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA {mart} TO {reader}"))
    log.info("Published %s -> %s", st, mart)


def export_table(s: Settings, engine: Engine, schema: str, table: str) -> None:
    """Write one table to data/marts/<table>.csv with an atomic rename."""
    s.marts.mkdir(parents=True, exist_ok=True)
    df = pd.read_sql(text(f"SELECT * FROM {schema}.{table}"), engine)
    tmp = s.marts / f"{table}.csv.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(s.marts / f"{table}.csv")
    log.info("Exported %s.csv (%d rows)", table, len(df))


def export_marts(s: Settings, engine: Engine) -> None:
    for t in TABLES:
        export_table(s, engine, s.mart_schema, t)

def load_sb_shots(s: Settings) -> pd.DataFrame:
    folder = s.raw / "statsbomb"
    files = sorted((folder / "shots").glob("*.csv"))
    if not files or not (folder / "matches.csv").exists():
        log.warning("No StatsBomb shots on disk; sb_shot will be empty")
        return pd.DataFrame(columns=SB_COLS).astype(
            {"under_pressure": bool, "shot_first_time": bool})
    shots = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    meta = pd.read_csv(folder / "matches.csv")[["match_id", "competition_id"]]
    shots = shots.merge(meta, on="match_id", how="left").rename(columns={"id": "event_id"})
    for c in ("under_pressure", "shot_first_time"):
        shots[c] = shots[c].eq(True)  # empty cells mean False
    log.info("Loaded %d StatsBomb shots from %d matches", len(shots), len(files))
    return shots[SB_COLS]

def load_fixtures(s: Settings) -> pd.DataFrame:
    path = s.raw / "understat" / "schedule.csv"
    cols = ["season", "match_date", "home_team", "away_team", "is_result"]
    if not s.understat_enabled or not path.exists():
        log.warning("No Understat schedule; fixture will be empty")
        return pd.DataFrame(columns=cols).astype({"is_result": bool})
    u = pd.read_csv(path)
    lookup = name_lookup(s, "understat")
    for c in ("home_team", "away_team"):
        u[c] = u[c].map(lookup).fillna(u[c])
    code = u["season"].astype(str).str.zfill(4)            # '2627'
    u["season"] = "20" + code.str[:2] + "/" + code.str[2:]  # '2026/27'
    u["match_date"] = pd.to_datetime(u["date"]).dt.normalize()
    u["is_result"] = u["is_result"].eq(True)
    return u[cols]
if __name__ == "__main__":
    from data.common.config import load_settings
    from data.common.db import make_engine
    from data.common.logging_setup import setup_logging

    settings = load_settings()
    setup_logging(settings.logs)
    build(settings, make_engine())
    log.info("Staging built. Stage 7's pipeline tests it and publishes it to mart.")
