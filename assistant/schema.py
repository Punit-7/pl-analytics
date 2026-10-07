"""Describe the warehouse tables to the model: the schema card."""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from data.common.seasons import current_season_start, season_label

# What each table and column means. tests/test_assistant_schema.py checks it against the database.
NOTES = {
    "mart.fact_team_match": {
        "about": "One row per team per match, so every match has two rows. "
        "Use this table for almost every question about a team.",
        "columns": {
            "match_id": "match number; joins to mart.fact_match",
            "season": "text such as '2023/24'",
            "match_date": "date of the match",
            "team": "the team this row is about",
            "opponent": "the other team",
            "venue": "'Home' or 'Away', from this team's point of view",
            "goals_for": "goals this team scored",
            "goals_against": "goals this team conceded",
            "shots": "shots by this team",
            "shots_on_target": "shots on target by this team",
            "corners": "corners won by this team",
            "fouls": "fouls committed by this team",
            "yellows": "yellow cards shown to this team",
            "reds": "red cards shown to this team",
            "xg_for": "expected goals created by this team",
            "xg_against": "expected goals conceded by this team",
            "points": "3 for a win, 1 for a draw, 0 for a loss",
            "result": "'W', 'D' or 'L'",
            "game_no": "1 for the team's first match of the season, 2 for the second, and so on",
        },
    },
    "mart.fact_match": {
        "about": "One row per match. Use it for questions about matches, not about one team.",
        "columns": {
            "match_id": "match number",
            "season": "text such as '2023/24'",
            "match_date": "date of the match",
            "home_team": "home team",
            "away_team": "away team",
            "home_goals": "full-time goals by the home team",
            "away_goals": "full-time goals by the away team",
            "home_ht_goals": "half-time goals by the home team",
            "away_ht_goals": "half-time goals by the away team",
            "referee": "referee as an initial and a surname, such as 'M Oliver'",
            "home_shots": "shots by the home team",
            "away_shots": "shots by the away team",
            "home_sot": "shots on target by the home team",
            "away_sot": "shots on target by the away team",
            "home_fouls": "fouls by the home team",
            "away_fouls": "fouls by the away team",
            "home_corners": "corners won by the home team",
            "away_corners": "corners won by the away team",
            "home_yellows": "yellow cards for the home team",
            "away_yellows": "yellow cards for the away team",
            "home_reds": "red cards for the home team",
            "away_reds": "red cards for the away team",
            "home_xg": "expected goals of the home team",
            "away_xg": "expected goals of the away team",
            "match_label": "text such as 'Arsenal 2-1 Chelsea (2026-09-20)'",
        },
    },
    "mart.fixture": {
        "about": "One row per scheduled match, played or not. Use it for matches still to come.",
        "columns": {
            "season": "text such as '2026/27'",
            "match_date": "scheduled date",
            "home_team": "home team",
            "away_team": "away team",
            "is_played": "true if the match has been played",
        },
    },
}

RULES = """Rules:
- Write one PostgreSQL SELECT statement. Use only the tables above.
- A season is text in the form '2023/24'. Seasons sort correctly as text.
- Team names must be written exactly as in the team list. In SQL, a quote inside a name is
  doubled: 'Nott''m Forest'.
- A league table is SUM(points) per team in one season. Goal difference is
  SUM(goals_for) - SUM(goals_against).
- To count something per team, including teams with a count of zero, use
  SUM(CASE WHEN ... THEN 1 ELSE 0 END), not a WHERE filter.
- Round only when the question asks for it: ROUND(CAST(x AS numeric), 2).
- Return only the columns the question asks for."""


def live_columns(engine: Engine, tables: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Column names and types of each table, read from the database."""
    out = {}
    for full in tables:
        schema, table = full.split(".")
        df = pd.read_sql(
            text("""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema = :s AND table_name = :t ORDER BY ordinal_position"""),
            engine,
            params={"s": schema, "t": table},
        )
        out[full] = list(zip(df["column_name"], df["data_type"], strict=True))
    return out


def value_hints(engine: Engine, today: date | None = None, start_month: int = 8) -> str:
    """The exact team names, the seasons covered, and the current season."""
    teams = pd.read_sql(text("SELECT DISTINCT team FROM mart.fact_team_match ORDER BY 1"), engine)
    span = pd.read_sql(
        text("SELECT MIN(season) AS first, MAX(season) AS last FROM mart.fact_match"), engine
    ).iloc[0]
    current = season_label(current_season_start(today, start_month))
    return (
        f"Team names: {', '.join(teams['team'])}.\n"
        f"Seasons: '{span['first']}' to '{span['last']}'. The current season is '{current}'. "
        "xG columns are empty before '2014/15'."
    )


def schema_card(engine: Engine, tables: list[str], full: bool = True) -> str:
    """Text that tells the model what it may query. full=False gives names and types only."""
    columns = live_columns(engine, tables)
    lines = []
    for table in tables:
        if not full:
            cols = ", ".join(f"{name} {dtype}" for name, dtype in columns[table])
            lines.append(f"{table}({cols})")
            continue
        notes = NOTES[table]
        lines.append(f"Table {table}: {notes['about']}")
        for name, dtype in columns[table]:
            lines.append(f"  {name} ({dtype}): {notes['columns'].get(name, '')}")
    if full:
        lines += ["", value_hints(engine), "", RULES]
    return "\n".join(lines)


if __name__ == "__main__":
    from data.common.config import load_settings
    from data.common.db import make_reader_engine

    card = schema_card(make_reader_engine(), load_settings().assistant["sql_tables"])
    print(card)
    print(f"\n{len(card)} characters, roughly {len(card) // 4} tokens")