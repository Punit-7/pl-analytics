"""Read P2's inputs from the warehouse."""
import pandas as pd
from sqlalchemy import text


def load_results(engine) -> pd.DataFrame:
    df = pd.read_sql(text("""
        SELECT match_id, season, match_date, home_team, away_team, home_goals, away_goals
        FROM mart.fact_match ORDER BY match_date"""), engine)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def load_fixtures(engine, season: str) -> pd.DataFrame:
    df = pd.read_sql(text("""
        SELECT season, match_date, home_team, away_team, is_played
        FROM mart.fixture WHERE season = :season ORDER BY match_date"""),
        engine, params={"season": season})
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def current_table(engine, season: str) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT team, SUM(points) AS pts, SUM(goals_for) AS gf, SUM(goals_against) AS ga
        FROM mart.fact_team_match WHERE season = :season GROUP BY team"""),
        engine, params={"season": season})


def load_closing_odds(engine) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT match_id, avg_close_home_odds, avg_close_draw_odds, avg_close_away_odds
        FROM mart.fact_match_odds WHERE avg_close_home_odds IS NOT NULL"""), engine)