"""Score published predictions against results. Only each match's first prediction counts."""
import logging

import pandas as pd
from sqlalchemy import text

from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from modelling.match.backtest import outcome_index
from modelling.match.benchmark import ODDS, implied_probs
from modelling.metrics import rps

log = logging.getLogger("modelling.match.ledger")

QUERY = """
WITH first_pred AS (
    SELECT DISTINCT ON (season, home_team, away_team) *
    FROM predictions.match_prediction
    ORDER BY season, home_team, away_team, created_at
)
SELECT p.prediction_id, p.created_at, p.model_version, p.season, p.match_date,
       p.home_team, p.away_team, p.p_home, p.p_draw, p.p_away,
       m.home_goals, m.away_goals,
       o.avg_close_home_odds, o.avg_close_draw_odds, o.avg_close_away_odds
FROM first_pred p
JOIN mart.fact_match m
  ON m.season = p.season AND m.home_team = p.home_team AND m.away_team = p.away_team
LEFT JOIN mart.fact_match_odds o ON o.match_id = m.match_id
ORDER BY p.match_date
"""


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    engine = make_engine()
    df = pd.read_sql(text(QUERY), engine)
    if df.empty:
        log.info("No published predictions have results yet")
        return
    y = outcome_index(df["home_goals"].to_numpy(), df["away_goals"].to_numpy())
    df["outcome"] = y
    df["model_rps"] = rps(df[["p_home", "p_draw", "p_away"]].to_numpy(), y)
    has_odds = df[ODDS].notna().all(axis=1).to_numpy()
    book, _ = implied_probs(df.loc[has_odds, ODDS].to_numpy())
    df.loc[has_odds, "book_rps"] = rps(book, y[has_odds])
    df["cumulative_model_rps"] = df["model_rps"].expanding().mean()

    with engine.begin() as conn:
        df.to_sql("ledger", conn, schema="predictions", if_exists="replace", index=False)
        conn.execute(text(f"GRANT SELECT ON predictions.ledger TO {s.reader_role}"))
    season_dir = df["season"].iloc[-1].replace("/", "-")
    out = ROOT / "modelling" / "ledger" / f"ledger_{season_dir}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    both = df[has_odds]
    log.info("%d scored | model RPS %.4f | on %d matches with odds: model %.4f vs odds %.4f",
             len(df), df["model_rps"].mean(), len(both),
             both["model_rps"].mean(), both["book_rps"].mean())


if __name__ == "__main__":
    main()