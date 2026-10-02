"""Publish predictions for the coming week's matches and a fresh season simulation."""
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from data.common.seasons import current_season_start, season_label
from modelling.match import simulate as sim
from modelling.match.data import current_table, load_fixtures, load_results
from modelling.match.dixon_coles import fit_dixon_coles

log = logging.getLogger("modelling.match.predict")

DDL = [
    "CREATE SCHEMA IF NOT EXISTS predictions",
    """CREATE TABLE IF NOT EXISTS predictions.match_prediction (
        prediction_id TEXT PRIMARY KEY, created_at TIMESTAMP NOT NULL,
        model_version TEXT NOT NULL, season TEXT NOT NULL, match_date DATE NOT NULL,
        home_team TEXT NOT NULL, away_team TEXT NOT NULL,
        p_home DOUBLE PRECISION NOT NULL, p_draw DOUBLE PRECISION NOT NULL,
        p_away DOUBLE PRECISION NOT NULL,
        exp_home_goals DOUBLE PRECISION, exp_away_goals DOUBLE PRECISION)""",
    """CREATE TABLE IF NOT EXISTS predictions.season_sim (
        created_at TIMESTAMP NOT NULL, model_version TEXT NOT NULL, season TEXT NOT NULL,
        team TEXT NOT NULL, exp_points DOUBLE PRECISION, p_title DOUBLE PRECISION,
        p_top4 DOUBLE PRECISION, p_relegation DOUBLE PRECISION,
        PRIMARY KEY (created_at, team))""",
]


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    m = s.modelling
    engine = make_engine()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = pd.Timestamp(now.date())
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    season = season_label(current_season_start(start_month=s.season_start_month))

    results = load_results(engine)
    model = fit_dixon_coles(results, today, m["dc_xi"], m["dc_history_days"], m["dc_max_goals"])
    remaining = sim.remaining_fixtures(load_fixtures(engine, season), results, season)
    window_end = today + pd.Timedelta(days=m["prediction_window_days"])
    upcoming = remaining[(remaining["match_date"] > today) & (remaining["match_date"] <= window_end)]

    rows = []
    for f in upcoming.itertuples():
        p_home, p_draw, p_away = model.outcome_probs(f.home_team, f.away_team)
        lam, mu = model.expected_goals(f.home_team, f.away_team)
        rows.append({"prediction_id": f"{season}|{f.home_team}|{f.away_team}|{stamp}",
                     "created_at": now, "model_version": m["model_version"], "season": season,
                     "match_date": f.match_date.date(), "home_team": f.home_team,
                     "away_team": f.away_team, "p_home": p_home, "p_draw": p_draw,
                     "p_away": p_away, "exp_home_goals": lam, "exp_away_goals": mu})
    preds = pd.DataFrame(rows)
    table = sim.simulate(model, current_table(engine, season), remaining,
                         m["n_sims"], m["random_seed"])
    table.insert(0, "season", season)
    table.insert(0, "model_version", m["model_version"])
    table.insert(0, "created_at", now)

    with engine.begin() as conn:
        for statement in DDL:
            conn.execute(text(statement))
        if not preds.empty:
            preds.to_sql("match_prediction", conn, schema="predictions",
                         if_exists="append", index=False)
        table.to_sql("season_sim", conn, schema="predictions", if_exists="append", index=False)
        conn.execute(text(f"GRANT USAGE ON SCHEMA predictions TO {s.reader_role}"))
        conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA predictions TO {s.reader_role}"))

    out_dir = ROOT / "modelling" / "predictions" / season.replace("/", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / f"{stamp}_matches.csv", index=False)
    table.to_csv(out_dir / f"{stamp}_season_sim.csv", index=False)
    log.info("Published %d match predictions (to %s) and a season simulation",
             len(preds), window_end.date())


if __name__ == "__main__":
    main()