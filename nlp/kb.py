"""Build nlp.kb_entity: every known player, club and venue, one row per alias."""
import logging

import pandas as pd
import soccerdata as sd
from sqlalchemy import text

from data.build_warehouse import name_lookup
from data.common.config import load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from data.common.seasons import current_season_start

log = logging.getLogger("nlp.kb")
KB_COLS = ["entity_id", "entity_type", "name", "alias", "alias_kind", "season", "team", "minutes"]


def load_players(s) -> pd.DataFrame:
    current = current_season_start(start_month=s.season_start_month)
    seasons = [f"{y}/{y + 1}" for y in range(s.nlp["first_season"], current + 1)]
    understat = sd.Understat(leagues="ENG-Premier League", seasons=seasons)
    ps = understat.read_player_season_stats().reset_index()
    log.info("Understat player columns: %s", list(ps.columns))
    ps["team"] = ps["team"].astype(str).str.split(",")   # transferred players list two clubs
    ps = ps.explode("team")
    ps["team"] = ps["team"].str.strip()
    lookup = name_lookup(s, "understat")
    ps["team"] = ps["team"].map(lookup).fillna(ps["team"])
    code = ps["season"].astype(str).str.zfill(4)
    ps["season"] = "20" + code.str[:2] + "/" + code.str[2:]
    return ps[["season", "team", "player", "player_id", "minutes"]]


def player_rows(ps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in ps.itertuples():
        eid = f"player:{r.player_id}"
        rows.append((eid, "PLAYER", r.player, r.player, "full", r.season, r.team, r.minutes))
        parts = r.player.split()
        if len(parts) > 1:
            rows.append((eid, "PLAYER", r.player, parts[-1], "surname",
                         r.season, r.team, r.minutes))
    return pd.DataFrame(rows, columns=KB_COLS)


def team_rows(engine, s) -> pd.DataFrame:
    teams = pd.read_sql(text("SELECT team FROM mart.dim_team"), engine)["team"]
    aliases = pd.read_csv(s.reference / "team_aliases.csv")
    rows = [(f"team:{t}", "TEAM", t, t, "name", None, t, None) for t in teams]
    rows += [(f"team:{a.team}", "TEAM", a.team, a.alias, "alias", None, a.team, None)
             for a in aliases.itertuples()]
    return pd.DataFrame(rows, columns=KB_COLS)


def venue_rows(s) -> pd.DataFrame:
    venues = pd.read_csv(s.reference / "venues.csv").fillna("")
    rows = []
    for v in venues.itertuples():
        eid = f"venue:{v.venue_id}"
        team = v.team or None
        rows.append((eid, "VENUE", v.venue, v.venue, "name", None, team, None))
        rows += [(eid, "VENUE", v.venue, a, "alias", None, team, None)
                 for a in v.aliases.split("|") if a]
    return pd.DataFrame(rows, columns=KB_COLS)


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    engine = make_engine()
    kb = pd.concat([player_rows(load_players(s)), team_rows(engine, s), venue_rows(s)],
                   ignore_index=True).drop_duplicates()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS nlp"))
        kb.to_sql("kb_entity", conn, schema="nlp", if_exists="replace", index=False)
        conn.execute(text("CREATE INDEX ON nlp.kb_entity (alias)"))
        conn.execute(text(f"GRANT SELECT ON nlp.kb_entity TO {s.reader_role}"))
    log.info("KB rows by type: %s", kb.groupby("entity_type").size().to_dict())


if __name__ == "__main__":
    main()