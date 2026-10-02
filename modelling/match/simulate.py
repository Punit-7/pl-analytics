"""Simulate the rest of the season: title, top-four and relegation probabilities."""
import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from data.common.seasons import current_season_start, season_label
from modelling.match.data import current_table, load_fixtures, load_results
from modelling.match.dixon_coles import DixonColes, fit_dixon_coles

log = logging.getLogger("modelling.match.simulate")


def remaining_fixtures(fixtures: pd.DataFrame, results: pd.DataFrame, season: str) -> pd.DataFrame:
    """Fixtures with no result in fact_match yet (the same source as the table)."""
    played = results.loc[results["season"] == season, ["home_team", "away_team"]]
    merged = fixtures.merge(played, on=["home_team", "away_team"], how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge")


def simulate(model: DixonColes, table: pd.DataFrame, fixtures: pd.DataFrame,
             n_sims: int, seed: int) -> pd.DataFrame:
    teams = sorted(set(table["team"]) | set(fixtures["home_team"]) | set(fixtures["away_team"]))
    ix = {t: i for i, t in enumerate(teams)}
    base = table.set_index("team").reindex(teams)[["pts", "gf", "ga"]].fillna(0).astype(float)
    pts = np.tile(base["pts"].to_numpy(), (n_sims, 1))
    gf = np.tile(base["gf"].to_numpy(), (n_sims, 1))
    ga = np.tile(base["ga"].to_numpy(), (n_sims, 1))
    rng = np.random.default_rng(seed)
    k = model.max_goals + 1

    for f in fixtures.itertuples():
        probs = model.score_matrix(f.home_team, f.away_team).ravel()
        draw = rng.choice(k * k, size=n_sims, p=probs)
        hg, ag = draw // k, draw % k
        h, a = ix[f.home_team], ix[f.away_team]
        gf[:, h] += hg
        ga[:, h] += ag
        gf[:, a] += ag
        ga[:, a] += hg
        pts[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))

    # Rank by points, then goal difference, then goals scored; a random number breaks any tie left.
    key = pts * 1e6 + (gf - ga) * 1e3 + gf + rng.random(pts.shape)
    order = np.argsort(-key, axis=1)
    position = np.empty_like(order)
    position[np.arange(n_sims)[:, None], order] = np.arange(1, len(teams) + 1)
    n = len(teams)
    return (pd.DataFrame({
        "team": teams,
        "exp_points": pts.mean(axis=0),
        "p_title": (position == 1).mean(axis=0),
        "p_top4": (position <= 4).mean(axis=0),
        "p_relegation": (position >= n - 2).mean(axis=0),
    }).sort_values("exp_points", ascending=False).reset_index(drop=True))


def run(s, engine, as_of: date) -> tuple[str, pd.DataFrame]:
    m = s.modelling
    season = season_label(current_season_start(start_month=s.season_start_month))
    results = load_results(engine)
    model = fit_dixon_coles(results, as_of, m["dc_xi"], m["dc_history_days"], m["dc_max_goals"])
    remaining = remaining_fixtures(load_fixtures(engine, season), results, season)
    log.info("%s: %d fixtures left to simulate", season, len(remaining))
    return season, simulate(model, current_table(engine, season), remaining,
                            m["n_sims"], m["random_seed"])


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    season, result = run(s, make_engine(), date.today())
    out_dir = ROOT / "modelling" / "predictions" / season.replace("/", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result.to_csv(out_dir / f"{stamp}_season_sim.csv", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()