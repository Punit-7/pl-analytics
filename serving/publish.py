"""Predictions for every match still to be played, and the season simulation."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import permutations
from pathlib import Path

import pandas as pd

from data.common.config import ROOT
from modelling.match.dixon_coles import DixonColes
from modelling.match.simulate import simulate

OUT = ROOT / "serving" / "predictions"
LATEST = OUT / "latest.json"


class NotReady(Exception):
    """The season has too few results to know its 20 teams."""


def remaining_pairs(results: pd.DataFrame, season: str) -> pd.DataFrame:
    """Every home/away pairing of this season's teams that has no result yet."""
    played = results[results["season"] == season]
    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    if len(teams) != 20:
        raise NotReady(f"{season}: {len(teams)} teams have played; 20 are needed")
    done = set(zip(played["home_team"], played["away_team"], strict=True))
    pairs = [p for p in permutations(teams, 2) if p not in done]
    return pd.DataFrame(pairs, columns=["home_team", "away_team"])


def match_predictions(
    model: DixonColes, pairs: pd.DataFrame, season: str, now: datetime, version: str
) -> pd.DataFrame:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for f in pairs.itertuples():
        matrix = model.score_matrix(f.home_team, f.away_team)
        p_home, p_draw, p_away = model.outcome_probs(f.home_team, f.away_team)
        best_home, best_away = divmod(int(matrix.argmax()), matrix.shape[1])
        outcome = {"home": p_home, "draw": p_draw, "away": p_away}
        lam, mu = model.expected_goals(f.home_team, f.away_team)
        rows.append(
            {
                "prediction_id": f"{season}|{f.home_team}|{f.away_team}|{stamp}",
                "created_at": now.replace(tzinfo=None).isoformat(timespec="seconds"),
                "model_version": version,
                "season": season,
                "home_team": f.home_team,
                "away_team": f.away_team,
                "p_home": round(p_home, 4),
                "p_draw": round(p_draw, 4),
                "p_away": round(p_away, 4),
                "exp_home_goals": round(lam, 3),
                "exp_away_goals": round(mu, 3),
                "most_likely_score": f"{best_home}-{best_away}",
                "most_likely_score_prob": round(float(matrix[best_home, best_away]), 4),
                "predicted_result": max(outcome, key=outcome.get),
            }
        )
    return pd.DataFrame(rows)


def current_table(results: pd.DataFrame, season: str) -> pd.DataFrame:
    """Points, goals for and goals against so far: the same numbers as P2's SQL."""
    m = results[results["season"] == season]
    home = pd.DataFrame(
        {
            "team": m["home_team"],
            "gf": m["home_goals"],
            "ga": m["away_goals"],
            "pts": (m["home_goals"] > m["away_goals"]) * 3 + (m["home_goals"] == m["away_goals"]),
        }
    )
    away = pd.DataFrame(
        {
            "team": m["away_team"],
            "gf": m["away_goals"],
            "ga": m["home_goals"],
            "pts": (m["away_goals"] > m["home_goals"]) * 3 + (m["home_goals"] == m["away_goals"]),
        }
    )
    return pd.concat([home, away]).groupby("team", as_index=False)[["pts", "gf", "ga"]].sum()


def publish(
    model: DixonColes,
    results: pd.DataFrame,
    season: str,
    now: datetime,
    version: str,
    cfg: dict,
    out: Path = OUT,
) -> dict:
    """Write the timestamped CSV files and latest.json. Return what was written."""
    pairs = remaining_pairs(results, season)
    preds = match_predictions(model, pairs, season, now, version)
    table = simulate(
        model, current_table(results, season), pairs, cfg["n_sims"], cfg["random_seed"]
    )
    table = table.round(4)
    table.insert(0, "season", season)
    table.insert(0, "model_version", version)
    table.insert(0, "created_at", now.replace(tzinfo=None).isoformat(timespec="seconds"))

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    folder = out / season.replace("/", "-")
    folder.mkdir(parents=True, exist_ok=True)
    preds.to_csv(folder / f"{stamp}_matches.csv", index=False)
    table.to_csv(folder / f"{stamp}_season_sim.csv", index=False)
    latest = {
        "created_at": now.replace(tzinfo=None).isoformat(timespec="seconds"),
        "model_version": version,
        "season": season,
        "matches": preds.drop(
            columns=["prediction_id", "created_at", "model_version", "season"]
        ).to_dict(orient="records"),
        "season_sim": table.drop(columns=["created_at", "model_version", "season"]).to_dict(
            orient="records"
        ),
    }
    (out / "latest.json").write_text(json.dumps(latest, indent=1), encoding="utf-8")
    return {
        "n_predictions": int(len(preds)),
        "stamp": stamp,
        "folder": str(folder.relative_to(out)),
    }
