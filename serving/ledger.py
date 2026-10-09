"""Score published predictions against results: RPS, the bookmaker gap, calibration."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from data.common.config import ROOT
from modelling.metrics import expected_calibration_error, rps
from serving.data import ODDS
from serving.gate import outcome_index

log = logging.getLogger(__name__)
FOLDERS = [ROOT / "modelling" / "predictions", ROOT / "serving" / "predictions"]
OUT = ROOT / "serving" / "state" / "ledger"
PROBS = ["p_home", "p_draw", "p_away"]


def load_predictions(folders=FOLDERS) -> pd.DataFrame:
    """Every published match prediction: P2's manual ones and the weekly job's."""
    frames = []
    for folder in folders:
        for f in sorted(folder.glob("*/*_matches.csv")):
            try:
                frames.append(pd.read_csv(f))
            except pd.errors.EmptyDataError:  # a week with no fixtures left a blank file
                continue
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["season", "home_team", "away_team", "created_at", *PROBS])
    preds = pd.concat(frames, ignore_index=True)
    preds["created_at"] = pd.to_datetime(preds["created_at"], format="ISO8601")
    return preds.drop(columns=["match_date"], errors="ignore")


def implied_probs(odds: np.ndarray) -> np.ndarray:
    """Bookmaker probabilities: 1/odds, rescaled to add up to 1 (as in P2 Stage 10)."""
    q = 1.0 / odds
    return q / q.sum(axis=1, keepdims=True)


def build_ledger(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per played match: the latest prediction published before the match day."""
    keys = ["season", "home_team", "away_team"]
    df = preds.merge(results, on=keys, how="inner")
    df = df[df["created_at"].dt.normalize() < df["match_date"]]  # made at least a day before
    df = df.sort_values("created_at").groupby(keys, as_index=False).last()
    if df.empty:
        return df
    df = df.sort_values(["match_date", "home_team"]).reset_index(drop=True)
    y = outcome_index(df["home_goals"].to_numpy(), df["away_goals"].to_numpy())
    df["outcome"] = y
    df["model_rps"] = rps(df[PROBS].to_numpy(), y)
    has_odds = df[ODDS].notna().all(axis=1).to_numpy()
    df["book_rps"] = np.nan
    if has_odds.any():
        book = implied_probs(df.loc[has_odds, ODDS].to_numpy(dtype=float))
        df.loc[has_odds, "book_rps"] = rps(book, y[has_odds])
    names = np.array(["home", "draw", "away"])
    df["predicted_result"] = names[df[PROBS].to_numpy().argmax(axis=1)]  # the headline pick
    df["pick_correct"] = df["predicted_result"].to_numpy() == names[y]
    df["days_ahead"] = (df["match_date"] - df["created_at"].dt.normalize()).dt.days
    return df


def summarise(ledger: pd.DataFrame) -> dict:
    """The numbers tracked every week. Calibration error pools all three outcome probabilities."""
    if ledger.empty:
        return {"n_scored": 0}
    y = ledger["outcome"].to_numpy()
    probs = ledger[PROBS].to_numpy()
    hit = np.zeros_like(probs)
    hit[np.arange(len(y)), y] = 1.0
    both = ledger[ledger["book_rps"].notna()]
    out = {
        "n_scored": int(len(ledger)),
        "model_rps": round(float(ledger["model_rps"].mean()), 4),
        "pick_accuracy": round(float(ledger["pick_correct"].mean()), 4),
        "calibration_error": round(expected_calibration_error(hit.ravel(), probs.ravel()), 4),
        "n_with_odds": int(len(both)),
    }
    if len(both):
        out["model_rps_on_odds_matches"] = round(float(both["model_rps"].mean()), 4)
        out["book_rps"] = round(float(both["book_rps"].mean()), 4)
        out["rps_gap_to_book"] = round(out["model_rps_on_odds_matches"] - out["book_rps"], 4)
    return out


def update(results: pd.DataFrame) -> dict:
    ledger = build_ledger(load_predictions(), results)
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["season", "match_date", "home_team", "away_team", "home_goals", "away_goals",
            "created_at", "model_version", "days_ahead", *PROBS, "predicted_result", "outcome",
            "pick_correct", "model_rps", "book_rps"]  # fmt: skip
    ledger.reindex(columns=keep).to_csv(OUT / "ledger.csv", index=False)
    summary = summarise(ledger)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("Ledger: %s", summary)
    return summary
