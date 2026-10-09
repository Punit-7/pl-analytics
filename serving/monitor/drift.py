"""Data drift: do recent matches look like the matches the model was built on?"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
FLOOR = 0.0001  # used as the share of an empty bin, so that the logarithm is defined
CAPS = {"home_goals": 4, "away_goals": 4, "total_goals": 6}  # the top bin is "this many or more"


def match_features(results: pd.DataFrame) -> pd.DataFrame:
    """The columns that are watched: goals, and the result as H, D or A."""
    out = pd.DataFrame(
        {
            "home_goals": results["home_goals"].astype(int),
            "away_goals": results["away_goals"].astype(int),
        }
    )
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    diff = out["home_goals"] - out["away_goals"]
    out["result"] = diff.map(lambda d: "H" if d > 0 else "D" if d == 0 else "A")
    return out


def windows(results: pd.DataFrame, today, n_current: int, n_reference: int):
    """The most recent `n_current` matches, and the `n_reference` matches before them."""
    past = results[results["match_date"] < pd.Timestamp(today).normalize()]
    past = past.sort_values("match_date")
    current = past.tail(n_current)
    reference = past.iloc[: len(past) - len(current)].tail(n_reference)
    return match_features(reference), match_features(current)


def psi(reference: pd.Series, current: pd.Series) -> float:
    """Population stability index: the sum over bins of (c - r) * ln(c / r)."""
    bins = sorted(set(reference) | set(current))
    r = reference.value_counts(normalize=True)
    c = current.value_counts(normalize=True)
    total = 0.0
    for b in bins:
        ri, ci = max(float(r.get(b, 0.0)), FLOOR), max(float(c.get(b, 0.0)), FLOOR)
        total += (ci - ri) * math.log(ci / ri)
    return total


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, alert: float) -> dict:
    """PSI per watched column, with a flag where it is above the alert level."""
    table = {}
    for column in reference.columns:
        ref, cur = reference[column], current[column]
        if column in CAPS:
            ref, cur = ref.clip(upper=CAPS[column]), cur.clip(upper=CAPS[column])
        value = psi(ref, cur)
        table[column] = {"psi": round(value, 4), "alert": bool(value > alert)}
    return table


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> bool:
    """Write Evidently's HTML drift report. A failure here must not stop the monitoring job."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        definition = DataDefinition(
            numerical_columns=["home_goals", "away_goals", "total_goals"],
            categorical_columns=["result"],
        )
        ref = Dataset.from_pandas(reference, data_definition=definition)
        cur = Dataset.from_pandas(current, data_definition=definition)
        snapshot = Report([DataDriftPreset()]).run(cur, ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(path))
        return True
    except Exception as e:
        log.warning("Evidently report not written: %s", e)
        return False
