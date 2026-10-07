"""Helpers shared by the evaluation scripts."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from datetime import UTC, date, datetime
from decimal import Decimal

from data.common.config import ROOT

REPORTS = ROOT / "assistant" / "reports"


def normalise(value):
    """Make equal answers compare as equal: 7, 7.0 and Decimal('7.00') are the same number."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int | float | Decimal):
        return round(float(value), 2)
    if isinstance(value, date | datetime):
        return value.isoformat()
    return str(value).strip()


def same_result(gold: list[tuple], pred: list[tuple]) -> bool:
    """True if both queries returned the same rows. Row order and column order are ignored."""
    g = [tuple(normalise(v) for v in row) for row in gold]
    p = [tuple(normalise(v) for v in row) for row in pred]
    if len(g) != len(p):
        return False
    if not g:
        return True
    width = len(g[0])
    if any(len(row) != width for row in p):
        return False
    target = Counter(g)
    orders = itertools.permutations(range(width)) if width <= 5 else [tuple(range(width))]
    return any(Counter(tuple(row[i] for i in order) for row in p) == target for order in orders)


def proportion(k: int, n: int) -> dict:
    """A share k/n with its standard error and an approximate 95% interval."""
    if n == 0:
        return {"value": None, "k": 0, "n": 0}
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return {
        "value": round(p, 3),
        "k": k,
        "n": n,
        "standard_error": round(se, 3),
        "low_95": round(max(0.0, p - 1.96 * se), 3),
        "high_95": round(min(1.0, p + 1.96 * se), 3),
    }


def run_info(cfg: dict, model: str) -> dict:
    return {
        "assistant_version": cfg["assistant_version"],
        "provider": cfg["provider"],
        "model": model,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def save_report(name: str, data: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Saved {path.relative_to(ROOT)}")
