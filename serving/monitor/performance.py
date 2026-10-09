"""Performance alerts: is the live model still doing its job? Run it after serving.weekly.

python -m serving.monitor.performance
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from data.common.config import ROOT, load_settings
from data.common.logging_setup import setup_logging
from modelling.metrics import expected_calibration_error, rps
from serving.data import load_results
from serving.gate import outcome_index, paired_bootstrap

log = logging.getLogger("serving.performance")
LEDGER = ROOT / "serving" / "state" / "ledger" / "ledger.csv"
HISTORY = ROOT / "serving" / "state" / "performance_history.jsonl"
REPORTS = ROOT / "serving" / "reports"
PROBS = ["p_home", "p_draw", "p_away"]
LEVELS = ["waiting", "green", "amber", "red"]  # waiting: too few matches to judge


def add_baseline(ledger: pd.DataFrame, results: pd.DataFrame, n_history: int) -> pd.DataFrame:
    """The freq baseline for each scored match: home/draw/away shares in the matches before it."""
    past = results.sort_values("match_date")
    dates = past["match_date"].to_numpy()
    y_past = outcome_index(past["home_goals"].to_numpy(), past["away_goals"].to_numpy())
    rows = []
    for day in pd.to_datetime(ledger["match_date"]):
        end = np.searchsorted(dates, np.datetime64(day), side="left")  # matches before that day
        window = y_past[max(0, end - n_history) : end]
        rows.append(np.bincount(window, minlength=3) / max(len(window), 1))
    out = ledger.copy()
    base = np.array(rows)
    out[["base_home", "base_draw", "base_away"]] = base
    out["base_rps"] = rps(base, out["outcome"].to_numpy().astype(int))
    return out


def check(ledger: pd.DataFrame, rules: dict, seed: int = 0) -> dict:
    """Judge the most recent scored matches. Returns the level and the reason for each check."""
    recent = ledger.sort_values("match_date").tail(rules["window_matches"])
    n = len(recent)
    if n < rules["min_matches"]:
        return {
            "level": "waiting",
            "n_matches": n,
            "checks": [],
            "note": f"only {n} scored matches; {rules['min_matches']} needed for a verdict",
        }
    checks = []

    # 1. Against the freq baseline: per-match difference, with a 95% interval.
    diff = recent["model_rps"].to_numpy() - recent["base_rps"].to_numpy()  # negative = model better
    low, high = paired_bootstrap(diff, seed=seed)
    mean = float(diff.mean())
    if low > 0:
        level, why = "red", "the model is clearly worse than the freq baseline"
    elif mean >= 0:
        level, why = "amber", "the model is not better than the freq baseline"
    else:
        level, why = "green", "the model beats the freq baseline"
    checks.append(
        {
            "name": "vs_baseline",
            "level": level,
            "reason": why,
            "value": round(mean, 4),
            "low_95": round(low, 4),
            "high_95": round(high, 4),
        }
    )

    # 2. Against the bookmaker: the gap is expected; only a large gap is flagged.
    both = recent[recent["book_rps"].notna()]
    if len(both) >= rules["min_matches"]:
        gap = float((both["model_rps"] - both["book_rps"]).mean())
        level = "amber" if gap > rules["max_gap_to_book"] else "green"
        why = f"gap {gap:.4f}; the limit is {rules['max_gap_to_book']}"
        checks.append(
            {
                "name": "vs_bookmaker",
                "level": level,
                "reason": why,
                "value": round(gap, 4),
                "n": len(both),
            }
        )

    # 3. Calibration: do 40% forecasts happen about 40% of the time?
    y = recent["outcome"].to_numpy().astype(int)
    probs = recent[PROBS].to_numpy()
    hit = np.zeros_like(probs)
    hit[np.arange(n), y] = 1.0
    ece = expected_calibration_error(hit.ravel(), probs.ravel())
    level = "amber" if ece > rules["max_calibration_error"] else "green"
    checks.append(
        {
            "name": "calibration",
            "level": level,
            "value": round(ece, 4),
            "reason": f"error {ece:.4f}; the limit is {rules['max_calibration_error']}",
        }
    )

    worst = max(checks, key=lambda c: LEVELS.index(c["level"]))["level"]
    return {
        "level": worst,
        "n_matches": n,
        "checks": checks,
        "first_match": str(recent["match_date"].min())[:10],
        "last_match": str(recent["match_date"].max())[:10],
    }


def escalate(result: dict, history: list[dict], n_scored: int, amber_runs: int) -> dict:
    """Amber on `amber_runs` runs in a row, each with new matches, becomes red."""
    new_matches = not history or history[-1]["n_scored"] < n_scored
    streak = [h["level"] for h in history[-(amber_runs - 1) :]] if amber_runs > 1 else []
    if (
        result["level"] == "amber"
        and new_matches
        and len(streak) == amber_runs - 1
        and all(level in ("amber", "red") for level in streak)
    ):
        result = {**result, "level": "red", "escalated": f"amber on {amber_runs} runs in a row"}
    return result


def alert_text(result: dict, run_at: str) -> str:
    """The body of the GitHub issue."""
    lines = [
        f"Weekly run {run_at}: performance level **{result['level'].upper()}**.",
        "",
        f"Judged on {result['n_matches']} matches, {result.get('first_match')} to "
        f"{result.get('last_match')}.",
    ]
    if result.get("escalated"):
        lines.append(f"Raised to red because it was {result['escalated']}.")
    lines += ["", "| Check | Level | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c['name']} | {c['level']} | {c['reason']} |" for c in result["checks"]]
    lines += ["", "Follow the response plan in `serving/RUNBOOK.md`, step 1 first."]
    return "\n".join(lines) + "\n"


def read_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    return [json.loads(line) for line in HISTORY.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    cfg = s.serving
    rules = cfg["alerts"]
    run = json.loads((REPORTS / "run.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(LEDGER, parse_dates=["match_date"]) if LEDGER.exists() else pd.DataFrame()
    if ledger.empty:
        result = {"level": "waiting", "n_matches": 0, "checks": [], "note": "no scored matches yet"}
    else:
        results = load_results(s, ingest=False)
        ledger = add_baseline(ledger, results, rules["baseline_history"])
        result = check(ledger, rules, seed=cfg["random_seed"])
    history = read_history()
    result = escalate(result, history, len(ledger), rules["amber_runs_to_red"])
    result["run_at"] = run["run_at"]

    (REPORTS / "performance.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if not history or history[-1]["n_scored"] < len(ledger):  # only runs with new matches count
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"run_at": run["run_at"], "n_scored": len(ledger), "level": result["level"]}
                )
                + "\n"
            )
    alert = REPORTS / "ALERT.md"
    if result["level"] == "red":
        alert.write_text(alert_text(result, run["run_at"]), encoding="utf-8")
    else:
        alert.unlink(missing_ok=True)
    log.info("Performance: %s on %d matches", result["level"], result["n_matches"])


if __name__ == "__main__":
    main()
