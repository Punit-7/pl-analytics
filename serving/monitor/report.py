"""The monitoring job: drift, MLflow, and a status page. Run it after serving.weekly.

python -m serving.monitor.report
"""

import json
import logging
from datetime import date

from data.common.config import ROOT, load_settings
from data.common.logging_setup import setup_logging
from serving.data import load_results
from serving.monitor import drift, tracking

log = logging.getLogger("serving.monitor")
REPORTS = ROOT / "serving" / "reports"


def performance_lines(perf: dict | None) -> list[str]:
    """The live-performance section, from performance.json (Stage 14)."""
    if not perf:
        return []
    lines = ["", "## Live performance", "", f"Level: **{perf['level'].upper()}**"]
    if perf.get("note"):
        return [*lines, "", perf["note"]]
    lines += ["", f"Judged on the last {perf['n_matches']} scored matches.", "",
              "| Check | Level | Detail |", "| --- | --- | --- |"]  # fmt: skip
    lines += [f"| {c['name']} | {c['level']} | {c['reason']} |" for c in perf["checks"]]
    return lines


def status_page(run: dict, table: dict, html_written: bool, perf: dict | None = None) -> str:
    model, data, ledger, gate = run["model"], run["data"], run["ledger"], run.get("gate")
    lines = [
        "# Weekly status",
        "",
        f"Last run: {run['run_at']}",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Live model | `{model['model_version']}` |",
        f"| Trained on results up to | {model['trained_through']} |",
        f"| Data | {data['n_matches']} matches, latest {data['days_since_latest']} days old"
        f"{' (STALE)' if data['stale'] else ''} |",
        f"| Predictions published | {run['published']['n_predictions']} |",
        f"| Gate | {gate['reason'] if gate else 'no challenger configured'} |",
        f"| Matches scored so far | {ledger.get('n_scored', 0)} |",
    ]
    for key, label in (
        ("model_rps", "Model RPS"),
        ("book_rps", "Bookmaker RPS"),
        ("rps_gap_to_book", "Gap to bookmaker (lower is better)"),
        ("pick_accuracy", "Headline picks correct"),
        ("calibration_error", "Calibration error"),
    ):
        if key in ledger:
            lines.append(f"| {label} | {ledger[key]} |")
    lines += performance_lines(perf)
    lines += ["", "## Drift", "", "| Column | PSI | Alert |", "| --- | --- | --- |"]
    for column, row in table.items():
        lines.append(f"| {column} | {row['psi']} | {'ALERT' if row['alert'] else 'ok'} |")
    lines += ["", "PSI compares the most recent matches with the matches before them."]
    if html_written:
        lines.append("The full Evidently report is attached to the workflow run as an artifact.")
    return "\n".join(lines) + "\n"


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    cfg = s.serving
    run = json.loads((REPORTS / "run.json").read_text(encoding="utf-8"))
    today = date.fromisoformat(run["model"]["fitted_on"])
    results = load_results(s, ingest=False)
    reference, current = drift.windows(
        results, today, cfg["drift_current_matches"], cfg["drift_reference_matches"]
    )
    enough = len(current) and len(reference)
    table = drift.drift_table(reference, current, cfg["psi_alert"]) if enough else {}
    (REPORTS / "drift.json").write_text(
        json.dumps({"reference_matches": len(reference), "current_matches": len(current),
                    "columns": table}, indent=1),
        encoding="utf-8",
    )  # fmt: skip
    html_written = bool(table) and drift.evidently_report(
        reference, current, REPORTS / "drift_report.html"
    )
    perf_file = REPORTS / "performance.json"
    perf = json.loads(perf_file.read_text(encoding="utf-8")) if perf_file.exists() else None
    page = status_page(run, table, html_written, perf)
    (REPORTS / "STATUS.md").write_text(page, encoding="utf-8")
    params, metrics, tags = tracking.flatten(run, table)
    if perf:
        tags["performance_level"] = perf["level"]
        for c in perf["checks"]:
            metrics[f"perf_{c['name']}"] = float(c["value"])
    tracking.log_run(cfg, run["model"]["model_version"], params, metrics, tags)
    log.info("Monitoring done. Drift: %s", {c: r["psi"] for c, r in table.items()})


if __name__ == "__main__":
    main()
