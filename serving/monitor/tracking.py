"""Record each weekly run in MLflow, so that the numbers can be charted over time."""

from __future__ import annotations

import os
from pathlib import Path

from data.common.config import ROOT


def flatten(run: dict, drift: dict) -> tuple[dict, dict, dict]:
    """Turn the run report into MLflow's three kinds of record: params, metrics and tags."""
    model, data, gate = run["model"], run["data"], run.get("gate") or {}
    params = {**model["config"], "config_id": model["config_id"]}
    metrics = {
        "n_matches": data["n_matches"],
        "days_since_latest_result": data["days_since_latest"],
        "n_predictions": run["published"]["n_predictions"],
    }
    for key, value in run["ledger"].items():
        metrics[f"live_{key}"] = value
    for key in ("champion_rps", "challenger_rps", "gain"):
        if gate.get(key) is not None:
            metrics[f"gate_{key}"] = gate[key]
    for column, row in drift.items():
        metrics[f"psi_{column}"] = row["psi"]
    if drift:
        metrics["psi_max"] = max(row["psi"] for row in drift.values())
    tags = {
        "model_version": model["model_version"],
        "trained_through": model["trained_through"],
        "data_stale": str(data["stale"]),
        "gate": "not run" if not gate else ("promoted" if gate["promoted"] else "kept champion"),
        "drift_alert": str(any(row["alert"] for row in drift.values())),
    }
    return params, {k: float(v) for k, v in metrics.items()}, tags


def log_run(cfg: dict, run_name: str, params: dict, metrics: dict, tags: dict) -> None:
    import mlflow  # imported here so that the rest of the code works without MLflow installed

    uri = os.getenv("MLFLOW_TRACKING_URI") or cfg["mlflow_uri"]  # the workflow sets the variable
    if uri.startswith("sqlite:///"):
        (ROOT / Path(uri.removeprefix("sqlite:///")).parent).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)