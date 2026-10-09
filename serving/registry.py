"""The champion configuration and the history of every champion/challenger decision."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date

from data.common.config import ROOT, Settings

REG = ROOT / "serving" / "state" / "registry"
CHAMPION, HISTORY = REG / "champion.json", REG / "history.csv"
KEYS = ("dc_xi", "dc_history_days", "dc_max_goals", "fit_rho")
COLUMNS = [
    "date",
    "champion_id",
    "challenger_id",
    "n_matches",
    "champion_rps",
    "challenger_rps",
    "gain",
    "gain_low_95",
    "gain_high_95",
    "promoted",
    "reason",
]


def config_id(config: dict) -> str:
    """A short, stable name for a configuration: the first 8 hex digits of its SHA-1 hash."""
    text = json.dumps({k: config[k] for k in KEYS}, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def default_config(s: Settings) -> dict:
    """P2's settings: the first champion."""
    m = s.modelling
    return {"dc_xi": m["dc_xi"], "dc_history_days": m["dc_history_days"],
            "dc_max_goals": m["dc_max_goals"], "fit_rho": True}  # fmt: skip


def load_champion(s: Settings) -> dict:
    if not CHAMPION.exists():
        save_champion(default_config(s), date.today())
    return json.loads(CHAMPION.read_text(encoding="utf-8"))["config"]


def save_champion(config: dict, since: date) -> None:
    REG.mkdir(parents=True, exist_ok=True)
    record = {"config": config, "config_id": config_id(config), "since": str(since)}
    CHAMPION.write_text(json.dumps(record, indent=1), encoding="utf-8")


def challenger_config(s: Settings, champion: dict) -> dict | None:
    """The champion's values with [serving.challenger] on top. None if nothing differs."""
    candidate = {**champion, **s.serving.get("challenger", {})}
    unknown = set(candidate) - set(KEYS)
    if unknown:
        raise ValueError(f"unknown keys in [serving.challenger]: {sorted(unknown)}")
    return None if config_id(candidate) == config_id(champion) else candidate


def record(row: dict) -> None:
    """Append one decision to history.csv."""
    REG.mkdir(parents=True, exist_ok=True)
    new = not HISTORY.exists()
    with HISTORY.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in COLUMNS})
