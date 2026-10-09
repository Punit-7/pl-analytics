"""The weekly job: check the data, run the gate, refit, publish, score.

python -m serving.weekly                 the full job, as GitHub Actions runs it
python -m serving.weekly --skip-ingest   use the CSV files already on disk
"""

import argparse
import json
import logging
from datetime import UTC, date, datetime

import pandas as pd

from data.common.config import ROOT, load_settings
from data.common.logging_setup import setup_logging
from serving import ledger, registry
from serving.data import load_results, validate
from serving.gate import fit, run_gate
from serving.model_io import save_model
from serving.publish import NotReady, publish

log = logging.getLogger("serving.weekly")
MODEL = ROOT / "serving" / "models" / "match_model.json"
RUN = ROOT / "serving" / "reports" / "run.json"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Weekly retrain and publish")
    parser.add_argument("--skip-ingest", action="store_true", help="do not download results")
    parser.add_argument("--skip-gate", action="store_true", help="do not test the challenger")
    parser.add_argument("--as-of", help="pretend today is this date (YYYY-MM-DD), for testing")
    args = parser.parse_args(argv)

    s = load_settings()
    setup_logging(s.logs)
    cfg = s.serving
    now = datetime.now(UTC).replace(microsecond=0)
    if args.as_of:
        now = datetime.fromisoformat(args.as_of).replace(tzinfo=UTC)
    today: date = now.date()

    # 1. Data: download, load, check. A failed check stops the job here.
    results = load_results(s, ingest=not args.skip_ingest, today=today)
    results = results[results["match_date"] < pd.Timestamp(today)].copy()
    stats = validate(results, s, today)
    results[["home_goals", "away_goals"]] = results[["home_goals", "away_goals"]].astype(int)
    season = stats["season"]
    log.info("Data: %s", stats)

    # 2. Champion/challenger gate.
    champion = registry.load_champion(s)
    challenger = registry.challenger_config(s, champion)
    gate = None
    if challenger and not args.skip_gate:
        decision = run_gate(results, champion, challenger, today, cfg)
        gate = {
            k: round(v, 5) if isinstance(v, float) else v for k, v in decision.as_dict().items()
        }
        registry.record(
            {
                **gate,
                "date": str(today),
                "champion_id": registry.config_id(champion),
                "challenger_id": registry.config_id(challenger),
            }
        )
        if decision.promoted:
            champion = challenger
            registry.save_champion(champion, today)

    # 3. Refit the live model with the champion configuration on everything before today.
    model = fit(results, champion, today)
    version = f"dc-{today.isoformat()}-{registry.config_id(champion)}"
    meta = {
        "model_version": version,
        "fitted_on": str(today),
        "trained_through": stats["latest_match_date"],
        "config": champion,
        "config_id": registry.config_id(champion),
    }
    save_model(MODEL, model, meta)

    # 4. Publish predictions for every remaining match, and the season simulation.
    try:
        published = publish(model, results, season, now, version, cfg)
    except NotReady as e:
        log.warning("Nothing published: %s", e)
        published = {"n_predictions": 0}

    # 5. Score every earlier prediction whose match now has a result.
    summary = ledger.update(results)

    RUN.parent.mkdir(parents=True, exist_ok=True)
    run = {"run_at": now.isoformat(), "data": stats, "model": meta, "gate": gate,
           "published": published, "ledger": summary}  # fmt: skip
    RUN.write_text(json.dumps(run, indent=1), encoding="utf-8")
    log.info("Done: %s, %d predictions published", version, published["n_predictions"])


if __name__ == "__main__":
    main()
