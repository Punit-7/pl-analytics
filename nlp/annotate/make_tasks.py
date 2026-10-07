"""Assign articles to splits, sample paragraphs, and pre-annotate train/dev tasks only."""
import argparse
import json
import logging
import math

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from nlp.models.gazetteer import Gazetteer, gazetteer_terms, load_kb

log = logging.getLogger("nlp.annotate.make_tasks")
LABELLED = ROOT / "nlp" / "data" / "labelled"
SPLITS = ROOT / "nlp" / "data" / "splits"


def assign_splits(articles: list[str], test_share: float, dev_share: float, seed: int) -> dict:
    order = np.random.default_rng(seed).permutation(sorted(articles))
    n_test, n_dev = round(len(order) * test_share), round(len(order) * dev_share)
    return {a: "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
            for i, a in enumerate(order)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recheck", type=int, help="write N tasks for the self-agreement check")
    args = parser.parse_args(argv)
    s = load_settings()
    setup_logging(s.logs)
    cfg, seed = s.nlp, s.nlp["random_seed"]

    if args.recheck:
        tasks = json.loads((LABELLED / "tasks.json").read_text(encoding="utf-8"))
        rng = np.random.default_rng(seed + 1)
        picked = [tasks[i] for i in rng.choice(len(tasks), size=args.recheck, replace=False)]
        recheck = [{"data": t["data"]} for t in picked]  # no suggestions
        (LABELLED / "recheck_tasks.json").write_text(
            json.dumps(recheck, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("Wrote %d recheck tasks", len(recheck))
        return

    engine = make_engine()
    docs = pd.read_sql(text("SELECT doc_id, article_title, season, text FROM nlp.document"), engine)
    split = assign_splits(docs["article_title"].unique().tolist(),
                          cfg["test_share"], cfg["dev_share"], seed)
    docs["split"] = docs["article_title"].map(split)
    SPLITS.mkdir(parents=True, exist_ok=True)
    splits = pd.Series(split, name="split").rename_axis("article_title")
    splits.to_csv(SPLITS / "article_splits.csv")

    per_season = math.ceil(cfg["label_target"] / docs["season"].nunique())
    sample = docs.sample(frac=1, random_state=seed).groupby("season").head(per_season)
    sample = sample.sample(n=min(cfg["label_target"], len(sample)), random_state=seed)

    gaz = Gazetteer(gazetteer_terms(load_kb(engine)))
    tasks = []
    for r in sample.itertuples():
        task = {"data": {"text": r.text, "doc_id": r.doc_id, "split": r.split,
                         "article": r.article_title}}
        if r.split != "test":  # test paragraphs are labelled from scratch
            task["predictions"] = [{"model_version": "gazetteer-1", "result": [
                {"from_name": "label", "to_name": "text", "type": "labels",
                 "value": {"start": a, "end": b, "text": r.text[a:b], "labels": [lab]}}
                for a, b, lab in gaz.predict(r.text)]}]
        tasks.append(task)
    LABELLED.mkdir(parents=True, exist_ok=True)
    (LABELLED / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    log.info("Wrote %d tasks: %s", len(tasks), sample["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()