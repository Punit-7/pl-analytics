"""Score the two baselines on the test set: gazetteer rules and spaCy's small model."""
import json
import logging

import pandas as pd
import spacy

from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from nlp.evaluate import span_scores
from nlp.models.gazetteer import Gazetteer, gazetteer_terms, load_kb

log = logging.getLogger("nlp.models.baselines")
SPLITS = ROOT / "nlp" / "data" / "splits"
REP = ROOT / "nlp" / "reports"
SPACY_MAP = {"PERSON": "PLAYER", "ORG": "TEAM", "FAC": "VENUE"}


def read_split(name: str) -> list[dict]:
    with open(SPLITS / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def spacy_predict(nlp, texts: list[str]) -> list[list[tuple]]:
    return [[(e.start_char, e.end_char, SPACY_MAP[e.label_])
             for e in doc.ents if e.label_ in SPACY_MAP]
            for doc in nlp.pipe(texts, batch_size=32)]


def summary(results: dict) -> pd.DataFrame:
    return pd.DataFrame({name: {"precision": r["micro"]["precision"], "recall": r["micro"]["recall"],
                                "f1": r["micro"]["f1"], "macro_f1": r["macro_f1"]}
                         for name, r in results.items()}).T.round(3)


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    REP.mkdir(parents=True, exist_ok=True)
    test = read_split("test")
    texts = [r["text"] for r in test]
    gold = [[tuple(x) for x in r["spans"]] for r in test]

    gaz = Gazetteer(gazetteer_terms(load_kb(make_engine())))
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    preds = {"gazetteer": [gaz.predict(t) for t in texts], "spacy_sm": spacy_predict(nlp, texts)}

    results = {name: span_scores(gold, p) for name, p in preds.items()}
    (REP / "ner_baselines.json").write_text(json.dumps(results, indent=2))
    with open(REP / "pred_baselines_test.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(test):
            f.write(json.dumps({"doc_id": r["doc_id"], "gold": gold[i],
                                **{name: p[i] for name, p in preds.items()}}) + "\n")
    log.info("%d test paragraphs\n%s", len(test), summary(results).to_string())
    for name, r in results.items():
        log.info("%s per label: %s", name,
                 {lab: round(v["f1"], 3) for lab, v in r["per_label"].items()})


if __name__ == "__main__":
    main()