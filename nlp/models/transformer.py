"""Fine-tune DistilRoBERTa for NER, then score it on the test set like the baselines."""

import json
import logging

import pandas as pd
import torch
from datasets import Dataset
from seqeval.metrics import f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data.common.config import ROOT, load_settings
from data.common.logging_setup import setup_logging
from nlp.dataset import bio_to_spans
from nlp.evaluate import LABELS, span_scores
from nlp.models.baselines import read_split, summary

log = logging.getLogger("nlp.models.transformer")
ART = ROOT / "nlp" / "artifacts"
REP = ROOT / "nlp" / "reports"
TAGS = ["O"] + [f"{p}-{lab}" for lab in LABELS for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}


def encode(rows: list[dict], tok, max_length: int) -> Dataset:
    ds = Dataset.from_list(
        [{"tokens": r["tokens"], "tags": [TAG2ID[t] for t in r["tags"]]} for r in rows]
    )

    def align(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
        labels = []
        for i, tags in enumerate(batch["tags"]):
            previous, row = None, []
            for w in enc.word_ids(batch_index=i):
                row.append(-100 if w is None or w == previous else tags[w])
                previous = w
            labels.append(row)
        enc["labels"] = labels
        return enc

    return ds.map(align, batched=True, remove_columns=["tokens", "tags"])


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    true_tags, pred_tags = [], []
    for p_row, l_row in zip(preds, labels, strict=True):
        keep = l_row != -100
        true_tags.append([ID2TAG[int(x)] for x in l_row[keep]])
        pred_tags.append([ID2TAG[int(x)] for x in p_row[keep]])
    return {"f1": f1_score(true_tags, pred_tags)}


@torch.no_grad()
def predict_spans(model, tok, rows: list[dict], max_length: int) -> list[list[tuple]]:
    model.eval()
    out, truncated = [], 0
    for r in rows:
        enc = tok(
            r["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = model(**enc).logits[0].argmax(-1).tolist()
        tags, previous = ["O"] * len(r["tokens"]), None
        word_ids = enc.word_ids(0)
        for pos, w in enumerate(word_ids):
            if w is not None and w != previous:
                tags[w] = ID2TAG[ids[pos]]
            previous = w
        if max(w for w in word_ids if w is not None) < len(r["tokens"]) - 1:
            truncated += 1
        tokens = [(t, a, b) for t, (a, b) in zip(r["tokens"], r["offsets"], strict=True)]
        out.append(bio_to_spans(tokens, tags))
    if truncated:
        log.warning(
            "%d paragraphs were longer than max_length; their ends were not tagged", truncated
        )
    return out


def error_table(rows, gold, pred) -> pd.DataFrame:
    out = []
    for r, g, p in zip(rows, gold, pred, strict=True):
        g, p = set(g), set(p)
        for kind, spans in (("false_positive", p - g), ("false_negative", g - p)):
            for a, b, lab in sorted(spans):
                out.append(
                    {
                        "doc_id": r["doc_id"],
                        "kind": kind,
                        "label": lab,
                        "text": r["text"][a:b],
                        "context": r["text"][max(0, a - 60) : b + 60],
                    }
                )
    return pd.DataFrame(out)


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    cfg = s.nlp
    set_seed(cfg["random_seed"])
    train, dev, test = (read_split(n) for n in ("train", "dev", "test"))

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"], num_labels=len(TAGS), id2label=ID2TAG, label2id=TAG2ID
    )
    args = TrainingArguments(
        output_dir=str(ART / "ner_runs"),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        logging_steps=10,
        seed=cfg["random_seed"],
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encode(train, tok, cfg["max_length"]),
        eval_dataset=encode(dev, tok, cfg["max_length"]),
        data_collator=DataCollatorForTokenClassification(tok),
        processing_class=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(ART / "ner_model"))
    tok.save_pretrained(str(ART / "ner_model"))

    gold = [[tuple(x) for x in r["spans"]] for r in test]
    preds = predict_spans(trainer.model, tok, test, cfg["max_length"])
    results = json.loads((REP / "ner_baselines.json").read_text())
    results["transformer"] = span_scores(gold, preds)
    (REP / "ner_results.json").write_text(json.dumps(results, indent=2))
    error_table(test, gold, preds).to_csv(REP / "ner_errors_transformer.csv", index=False)
    log.info("\n%s", summary(results).to_string())
    log.info(
        "transformer per label: %s",
        {lab: round(v["f1"], 3) for lab, v in results["transformer"]["per_label"].items()},
    )


if __name__ == "__main__":
    main()
