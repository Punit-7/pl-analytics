"""Tag every paragraph with the fine-tuned model, link each mention, store in nlp.mention."""

import logging
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import text
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data.common.config import ROOT, load_settings
from data.common.db import make_engine
from data.common.logging_setup import setup_logging
from nlp.dataset import tokenize
from nlp.linking.linker import Linker
from nlp.models.gazetteer import load_kb
from nlp.models.transformer import predict_spans

log = logging.getLogger("nlp.run")
MODEL = ROOT / "nlp" / "artifacts" / "ner_model"


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    cfg = s.nlp
    engine = make_engine()
    docs = pd.read_sql(text("SELECT doc_id, season, team, text FROM nlp.document"), engine)
    tok = AutoTokenizer.from_pretrained(str(MODEL), add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL))
    rows = []
    for d in docs.itertuples():
        toks = tokenize(d.text)
        rows.append({"tokens": [t for t, _, _ in toks], "offsets": [[a, b] for _, a, b in toks]})
    spans = predict_spans(model, tok, rows, cfg["max_length"])

    linker = Linker(load_kb(engine), cfg["link_threshold"])
    now = datetime.now(UTC).replace(tzinfo=None)
    mentions = []
    for d, doc_spans in zip(docs.itertuples(), spans, strict=True):
        for a, b, label in doc_spans:
            entity_id, score = linker.resolve(d.text[a:b], label, d.season, d.team)
            mentions.append(
                {
                    "mention_id": f"{d.doc_id}:{a}",
                    "doc_id": d.doc_id,
                    "start_char": a,
                    "end_char": b,
                    "label": label,
                    "mention": d.text[a:b],
                    "entity_id": entity_id,
                    "link_score": score,
                    "model_version": cfg["model_version"],
                    "created_at": now,
                }
            )
    df = pd.DataFrame(mentions)
    with engine.begin() as conn:
        df.to_sql("mention", conn, schema="nlp", if_exists="replace", index=False)
        conn.execute(text("ALTER TABLE nlp.mention ADD PRIMARY KEY (mention_id)"))
        conn.execute(text(f"GRANT SELECT ON nlp.mention TO {s.reader_role}"))
    log.info(
        "%d mentions in %d paragraphs; %.0f%% linked",
        len(df),
        len(docs),
        100 * df["entity_id"].notna().mean(),
    )


if __name__ == "__main__":
    main()
