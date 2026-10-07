"""Write one test question per sampled paragraph: python -m assistant.eval.make_rag_questions"""

import argparse
import logging

import pandas as pd

from assistant.llm import LLM, LLMError
from assistant.retrieval import load_docs
from data.common.config import ROOT, load_settings
from data.common.db import make_reader_engine
from data.common.logging_setup import setup_logging

log = logging.getLogger("assistant.eval.rag_questions")
OUT = ROOT / "assistant" / "eval" / "rag_questions.csv"
SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}
SYSTEM = """Write one question that the paragraph answers.
- The question must name the club and the season, because the paragraph may not.
- It must be answerable from the paragraph alone.
- Do not copy a full sentence from the paragraph.
Reply with JSON of the form {"question": "..."}."""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args(argv)
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT.name} exists. Your review marks are in it. Use --force to replace.")

    s = load_settings()
    setup_logging(s.logs)
    cfg = s.assistant
    docs = load_docs(make_reader_engine())
    docs = docs[docs["text"].str.len() >= 300]  # short paragraphs make vague questions
    sample = docs.sample(n=min(args.n, len(docs)), random_state=cfg["seed"])
    llm = LLM(cfg)
    rows = []
    for i, d in enumerate(sample.itertuples(), start=1):
        user = f"Club: {d.team}\nSeason: {d.season}\nParagraph: {d.text[:1200]}"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            data, _ = llm.chat_json(messages, SCHEMA)
        except LLMError as e:
            log.warning("Skipped %s: %s", d.doc_id, e)
            continue
        rows.append(
            {
                "doc_id": d.doc_id,
                "team": d.team,
                "season": d.season,
                "question": str(data.get("question", "")).strip(),
                "keep": "y",
                "paragraph": d.text[:300],
            }
        )
        log.info("%d/%d %s", i, len(sample), rows[-1]["question"])
    if not rows:
        raise SystemExit("No questions were written. Check the log above.")
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows)} questions to {OUT.relative_to(ROOT)}. Review the keep column.")


if __name__ == "__main__":
    main()
