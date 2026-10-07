"""Compare the three search methods: python -m assistant.eval.rag_eval"""

import logging

import pandas as pd

from assistant.eval.common import run_info, save_report
from assistant.eval.make_rag_questions import OUT as QUESTIONS
from assistant.llm import LLM
from assistant.retrieval import MODES, make_retriever
from data.common.config import load_settings
from data.common.db import make_reader_engine
from data.common.logging_setup import setup_logging

log = logging.getLogger("assistant.eval.rag")
DEPTH = 10  # how far down the results we look for the right paragraph


def rank_of(doc_id: str, hits: list) -> int | None:
    """Position (1 = first) of the paragraph the question was written from, if it was found."""
    for h in hits:
        if h.doc_id == doc_id:
            return h.rank
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "recall_at_1": round(sum(r == 1 for r in ranks) / n, 3),
        "recall_at_5": round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r is not None) / n, 3),
    }


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    cfg = s.assistant
    questions = pd.read_csv(QUESTIONS, encoding="utf-8-sig")
    questions = questions[questions["keep"].astype(str).str.strip().str.lower() == "y"]
    llm = LLM(cfg)
    retriever = make_retriever(make_reader_engine(), llm, cfg)
    modes = MODES if retriever.embeddings is not None else ("bm25",)
    results = {}
    for mode in modes:
        ranks = [
            rank_of(q.doc_id, retriever.search(q.question, mode=mode, top_k=DEPTH))
            for q in questions.itertuples()
        ]
        results[mode] = metrics(ranks)
        log.info("%s: %s", mode, results[mode])
    save_report("rag_eval", {**run_info(cfg, cfg["embed_model"]), "n": len(questions), **results})
    print(pd.DataFrame(results).T.to_string())


if __name__ == "__main__":
    main()
