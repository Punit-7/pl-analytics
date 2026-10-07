"""Embed every paragraph in nlp.document and save the matrix: python -m assistant.index"""

import json
import logging
import time
from datetime import UTC, datetime

import numpy as np

from assistant.llm import LLM
from assistant.retrieval import ART, index_text, load_docs
from data.common.config import load_settings
from data.common.db import make_reader_engine
from data.common.logging_setup import setup_logging

log = logging.getLogger("assistant.index")
MAX_CHARS = 2000  # longer paragraphs are cut; the embedding model reads about 500 tokens


def embed_all(llm: LLM, texts: list[str], batch: int) -> np.ndarray:
    parts = []
    start = time.perf_counter()
    for i in range(0, len(texts), batch):
        parts.append(llm.embed(texts[i : i + batch]))
        if (i // batch) % 20 == 0:
            log.info("%d of %d paragraphs, %.0f s", i, len(texts), time.perf_counter() - start)
    matrix = np.vstack(parts).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)  # unit length, so a dot product is a cosine


def main() -> None:
    s = load_settings()
    setup_logging(s.logs)
    cfg = s.assistant
    docs = load_docs(make_reader_engine())
    texts = [index_text(r.article_title, r.section, r.text)[:MAX_CHARS] for r in docs.itertuples()]
    matrix = embed_all(LLM(cfg), texts, cfg["embed_batch"])
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "embeddings.npy", matrix)
    meta = {
        "doc_ids": list(docs["doc_id"]),
        "embed_model": cfg["embed_model"],
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (ART / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    log.info("Saved %d embeddings of %d numbers each", *matrix.shape)


if __name__ == "__main__":
    main()