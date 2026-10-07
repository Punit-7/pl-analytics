"""Search the article paragraphs: BM25 keywords, embeddings, and both combined."""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from data.common.config import ROOT

log = logging.getLogger(__name__)
ART = ROOT / "assistant" / "artifacts"
TOKEN = re.compile(r"[a-z0-9]+")
# Letters that NFKD does not split into a base letter and an accent (the lesson from P3).
FOLD = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "ł": "l", "đ": "d"})
MODES = ("bm25", "dense", "hybrid")
DEPTH = 50  # how many results each method passes to the fusion


def tokenize(s: str) -> list[str]:
    """Lower-case words and numbers with accents removed: 'Ødegaard's' -> ['odegaard', 's']."""
    s = unicodedata.normalize("NFKD", s.translate(FOLD))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return TOKEN.findall(s.casefold())


class BM25:
    """Okapi BM25 keyword scoring, written out so every number can be checked by hand."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float64)
        self.avg_len = float(self.lengths.mean()) if self.n else 0.0
        found = defaultdict(list)  # term -> [(document number, times the term appears)]
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                found[term].append((i, tf))
        self.postings = {
            term: (np.array([i for i, _ in p]), np.array([tf for _, tf in p], dtype=np.float64))
            for term, p in found.items()
        }

    def idf(self, term: str) -> float:
        df = len(self.postings[term][0]) if term in self.postings else 0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n)
        for term in set(query):
            if term not in self.postings:
                continue
            idx, tf = self.postings[term]
            length_part = 1 - self.b + self.b * self.lengths[idx] / self.avg_len
            out[idx] += self.idf(term) * tf * (self.k1 + 1) / (tf + self.k1 * length_part)
        return out


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: each list adds 1 / (k + rank) to a document's score."""
    score = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] += 1 / (k + rank)
    return sorted(score, key=lambda d: (-score[d], d))


def index_text(title: str, section: str, body: str) -> str:
    """The text that is searched: the article title and section give a paragraph its context."""
    return f"{title}. {section}. {body}"


@dataclass
class Hit:
    rank: int
    doc_id: str
    title: str
    section: str
    season: str
    team: str
    url: str
    text: str


class Retriever:
    def __init__(
        self,
        docs: pd.DataFrame,
        embeddings: np.ndarray | None,
        embed: Callable[[list[str]], np.ndarray] | None,
        cfg: dict,
    ):
        self.docs = docs.reset_index(drop=True)
        self.embeddings, self.embed, self.cfg = embeddings, embed, cfg
        texts = [index_text(r.article_title, r.section, r.text) for r in self.docs.itertuples()]
        self.bm25 = BM25([tokenize(t) for t in texts], cfg["bm25_k1"], cfg["bm25_b"])

    def _ranking(self, scores: np.ndarray, allowed: np.ndarray) -> list[int]:
        scores = np.where(allowed, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:DEPTH]
        return [int(i) for i in order if scores[i] > 0]

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        season: str = "",
        team: str = "",
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "bm25" and self.embeddings is None:
            log.warning("No embedding index: using BM25 only. Run python -m assistant.index")
            mode = "bm25"
        allowed = np.ones(len(self.docs), dtype=bool)
        if season:
            allowed &= (self.docs["season"] == season).to_numpy()
        if team:
            allowed &= (self.docs["team"] == team).to_numpy()
        rankings = []
        if mode in ("bm25", "hybrid"):
            rankings.append(self._ranking(self.bm25.scores(tokenize(query)), allowed))
        if mode in ("dense", "hybrid"):
            q = self.embed([query])[0]
            q = q / (np.linalg.norm(q) or 1.0)
            rankings.append(self._ranking(self.embeddings @ q, allowed))
        order = rankings[0] if len(rankings) == 1 else rrf(rankings, self.cfg["rrf_k"])
        hits = []
        for rank, i in enumerate(order[: top_k or self.cfg["top_k"]], start=1):
            r = self.docs.iloc[i]
            hits.append(
                Hit(rank, r.doc_id, r.article_title, r.section, r.season, r.team, r.url, r.text)
            )
        return hits


def load_docs(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(
        text("""
        SELECT doc_id, article_title, url, season, team, section, text
        FROM nlp.document ORDER BY doc_id"""),
        engine,
    )


def load_embeddings(docs: pd.DataFrame, cfg: dict) -> np.ndarray | None:
    """The saved embedding matrix, or None if it is missing or does not match the documents."""
    meta_file, matrix_file = ART / "index_meta.json", ART / "embeddings.npy"
    if not (meta_file.exists() and matrix_file.exists()):
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta["doc_ids"] != list(docs["doc_id"]) or meta["embed_model"] != cfg["embed_model"]:
        log.warning("The embedding index is out of date. Run python -m assistant.index")
        return None
    return np.load(matrix_file)


def make_retriever(engine: Engine, llm, cfg: dict) -> Retriever:
    docs = load_docs(engine)
    return Retriever(docs, load_embeddings(docs, cfg), llm.embed, cfg)


if __name__ == "__main__":
    import sys

    from assistant.llm import LLM
    from data.common.config import load_settings
    from data.common.db import make_reader_engine

    cfg = load_settings().assistant
    retriever = make_retriever(make_reader_engine(), LLM(cfg), cfg)
    query = " ".join(sys.argv[1:]) or "Arsenal title race"
    for mode in MODES if retriever.embeddings is not None else ("bm25",):
        print(f"\n{mode}")
        for h in retriever.search(query, mode=mode):
            print(f"  {h.rank}. {h.title} ({h.section}): {h.text[:90]}")
