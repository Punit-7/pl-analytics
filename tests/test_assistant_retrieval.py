import numpy as np
import pandas as pd
import pytest

from assistant.retrieval import BM25, Retriever, rrf, tokenize

CFG = {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60, "top_k": 2}
DOCS = pd.DataFrame(
    [
        ("1-0", "2023–24 Arsenal F.C. season", "u1", "2023/24", "Arsenal", "August",
         "Arsenal beat Chelsea at the Emirates Stadium."),
        ("2-0", "2023–24 Chelsea F.C. season", "u2", "2023/24", "Chelsea", "August",
         "Chelsea drew with Liverpool at Stamford Bridge."),
        ("3-0", "2022–23 Arsenal F.C. season", "u3", "2022/23", "Arsenal", "May",
         "Ødegaard scored twice as the title race ended."),
    ],
    columns=["doc_id", "article_title", "url", "season", "team", "section", "text"],
)  # fmt: skip


def fake_embed(texts):
    """A stand-in embedding: counts of the letters a to z. Enough to test the plumbing."""
    rows = [[t.lower().count(chr(97 + i)) for i in range(26)] for t in texts]
    return np.array(rows, dtype=np.float32)


def make():
    texts = [f"{r.article_title}. {r.section}. {r.text}" for r in DOCS.itertuples()]
    matrix = fake_embed(texts)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return Retriever(DOCS, matrix, fake_embed, CFG)


def test_tokenize_folds_accents_and_special_letters():
    assert tokenize("Ødegaard's 2–1 Guéhi") == ["odegaard", "s", "2", "1", "guehi"]


def test_bm25_matches_hand_calculation():
    docs = [["arsenal", "beat", "chelsea"], ["chelsea", "drew"],
            ["liverpool", "won", "the", "title", "today"]]  # fmt: skip
    bm25 = BM25(docs, k1=1.5, b=0.75)
    assert bm25.idf("arsenal") == pytest.approx(0.98083, abs=1e-5)
    scores = bm25.scores(["arsenal"])
    assert scores[0] == pytest.approx(1.02705, abs=1e-5)
    assert scores[1] == 0 and scores[2] == 0


def test_rrf_matches_hand_calculation():
    assert rrf([[1, 2, 3], [2, 3, 4]], k=60) == [2, 3, 1, 4]


def test_keyword_search_finds_a_name_written_without_its_special_letter():
    hits = make().search("Odegaard title race", mode="bm25")
    assert hits[0].doc_id == "3-0"


def test_filters_keep_only_the_wanted_club_and_season():
    hits = make().search("Arsenal", mode="hybrid", team="Arsenal", season="2023/24")
    assert [h.doc_id for h in hits] == ["1-0"]


def test_missing_index_falls_back_to_keywords():
    retriever = Retriever(DOCS, None, fake_embed, CFG)
    assert retriever.search("Stamford Bridge", mode="hybrid")[0].doc_id == "2-0"


def test_citation_check():
    rag = pytest.importorskip("assistant.rag")  # created in Stage 9
    NO_ANSWER, check_citations = rag.NO_ANSWER, rag.check_citations
    assert check_citations("Arsenal won [1] and drew [2].", 2) == ([1, 2], True)
    assert check_citations("Arsenal won [3].", 2) == ([3], False)
    assert check_citations("Arsenal won.", 2) == ([], False)
    assert check_citations(NO_ANSWER, 2) == ([], True)
