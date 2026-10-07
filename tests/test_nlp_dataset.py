import json

import pytest

from data.common.config import ROOT
from nlp.dataset import bio_to_spans, clean_span, to_bio, tokenize
from nlp.evaluate import span_scores

TEXT = "Saka's goal at Anfield gave Arsenal the lead."
SPANS = [(0, 4, "PLAYER"), (15, 22, "VENUE"), (28, 35, "TEAM")]


def test_bio_round_trip():
    tokens = tokenize(TEXT)
    tags, misaligned = to_bio(tokens, SPANS)
    assert misaligned == 0
    assert tags[:3] == ["B-PLAYER", "O", "O"]  # Saka ' s
    assert bio_to_spans(tokens, tags) == SPANS


def test_clean_span_applies_boundary_rules():
    text = "the Saints beat Spurs' rivals at St Mary's."
    a, b = clean_span(text, 0, 10, "TEAM")
    assert text[a:b] == "Saints"
    i = text.index("Spurs'")
    a, b = clean_span(text, i, i + 6, "TEAM")
    assert text[a:b] == "Spurs"
    i = text.index("St Mary's")
    a, b = clean_span(text, i, i + 10, "VENUE")  # includes the full stop
    assert text[a:b] == "St Mary's"


def test_perfect_prediction_scores_one():
    assert span_scores([SPANS], [SPANS])["micro"]["f1"] == 1.0


def test_no_article_in_two_splits():
    articles = {}
    for name in ("train", "dev", "test"):
        path = ROOT / "nlp" / "data" / "splits" / f"{name}.jsonl"
        if not path.exists():
            pytest.skip("run python -m nlp.dataset first")
        articles[name] = {json.loads(line)["article"] for line in open(path, encoding="utf-8")}
    assert not articles["train"] & articles["test"]
    assert not articles["train"] & articles["dev"]
    assert not articles["dev"] & articles["test"]
