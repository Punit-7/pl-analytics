from nlp.corpus import paragraphs

ARTICLE = {
    "title": "T",
    "revid": 1,
    "url": "u",
    "season": "2024/25",
    "team": "Arsenal",
    "text": "Intro "
    + "x" * 250
    + "\n== Season ==\n"
    + "y" * 250
    + "\nshort line\n== References ==\n"
    + "z" * 250,
}


def test_sections_and_skips():
    rows = paragraphs(ARTICLE, min_chars=200)
    assert [r["section"] for r in rows] == ["Introduction", "Season"]
    assert all(len(r["text"]) >= 200 for r in rows)


def test_doc_ids_are_unique():
    rows = paragraphs(ARTICLE, min_chars=200)
    assert len({r["doc_id"] for r in rows}) == len(rows)
