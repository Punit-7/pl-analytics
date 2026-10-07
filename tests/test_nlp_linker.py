import pandas as pd

from nlp.linking.linker import Linker, normalise

KB = pd.DataFrame([
    ("player:1", "PLAYER", "Ben White", "Ben White", "full", "2024/25", "Arsenal", 2000),
    ("player:1", "PLAYER", "Ben White", "White", "surname", "2024/25", "Arsenal", 2000),
    ("player:2", "PLAYER", "Kyle White", "White", "surname", "2024/25", "Fulham", 900),
    ("team:Newcastle", "TEAM", "Newcastle", "Newcastle United", "alias", None, "Newcastle", None),
    ("team:Man United", "TEAM", "Man United", "United", "alias", None, "Man United", None),
], columns=["entity_id", "entity_type", "name", "alias", "alias_kind", "season", "team", "minutes"])


def test_normalise_removes_accents():
    assert normalise("Rúben Dias") == "ruben dias"


def test_shared_surname_uses_article_club():
    linker = Linker(KB, threshold=85)
    assert linker.resolve("White", "PLAYER", "2024/25", "Fulham")[0] == "player:2"


def test_shared_surname_without_context_is_nil():
    linker = Linker(KB, threshold=85)
    assert linker.resolve("White", "PLAYER", "2024/25", "Chelsea")[0] is None


def test_united_means_the_articles_club():
    linker = Linker(KB, threshold=85)
    assert linker.resolve("United", "TEAM", None, "Newcastle")[0] == "team:Newcastle"