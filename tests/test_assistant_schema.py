import pytest

from assistant.schema import NOTES, live_columns, schema_card
from data.common.db import make_reader_engine


@pytest.fixture(scope="module")
def engine():
    try:
        e = make_reader_engine()
        live_columns(e, ["mart.fact_match"])
    except Exception as err:
        pytest.skip(f"no read-only database connection: {err}")
    return e


def test_notes_describe_exactly_the_columns_in_the_database(engine):
    live = live_columns(engine, list(NOTES))
    for table, notes in NOTES.items():
        assert [name for name, _ in live[table]], f"{table} is missing or not readable"
        assert set(notes["columns"]) == {name for name, _ in live[table]}, table


def test_card_names_real_teams_and_is_small_enough(engine):
    card = schema_card(engine, list(NOTES))
    assert "Man United" in card and "Nott'm Forest" in card
    assert len(card) / 4 < 2500  # rough token count; must leave room in a 4,096-token window
