import pytest
from sqlalchemy import inspect, text

from data.common.db import make_engine


@pytest.fixture(scope="module")
def con():
    engine = make_engine()
    if not inspect(engine).has_table("kb_entity", schema="nlp"):
        pytest.skip("run python -m nlp.kb first")
    with engine.connect() as c:
        yield c


def test_all_three_types_present(con):
    types = {r[0] for r in con.execute(text("SELECT DISTINCT entity_type FROM nlp.kb_entity"))}
    assert types == {"PLAYER", "TEAM", "VENUE"}


def test_no_empty_aliases(con):
    n = con.execute(text("SELECT COUNT(*) FROM nlp.kb_entity WHERE alias IS NULL OR alias = ''"))
    assert n.scalar_one() == 0