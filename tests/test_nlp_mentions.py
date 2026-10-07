import pytest
from sqlalchemy import inspect, text

from data.common.db import make_engine


@pytest.fixture(scope="module")
def con():
    engine = make_engine()
    if not inspect(engine).has_table("mention", schema="nlp"):
        pytest.skip("run python -m nlp.run first")
    with engine.connect() as c:
        yield c


def test_offsets_point_at_the_mention(con):
    bad = con.execute(text("""
        SELECT m.mention_id FROM nlp.mention m JOIN nlp.document d USING (doc_id)
        WHERE SUBSTRING(d.text FROM m.start_char + 1 FOR m.end_char - m.start_char) <> m.mention
        LIMIT 5""")).fetchall()
    assert bad == []


def test_linked_ids_exist_in_kb(con):
    bad = con.execute(text("""
        SELECT DISTINCT m.entity_id FROM nlp.mention m
        LEFT JOIN nlp.kb_entity k ON k.entity_id = m.entity_id
        WHERE m.entity_id IS NOT NULL AND k.entity_id IS NULL LIMIT 5""")).fetchall()
    assert bad == []