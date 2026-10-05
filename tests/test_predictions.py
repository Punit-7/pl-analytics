import pytest
from sqlalchemy import inspect, text

from data.common.db import make_engine


def test_every_prediction_was_made_before_its_match_day():
    engine = make_engine()
    if not inspect(engine).has_table("match_prediction", schema="predictions"):
        pytest.skip("no predictions published yet")
    with engine.connect() as conn:
        late = conn.execute(
            text("""
            SELECT prediction_id FROM predictions.match_prediction
            WHERE CAST(created_at AS DATE) >= match_date""")
        ).fetchall()
    assert late == [], f"predictions made on or after match day: {late[:5]}"
