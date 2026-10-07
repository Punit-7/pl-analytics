from datetime import date
from decimal import Decimal

from assistant.eval.common import proportion, same_result
from assistant.sql_runner import check_sql
from assistant.text2sql import load_questions


def test_same_rows_in_another_order_match():
    assert same_result([("Arsenal", 89), ("Chelsea", 63)], [("Chelsea", 63), ("Arsenal", 89)])


def test_number_types_and_column_order_do_not_matter():
    assert same_result([(Decimal("2.50"), "Home")], [("Home", 2.5)])
    assert same_result([(date(2024, 5, 19),)], [("2024-05-19",)])


def test_different_values_or_shapes_do_not_match():
    assert not same_result([("Arsenal", 89)], [("Arsenal", 88)])
    assert not same_result([("Arsenal", 89)], [("Arsenal",)])
    assert not same_result([("Arsenal",)], [("Arsenal",), ("Arsenal",)])


def test_proportion_matches_hand_calculation():
    p = proportion(26, 40)
    assert p["value"] == 0.65 and p["standard_error"] == 0.075
    assert (p["low_95"], p["high_95"]) == (0.502, 0.798)


def test_question_file_is_well_formed():
    questions = load_questions()
    assert len({q["id"] for q in questions}) == len(questions)
    assert sum(q["split"] == "test" for q in questions) == 40
    assert sum(q["split"] == "dev" for q in questions) == 10
    for q in questions:
        assert q["difficulty"] in ("easy", "medium", "hard")
        check_sql(q["gold_sql"])  # every gold query is a single SELECT
