from assistant.eval.compare import compare


def accuracy(value, low, model="llama3.2:3b"):
    return {"model": model, "execution_accuracy": {"value": value, "low_95": low}}


def test_drop_below_the_old_interval_is_flagged():
    r = compare(accuracy(0.90, 0.80), accuracy(0.75, 0.6), ("execution_accuracy",))
    assert r["dropped"] and r["limit"] == 0.80


def test_small_drop_inside_the_interval_is_not_flagged():
    r = compare(accuracy(0.90, 0.80), accuracy(0.85, 0.7), ("execution_accuracy",))
    assert not r["dropped"]


def test_plain_number_uses_the_fixed_tolerance():
    old, new = {"hybrid": {"recall_at_5": 0.80}}, {"hybrid": {"recall_at_5": 0.74}}
    r = compare(old, new, ("hybrid", "recall_at_5"))
    assert r["limit"] == 0.75 and r["dropped"]


def test_a_different_model_is_reported():
    r = compare(accuracy(0.9, 0.8), accuracy(0.9, 0.8, "other"), ("execution_accuracy",))
    assert r["model_changed"] and not r["dropped"]


def test_a_perfect_score_still_allows_one_miss():
    old = {"tool_selection_accuracy": {"value": 1.0, "low_95": 1.0}}
    one_miss = {"tool_selection_accuracy": {"value": 0.967, "low_95": 0.9}}
    r = compare(old, one_miss, ("tool_selection_accuracy",))
    assert r["limit"] == 0.95 and not r["dropped"]
