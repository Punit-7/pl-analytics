"""Run every gold query once: python -m assistant.eval.check_gold"""

import json
from pathlib import Path

from assistant.sql_runner import SQLFailed, run_sql
from data.common.config import load_settings
from data.common.db import make_reader_engine

QUESTIONS = Path(__file__).with_name("sql_questions.jsonl")


def load_questions() -> list[dict]:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    cfg = load_settings().assistant
    engine = make_reader_engine()
    problems = 0
    for q in load_questions():
        try:
            result = run_sql(engine, q["gold_sql"], cfg["sql_timeout_ms"], 5000)
        except SQLFailed as e:
            print(f"{q['id']}  ERROR  {e}")
            problems += 1
            continue
        empty = not result.rows or result.rows[0][0] is None
        problems += empty
        first = result.rows[0] if result.rows else ""
        print(f"{q['id']}  {'EMPTY' if empty else 'ok':5}  {len(result.rows):3} row(s)  {first}")
    print(f"\n{problems} problem(s)")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()