"""Execution accuracy of text-to-SQL: python -m assistant.eval.sql_eval --mode full"""

import argparse
import logging
from collections import Counter

import pandas as pd

from assistant.eval.common import REPORTS, proportion, run_info, same_result, save_report
from assistant.llm import LLM, cost_usd
from assistant.sql_runner import run_sql
from assistant.text2sql import MODES, Text2SQL, load_questions
from data.common.config import load_settings
from data.common.db import make_reader_engine
from data.common.logging_setup import setup_logging

log = logging.getLogger("assistant.eval.sql")


def judge(engine, cfg: dict, question: dict, answer) -> str:
    """One of: correct, wrong_result, sql_error, model_error."""
    if answer.error.startswith("model:"):
        return "model_error"
    if answer.result is None:
        return "sql_error"
    gold = run_sql(engine, question["gold_sql"], cfg["sql_timeout_ms"], 5000)
    return "correct" if same_result(gold.rows, answer.result.rows) else "wrong_result"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--limit", type=int, help="only the first N questions, for a quick try")
    parser.add_argument("--tag", help="name for the report files; defaults to the mode")
    args = parser.parse_args(argv)

    s = load_settings()
    setup_logging(s.logs)
    cfg = s.assistant
    engine, llm = make_reader_engine(), LLM(cfg)
    t2s = Text2SQL(llm, engine, cfg, mode=args.mode)
    questions = load_questions(args.split)[: args.limit]
    rows = []
    for i, q in enumerate(questions, start=1):
        answer = t2s.ask(q["question"])
        outcome = judge(engine, cfg, q, answer)
        log.info("%d/%d %s %s (%.0f s)", i, len(questions), q["id"], outcome, answer.seconds)
        rows.append(
            {
                "id": q["id"],
                "difficulty": q["difficulty"],
                "outcome": outcome,
                "attempts": answer.attempts,
                "seconds": round(answer.seconds, 1),
                "prompt_tokens": answer.prompt_tokens,
                "output_tokens": answer.output_tokens,
                "question": q["question"],
                "predicted_sql": answer.sql,
                "error": answer.error,
            }
        )
    df = pd.DataFrame(rows)
    correct = df["outcome"] == "correct"
    tag = args.tag or args.mode
    report = {
        **run_info(cfg, llm.model),
        "mode": args.mode,
        "split": args.split,
        "execution_accuracy": proportion(int(correct.sum()), len(df)),
        "by_difficulty": {
            d: proportion(
                int(correct[df["difficulty"] == d].sum()), int((df["difficulty"] == d).sum())
            )
            for d in ("easy", "medium", "hard")
        },
        "outcomes": dict(Counter(df["outcome"])),
        "mean_seconds": round(float(df["seconds"].mean()), 1),
        "mean_prompt_tokens": round(float(df["prompt_tokens"].mean())),
        "mean_output_tokens": round(float(df["output_tokens"].mean())),
        "cost_usd_per_question": round(
            cost_usd(int(df["prompt_tokens"].sum()), int(df["output_tokens"].sum()), cfg) / len(df),
            6,
        ),
    }
    save_report(f"sql_eval_{tag}", report)
    df.to_csv(REPORTS / f"sql_eval_{tag}.csv", index=False, encoding="utf-8")
    acc = report["execution_accuracy"]
    print(f"Execution accuracy ({args.mode}, {args.split}): {acc['k']}/{acc['n']} = {acc['value']}")
    print("By difficulty:", {d: v["value"] for d, v in report["by_difficulty"].items()})
    print("Outcomes:", report["outcomes"])


if __name__ == "__main__":
    main()
