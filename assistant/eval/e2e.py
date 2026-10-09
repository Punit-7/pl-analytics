"""End-to-end review. --run writes the answers; you mark them; --score reads your marks."""

import argparse

import pandas as pd

from assistant.agent import build_agent
from assistant.eval.common import proportion, run_info, save_report
from data.common.config import ROOT, load_settings
from data.common.logging_setup import setup_logging

EVAL = ROOT / "assistant" / "eval"
QUESTIONS, ANSWERS = EVAL / "e2e_questions.csv", EVAL / "e2e_answers.csv"


def run(force: bool) -> None:
    if ANSWERS.exists() and not force:
        raise SystemExit(f"{ANSWERS.name} exists. Your marks are in it. Use --force to replace.")
    agent = build_agent()
    rows = []
    for q in pd.read_csv(QUESTIONS).itertuples():
        r = agent.ask(q.question)
        rows.append(
            {
                "id": q.id,
                "type": q.type,
                "question": q.question,
                "tool": r.tool or "none",
                "answer": r.answer,
                "sql": r.sql,
                "sources": " | ".join(f"[{h.rank}] {h.text[:200]}" for h in r.sources),
                "seconds": round(r.seconds, 1),
                "correct": "",
                "grounded": "",
            }
        )
        print(f"{q.id} {r.tool or 'none'} ({r.seconds:.0f} s): {r.answer[:90]}")
    pd.DataFrame(rows).to_csv(ANSWERS, index=False, encoding="utf-8-sig")
    print(f"\nNow open {ANSWERS.relative_to(ROOT)} and fill correct and grounded with y or n.")


def score() -> None:
    cfg = load_settings().assistant
    df = pd.read_csv(ANSWERS, encoding="utf-8-sig").fillna("")
    marks = {c: df[c].astype(str).str.strip().str.lower() for c in ("correct", "grounded")}
    if not marks["correct"].isin(["y", "n"]).all():
        raise SystemExit("Fill the correct column with y or n in every row first.")
    judged = marks["grounded"].isin(["y", "n"])  # leave grounded empty where it does not apply
    report = {
        **run_info(cfg, cfg["hosted_model"] if cfg["provider"] != "ollama" else cfg["chat_model"]),
        "answers_correct": proportion(int((marks["correct"] == "y").sum()), len(df)),
        "answers_grounded": proportion(
            int((marks["grounded"][judged] == "y").sum()), int(judged.sum())
        ),
        "correct_by_type": {
            t: proportion(int((marks["correct"][g.index] == "y").sum()), len(g))
            for t, g in df.groupby("type")
        },
        "mean_seconds": round(float(df["seconds"].mean()), 1),
    }
    save_report("e2e_review", report)
    print(
        "Correct:",
        report["answers_correct"]["value"],
        "Grounded:",
        report["answers_grounded"]["value"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    setup_logging(load_settings().logs)
    if args.run:
        run(args.force)
    elif args.score:
        score()
    else:
        parser.print_help()
