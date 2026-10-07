"""Does the model pick the right tool? python -m assistant.eval.route_eval"""

import argparse
import json
import logging
from collections import Counter

import pandas as pd

from assistant.agent import TOOLS, parse_text_tool_call, start_messages
from assistant.eval.common import REPORTS, proportion, run_info, save_report
from assistant.llm import LLM
from data.common.config import ROOT, load_settings
from data.common.logging_setup import setup_logging
from data.common.seasons import current_season_start, season_label

log = logging.getLogger("assistant.eval.route")
QUESTIONS = ROOT / "assistant" / "eval" / "routing_questions.jsonl"


def chosen_tool(llm: LLM, question: str, season: str) -> str:
    """The first tool the model calls for this question, or 'none'."""
    reply = llm.chat(start_messages(question, season), tools=TOOLS)
    if reply.tool_calls:
        return reply.tool_calls[0].name
    parsed = parse_text_tool_call(reply.text)
    return parsed.name if parsed else "none"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="route_eval", help="name for the report files")
    args = parser.parse_args(argv)
    s = load_settings()
    setup_logging(s.logs)
    cfg = s.assistant
    llm = LLM(cfg)
    season = season_label(current_season_start(start_month=s.season_start_month))
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    for i, r in enumerate(rows, start=1):
        r["chosen"] = chosen_tool(llm, r["question"], season)
        r["correct"] = r["chosen"] == r["tool"]
        log.info("%d/%d %s -> %s (%s)", i, len(rows), r["id"], r["chosen"], r["correct"])
    df = pd.DataFrame(rows)
    report = {
        **run_info(cfg, llm.model),
        "tool_selection_accuracy": proportion(int(df["correct"].sum()), len(df)),
        "by_tool": {t: proportion(int(g["correct"].sum()), len(g)) for t, g in df.groupby("tool")},
        "mistakes": dict(
            Counter(f"{r.tool} -> {r.chosen}" for r in df[~df["correct"]].itertuples())
        ),
    }
    save_report(args.tag, report)
    df.to_csv(REPORTS / f"{args.tag}.csv", index=False, encoding="utf-8")
    acc = report["tool_selection_accuracy"]
    print(f"Tool-selection accuracy: {acc['k']}/{acc['n']} = {acc['value']}")
    print("Mistakes:", report["mistakes"] or "none")


if __name__ == "__main__":
    main()