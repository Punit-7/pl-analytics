"""Turn a question into SQL with the language model, run it safely, return the rows."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from sqlalchemy.engine import Engine

from assistant.llm import LLM, LLMError
from assistant.schema import schema_card
from assistant.sql_runner import QueryResult, SQLFailed, SQLRejected, run_sql
from data.common.config import ROOT

log = logging.getLogger(__name__)
QUESTIONS = ROOT / "assistant" / "eval" / "sql_questions.jsonl"
SQL_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}},
    "required": ["sql"],
    "additionalProperties": False,
}
MODES = ("baseline", "full", "repair")


@dataclass
class SQLAnswer:
    question: str
    sql: str = ""
    result: QueryResult | None = None
    error: str = ""
    attempts: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def load_questions(split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if split is None or r["split"] == split]


class Text2SQL:
    """mode 'baseline': table and column names only. 'full': schema card and examples.
    'repair': 'full', plus one retry when the database reports an error."""

    def __init__(self, llm: LLM, engine: Engine, cfg: dict, mode: str = "repair"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.llm, self.engine, self.cfg, self.mode = llm, engine, cfg, mode
        card = schema_card(engine, cfg["sql_tables"], full=mode != "baseline")
        self.system = (
            "You write PostgreSQL queries for a Premier League football warehouse.\n"
            'Reply with JSON of the form {"sql": "..."} and nothing else.\n\n' + card
        )
        self.examples = [] if mode == "baseline" else load_questions("dev")[: cfg["few_shot"]]

    def messages(self, question: str) -> list[dict]:
        out = [{"role": "system", "content": self.system}]
        for ex in self.examples:  # few-shot examples, written as earlier turns of the chat
            out.append({"role": "user", "content": ex["question"]})
            out.append({"role": "assistant", "content": json.dumps({"sql": ex["gold_sql"]})})
        out.append({"role": "user", "content": question})
        return out

    def ask(self, question: str) -> SQLAnswer:
        answer = SQLAnswer(question)
        messages = self.messages(question)
        tries = 2 if self.mode == "repair" else 1
        for _ in range(tries):
            answer.attempts += 1
            try:
                data, reply = self.llm.chat_json(messages, SQL_SCHEMA)
            except LLMError as e:
                answer.error = f"model: {e}"
                return answer
            answer.prompt_tokens += reply.prompt_tokens
            answer.output_tokens += reply.output_tokens
            answer.seconds += reply.seconds
            answer.sql = str(data.get("sql", ""))
            try:
                answer.result = run_sql(
                    self.engine, answer.sql, self.cfg["sql_timeout_ms"], self.cfg["sql_max_rows"]
                )
                answer.error = ""
                return answer
            except (SQLRejected, SQLFailed) as e:
                answer.error = str(e)
                log.info("SQL failed (%s): %s", answer.error, answer.sql)
                messages += [
                    {"role": "assistant", "content": json.dumps({"sql": answer.sql})},
                    {
                        "role": "user",
                        "content": f"That query failed with this error: {answer.error}\n"
                        "Reply with a corrected query in the same JSON form.",
                    },
                ]
        return answer


if __name__ == "__main__":
    import sys

    from data.common.config import load_settings
    from data.common.db import make_reader_engine

    cfg = load_settings().assistant
    t2s = Text2SQL(LLM(cfg), make_reader_engine(), cfg)
    a = t2s.ask(" ".join(sys.argv[1:]) or "How many points did Arsenal win in 2023/24?")
    print("SQL:", a.sql)
    if a.result:
        print(a.result.columns)
        for row in a.result.rows[:20]:
            print(row)
    else:
        print("Error:", a.error)
    print(f"{a.attempts} attempt(s), {a.seconds:.1f} s, {a.prompt_tokens} prompt tokens")
