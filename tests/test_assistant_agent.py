import json

import pytest

import assistant.agent as agent_module
from assistant.agent import Agent, clean_season, parse_text_tool_call
from assistant.llm import Reply, ToolCall
from assistant.retrieval import Hit
from assistant.sql_runner import QueryResult
from assistant.text2sql import SQLAnswer
from assistant.tools import ToolError

CFG = {"max_steps": 3, "source_chars": 200, "assistant_version": "test"}


class FakeLLM:
    """Returns scripted replies and records what it was sent."""

    model = "fake"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None, schema=None, answer_only=False):
        self.seen.append({"messages": list(messages), "answer_only": answer_only})
        return self.replies.pop(0)

    def assistant_message(self, reply):
        return {"role": "assistant", "content": reply.text}

    def tool_message(self, call, content):
        return {"role": "tool", "tool_name": call.name, "content": content}


class FakeText2SQL:
    def ask(self, question):
        result = QueryResult(["points"], [(89,)], False, 0.01)
        return SQLAnswer(question, sql="SELECT 89", result=result, attempts=1)


class FakeMatch:
    season = "2026/27"

    def team(self, name):
        return name

    def predict_match(self, home_team, away_team):
        if away_team == "Nowhere":
            raise ToolError("Unknown team 'Nowhere'.")
        return {"p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2}

    def team_ratings(self, team=""):
        return {"ratings": [{"team": "Arsenal", "strength": 0.9}]}


class FakeRetriever:
    def search(self, query, season="", team=""):
        return [Hit(1, "1-0", "2023–24 Arsenal F.C. season", "May", "2023/24", "Arsenal", "u", "x")]


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, "TRACES", tmp_path / "traces.jsonl")

    def make(replies):
        llm = FakeLLM(replies)
        return Agent(llm, FakeText2SQL(), FakeMatch(), FakeRetriever(), CFG), llm, tmp_path

    return make


def test_tool_call_then_answer(make_agent):
    agent, llm, tmp = make_agent(
        [
            Reply("", [ToolCall("query_warehouse", {"question": "points?"})]),
            Reply("Arsenal won 89 points."),
        ]
    )
    r = agent.ask("How many points did Arsenal win?")
    assert (r.tool, r.sql, r.rows, r.answer) == (
        "query_warehouse", "SELECT 89", [(89,)], "Arsenal won 89 points.",
    )  # fmt: skip
    tool_turn = llm.seen[1]["messages"][-1]
    assert tool_turn["role"] == "tool" and json.loads(tool_turn["content"])["rows"] == [[89]]
    trace = json.loads((tmp / "traces.jsonl").read_text(encoding="utf-8"))
    assert trace["tool"] == "query_warehouse" and trace["row_count"] == 1


def test_tool_call_written_as_text_is_still_run(make_agent):
    text = '{"name": "predict_match", "parameters": {"home_team": "Arsenal", "away_team": "Leeds"}}'
    agent, _, _ = make_agent([Reply(text), Reply("Arsenal are favourites.")])
    r = agent.ask("Arsenal v Leeds?")
    assert r.tool == "predict_match" and r.steps[0].ok


def test_tool_error_goes_back_to_the_model(make_agent):
    call = ToolCall("predict_match", {"home_team": "Arsenal", "away_team": "Nowhere"})
    agent, llm, _ = make_agent([Reply("", [call]), Reply("I do not know that team.")])
    r = agent.ask("Arsenal v Nowhere?")
    assert not r.steps[0].ok and "Unknown team" in llm.seen[1]["messages"][-1]["content"]


def test_after_a_working_tool_the_model_must_answer(make_agent):
    agent, llm, _ = make_agent([Reply("", [ToolCall("team_ratings", {})]), Reply("Arsenal lead.")])
    agent.ask("Ratings?")
    assert [c["answer_only"] for c in llm.seen] == [False, True]


def test_one_retry_after_an_error_then_the_loop_ends(make_agent):
    bad = ToolCall("predict_match", {"home_team": "Arsenal", "away_team": "Nowhere"})
    agent, llm, _ = make_agent([Reply("", [bad]), Reply("", [bad]), Reply("I cannot predict it.")])
    r = agent.ask("Arsenal v Nowhere?")
    assert [c["answer_only"] for c in llm.seen] == [False, False, True]
    assert len(r.steps) == 2 and r.answer == "I cannot predict it."


def test_no_tool_call_means_a_direct_reply(make_agent):
    agent, _, _ = make_agent([Reply("I can only help with Premier League questions.")])
    r = agent.ask("What is the capital of France?")
    assert r.tool == "" and r.steps == []


def test_sources_are_kept_and_citations_checked(make_agent):
    agent, _, _ = make_agent(
        [Reply("", [ToolCall("search_articles", {"query": "title"})]), Reply("They won it [1].")]
    )
    r = agent.ask("What happened?")
    assert len(r.sources) == 1 and r.citations_ok


def test_parse_text_tool_call_ignores_ordinary_text():
    assert parse_text_tool_call("Arsenal won {the} league.") is None
    assert parse_text_tool_call('{"name": "not_a_tool", "arguments": {}}') is None


def test_clean_season_accepts_common_spellings():
    assert (
        clean_season("2023-24") == clean_season("2023/2024") == clean_season("2023–24") == "2023/24"
    )
    assert clean_season("last year") == ""
