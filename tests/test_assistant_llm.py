import pytest

from assistant.llm import LLM, LLMError, ToolCall, cost_usd

CFG = {
    "provider": "ollama", "ollama_url": "http://localhost:11434", "chat_model": "llama3.2:3b",
    "embed_model": "all-minilm", "num_ctx": 4096, "temperature": 0.0, "seed": 1,
    "keep_alive": "30m", "request_timeout_seconds": 5, "hosted_base_url": "https://example.com/v1",
    "hosted_model": "some-model", "hosted_price_in": 0.5, "hosted_price_out": 2.0,
}  # fmt: skip


class FakeResponse:
    def __init__(self, data, status_code=200):
        self.data, self.status_code, self.text = data, status_code, str(data)

    def json(self):
        return self.data


class FakeSession:
    """Stands in for requests.Session, so the tests need no running model."""

    def __init__(self, data, status_code=200):
        self.data, self.status_code, self.calls = data, status_code, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        return FakeResponse(self.data, self.status_code)


def test_ollama_request_and_tool_call():
    data = {
        "message": {
            "content": "",
            "tool_calls": [
                {"function": {"name": "predict_match", "arguments": {"home_team": "A"}}}
            ],
        },
        "prompt_eval_count": 120,
        "eval_count": 15,
    }
    session = FakeSession(data)
    reply = LLM(CFG, session).chat([{"role": "user", "content": "hi"}], tools=[{"x": 1}])
    url, body, _ = session.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert body["stream"] is False and body["options"]["num_ctx"] == 4096
    assert reply.tool_calls == [ToolCall("predict_match", {"home_team": "A"})]
    assert (reply.prompt_tokens, reply.output_tokens) == (120, 15)


def test_hosted_request_and_tool_call(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "team_ratings", "arguments": "{}"}}
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 5},
    }
    session = FakeSession(data)
    llm = LLM({**CFG, "provider": "openai_compatible"}, session)
    reply = llm.chat([{"role": "user", "content": "hi"}], schema={"type": "object"})
    url, body, headers = session.calls[0]
    assert url == "https://example.com/v1/chat/completions"
    assert headers == {"Authorization": "Bearer test-key"}
    assert body["response_format"]["type"] == "json_schema"
    assert reply.tool_calls == [ToolCall("team_ratings", {}, "c1")]
    assert llm.tool_message(reply.tool_calls[0], "x")["tool_call_id"] == "c1"


def test_chat_json_rejects_text_that_is_not_json():
    llm = LLM(CFG, FakeSession({"message": {"content": "not json"}}))
    with pytest.raises(LLMError):
        llm.chat_json([{"role": "user", "content": "hi"}], {"type": "object"})


def test_error_status_raises():
    with pytest.raises(LLMError):
        LLM(CFG, FakeSession({"error": "model not found"}, 404)).chat([])


def test_cost_is_zero_locally_and_priced_when_hosted():
    assert cost_usd(1_000_000, 1_000_000, CFG) == 0.0
    assert cost_usd(2000, 100, {**CFG, "provider": "openai_compatible"}) == pytest.approx(0.0012)


def test_answer_only_removes_tools_locally_and_sets_tool_choice_when_hosted(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    local = FakeSession({"message": {"content": "ok"}})
    LLM(CFG, local).chat([], tools=[{"x": 1}], answer_only=True)
    assert "tools" not in local.calls[0][1]
    data = {"choices": [{"message": {"content": "ok"}}]}
    hosted = FakeSession(data)
    LLM({**CFG, "provider": "openai_compatible"}, hosted).chat(
        [], tools=[{"x": 1}], answer_only=True
    )
    assert hosted.calls[0][1]["tool_choice"] == "none"
