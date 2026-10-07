"""One client for the language model: local Ollama, or a hosted OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import requests
from dotenv import load_dotenv

from data.common.config import ROOT


class LLMError(RuntimeError):
    """The model could not be reached, or it returned something unusable."""


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class LLM:
    def __init__(self, cfg: dict, session: requests.Session | None = None):
        self.cfg = cfg
        self.hosted = cfg["provider"] == "openai_compatible"
        self.model = cfg["hosted_model"] if self.hosted else cfg["chat_model"]
        self.http = session or requests.Session()
        self.key = ""
        if self.hosted:
            load_dotenv(ROOT / ".env")
            self.key = os.getenv("LLM_API_KEY", "")
            if not self.key or not self.model:
                raise LLMError(
                    "Hosted provider needs LLM_API_KEY in .env and hosted_model in config"
                )

    def _post(self, url: str, body: dict, headers: dict | None = None) -> dict:
        try:
            r = self.http.post(
                url, json=body, headers=headers, timeout=self.cfg["request_timeout_seconds"]
            )
        except requests.ConnectionError as e:
            raise LLMError(f"Cannot reach {url}. Is Ollama running?") from e
        except requests.Timeout as e:
            raise LLMError(f"No answer from {url} in time") from e
        if r.status_code != 200:
            raise LLMError(f"{url} returned {r.status_code}: {r.text[:300]}")
        return r.json()

    def chat(
        self,
        messages: list[dict],
        tools: list | None = None,
        schema: dict | None = None,
        answer_only: bool = False,
    ) -> Reply:
        """Send the conversation. `tools` offers functions; `schema` forces JSON of that shape;
        `answer_only` tells the model to reply in text and not call another tool."""
        start = time.perf_counter()
        call = self._chat_hosted if self.hosted else self._chat_ollama
        reply = call(messages, tools, schema, answer_only)
        reply.seconds = time.perf_counter() - start
        return reply

    def _chat_ollama(self, messages, tools, schema, answer_only) -> Reply:
        c = self.cfg
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": c["keep_alive"],
            "options": {
                "temperature": c["temperature"],
                "seed": c["seed"],
                "num_ctx": c["num_ctx"],
            },
        }
        if tools and not answer_only:  # Ollama has no tool_choice setting: leave the tools out
            body["tools"] = tools
        if schema:
            body["format"] = schema
        if "think" in c:
            body["think"] = c["think"]
        data = self._post(f"{c['ollama_url']}/api/chat", body)
        msg = data.get("message") or {}
        calls = [
            ToolCall(t["function"]["name"], t["function"].get("arguments") or {})
            for t in msg.get("tool_calls") or []
        ]
        return Reply(
            msg.get("content") or "",
            calls,
            data.get("prompt_eval_count", 0),
            data.get("eval_count", 0),
        )

    def _chat_hosted(self, messages, tools, schema, answer_only) -> Reply:
        body = {"model": self.model, "messages": messages, "temperature": self.cfg["temperature"]}
        if tools:
            body["tools"] = tools
            if answer_only:
                body["tool_choice"] = "none"
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "strict": True, "schema": schema},
            }
        url = f"{self.cfg['hosted_base_url'].rstrip('/')}/chat/completions"
        data = self._post(url, body, {"Authorization": f"Bearer {self.key}"})
        msg = data["choices"][0]["message"]
        calls = [
            ToolCall(
                t["function"]["name"],
                json.loads(t["function"].get("arguments") or "{}"),
                t.get("id", ""),
            )
            for t in msg.get("tool_calls") or []
        ]
        usage = data.get("usage") or {}
        return Reply(
            msg.get("content") or "",
            calls,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    def chat_json(self, messages: list[dict], schema: dict) -> tuple[dict, Reply]:
        """Ask for JSON that follows `schema` and parse it."""
        reply = self.chat(messages, schema=schema)
        try:
            return json.loads(reply.text), reply
        except json.JSONDecodeError as e:
            raise LLMError(f"Model did not return valid JSON: {reply.text[:200]}") from e

    def assistant_message(self, reply: Reply) -> dict:
        """The model's own turn, in the shape this provider expects back."""
        calls = []
        for c in reply.tool_calls:
            args = json.dumps(c.arguments) if self.hosted else c.arguments
            call = {"type": "function", "function": {"name": c.name, "arguments": args}}
            if self.hosted:
                call["id"] = c.id
            calls.append(call)
        return {"role": "assistant", "content": reply.text, "tool_calls": calls}

    def tool_message(self, call: ToolCall, content: str) -> dict:
        """A tool's result, in the shape this provider expects."""
        if self.hosted:
            return {"role": "tool", "tool_call_id": call.id, "content": content}
        return {"role": "tool", "tool_name": call.name, "content": content}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embeddings always come from the local Ollama model, one row per text."""
        body = {"model": self.cfg["embed_model"], "input": texts}
        data = self._post(f"{self.cfg['ollama_url']}/api/embed", body)
        return np.asarray(data["embeddings"], dtype=np.float32)

    def local_models(self) -> list[str]:
        """Names of the models Ollama has downloaded."""
        try:
            r = self.http.get(f"{self.cfg['ollama_url']}/api/tags", timeout=10)
        except requests.ConnectionError as e:
            raise LLMError("Cannot reach Ollama. Start it from the Start menu.") from e
        return [m["name"] for m in r.json().get("models", [])]


def cost_usd(prompt_tokens: int, output_tokens: int, cfg: dict) -> float:
    """Cost of a hosted call from the prices in config.toml. A local model costs 0."""
    if cfg["provider"] != "openai_compatible":
        return 0.0
    return (prompt_tokens * cfg["hosted_price_in"] + output_tokens * cfg["hosted_price_out"]) / 1e6


if __name__ == "__main__":
    from data.common.config import load_settings

    cfg = load_settings().assistant
    llm = LLM(cfg)
    if not llm.hosted:
        print("Models:", ", ".join(llm.local_models()))
    reply = llm.chat([{"role": "user", "content": "Reply with the single word: pong"}])
    print("Reply:", reply.text.strip())
    print(f"{reply.prompt_tokens} prompt tokens, {reply.output_tokens} output tokens")
    print(f"{reply.seconds:.1f} s, {reply.output_tokens / reply.seconds:.1f} output tokens/s")
    print("Embedding shape:", llm.embed(["Arsenal", "Chelsea"]).shape)