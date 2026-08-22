"""Pluggable generation backend.

The retrieval stack (DuckDB, BM25, local ONNX embeddings) is entirely local and
open-source. Generation is the one step that can go either way, so it sits behind a
small protocol with two implementations:

  * ClaudeBackend -- Claude via the Anthropic SDK. Reliable multi-turn tool calling,
    which the agent depends on. This is the demo default.
  * LocalBackend  -- any OpenAI-compatible local server (ollama, llama.cpp,
    LM Studio, vLLM). Set RENTWISE_LLM=local to claim the fully-offline path.
  * NullBackend   -- no model at all. Retrieval still runs and answers are rendered
    from templates, so the pipeline is demonstrable with no credentials. Not a
    substitute for the real thing; it exists so a missing key never blocks a demo.

Selection is by env var so the toggle can be shown live:
    RENTWISE_LLM=claude | local | none
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("RENTWISE_CLAUDE_MODEL", "claude-opus-5")
LOCAL_MODEL = os.environ.get("RENTWISE_LOCAL_MODEL", "qwen3:4b")
LOCAL_BASE_URL = os.environ.get("RENTWISE_LOCAL_BASE_URL", "http://localhost:11434/v1")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Turn:
    """One model turn, normalized across backends."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMBackend(Protocol):
    name: str

    def start(self, system: str, user: str, tools: list[dict]) -> list[Any]:
        """Build the initial provider-native message list."""

    def step(self, system: str, messages: list[Any], tools: list[dict]) -> Turn:
        """Run one turn."""

    def append_turn(self, messages: list[Any], turn: Turn) -> None:
        """Append the assistant turn to history, provider-natively."""

    def append_results(self, messages: list[Any], results: list[tuple[ToolCall, str]]) -> None:
        """Append tool results to history."""


class ClaudeBackend:
    """Claude via the Anthropic SDK (>=1.0).

    Adaptive thinking is on and effort is high, because choosing a retrieval path and
    writing correct SQL over an 8-table schema is exactly the kind of work that
    benefits from it. Sampling parameters are deliberately absent -- they were removed
    in SDK 1.x and current models do not use them.
    """

    name = "claude"

    def __init__(self, model: str = CLAUDE_MODEL, max_tokens: int = 8000) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        # The client constructs happily with no credentials and only fails on the
        # first request, which would surface as a 500 mid-demo. Fail here instead so
        # build_backend() can fall back while the server is still starting.
        if not (self.client.api_key or self.client.auth_token):
            raise RuntimeError(
                "no Anthropic credentials found -- set ANTHROPIC_API_KEY in backend/.env"
            )
        self.model = model
        self.max_tokens = max_tokens

    def start(self, system: str, user: str, tools: list[dict]) -> list[Any]:
        return [{"role": "user", "content": user}]

    def step(self, system: str, messages: list[Any], tools: list[dict]) -> Turn:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # Cache the system prompt + schema: it is identical on every request and
            # is the largest stable prefix we send.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            tools=[
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["input_schema"],
                }
                for t in tools
            ],
            messages=messages,
        )

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "explanation", None) or "policy refusal"
            raise RuntimeError(f"Claude declined this request: {detail}")

        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }
        return Turn(text=text, tool_calls=calls, raw=response, usage=usage)

    def append_turn(self, messages: list[Any], turn: Turn) -> None:
        # Append the whole content list, not just text -- thinking blocks must be
        # echoed back unchanged for the model to continue its own reasoning.
        messages.append({"role": "assistant", "content": turn.raw.content})

    def append_results(self, messages: list[Any], results: list[tuple[ToolCall, str]]) -> None:
        # All tool results for one assistant turn go in a SINGLE user message.
        # Splitting them teaches the model to stop making parallel calls.
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call.id, "content": output}
                    for call, output in results
                ],
            }
        )


class LocalBackend:
    """Any OpenAI-compatible local server -- ollama, llama.cpp, LM Studio, vLLM.

    Uses urllib rather than the openai package to keep the dependency surface small;
    the chat-completions tool-calling shape is stable enough to hand-roll.
    """

    name = "local"

    def __init__(
        self,
        model: str = LOCAL_MODEL,
        base_url: str = LOCAL_BASE_URL,
        max_tokens: int = 4000,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens

    def _post(self, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"No local model server at {self.base_url}. Start one, e.g. "
                f"`ollama serve` after `ollama pull {self.model}`."
            ) from exc

    def start(self, system: str, user: str, tools: list[dict]) -> list[Any]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def step(self, system: str, messages: list[Any], tools: list[dict]) -> Turn:
        data = self._post(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": t["input_schema"],
                        },
                    }
                    for t in tools
                ],
            }
        )
        choice = data["choices"][0]["message"]
        calls = []
        for tc in choice.get("tool_calls") or []:
            fn = tc["function"]
            raw_args = fn.get("arguments") or "{}"
            try:
                # Small local models occasionally emit malformed JSON here; a bad
                # parse should cost one retry turn, not crash the request.
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                log.warning("local model emitted invalid tool JSON: %r", raw_args)
                args = {}
            calls.append(ToolCall(id=tc.get("id") or fn["name"], name=fn["name"], arguments=args))

        usage = data.get("usage") or {}
        return Turn(
            text=choice.get("content") or "",
            tool_calls=calls,
            raw=choice,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        )

    def append_turn(self, messages: list[Any], turn: Turn) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": turn.text,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in turn.tool_calls
                ],
            }
        )

    def append_results(self, messages: list[Any], results: list[tuple[ToolCall, str]]) -> None:
        for call, output in results:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})


class NullBackend:
    """No model. Retrieval runs; the answer is templated from what was retrieved."""

    name = "none"

    def start(self, system: str, user: str, tools: list[dict]) -> list[Any]:
        return [{"role": "user", "content": user}]

    def step(self, system: str, messages: list[Any], tools: list[dict]) -> Turn:
        return Turn(text="", tool_calls=[])

    def append_turn(self, messages: list[Any], turn: Turn) -> None:
        pass

    def append_results(self, messages: list[Any], results: list[tuple[ToolCall, str]]) -> None:
        pass


def build_backend(kind: str | None = None) -> LLMBackend:
    kind = (kind or os.environ.get("RENTWISE_LLM") or "claude").lower()
    if kind == "local":
        return LocalBackend()
    if kind in {"none", "null", "off"}:
        return NullBackend()
    if kind != "claude":
        raise ValueError(f"unknown RENTWISE_LLM={kind!r} (expected claude|local|none)")
    try:
        return ClaudeBackend()
    except Exception as exc:  # noqa: BLE001 - missing key must not be a hard crash
        log.warning("Claude backend unavailable (%s); falling back to retrieval-only.", exc)
        return NullBackend()
