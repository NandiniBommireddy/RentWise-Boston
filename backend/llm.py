"""Local generation backend. No cloud APIs.

RentWise runs entirely on the machine in front of you. Retrieval is DuckDB + its FTS
extension for BM25 + bge-small ONNX embeddings; generation is a local model served
over an OpenAI-compatible HTTP endpoint. Nothing in this file talks to a hosted API.

Two implementations:

  * LocalBackend -- any OpenAI-compatible server: ollama, llama.cpp's llama-server,
    LM Studio, vLLM. Selected by default.
  * NullBackend  -- no model at all. Retrieval still runs and answers are rendered
    from templates, so the pipeline is demonstrable before a model is pulled.

    RENTWISE_LLM=local | none

## Context length matters more than anything else here

The agent sends a ~1,500-token system prompt, ~800 tokens of tool schemas, and up to
~1,500 tokens of retrieved property cards per turn. That is comfortably past ollama's
default 4096-token context, and ollama does not error when you exceed it -- it
silently drops the oldest tokens, which usually means the tool definitions vanish and
the model stops calling tools for no visible reason.

Start the server with a real context window:

    OLLAMA_CONTEXT_LENGTH=16384 ollama serve

`LocalBackend.probe()` checks this at startup and warns loudly if it looks too small.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

# A non-reasoning instruct model, deliberately. Qwen3 is a hybrid-reasoning model and
# emits 700-900 hidden thinking tokens even for a two-sentence answer; at the ~26 tok/s
# this class of hardware sustains, that is 30-70s per question. Neither ollama's
# `think: false` nor Qwen3's `/no_think` switch suppressed it. Swapping to a plain
# instruct model cut the same answers to 56-75 output tokens and 2-3s, a 17-22x
# improvement, and the model is smaller (1.9 GB vs 2.5 GB).
LOCAL_MODEL = os.environ.get("RENTWISE_LOCAL_MODEL", "qwen2.5:3b-instruct")
LOCAL_BASE_URL = os.environ.get("RENTWISE_LOCAL_BASE_URL", "http://localhost:11434/v1")
LOCAL_TIMEOUT = int(os.environ.get("RENTWISE_LOCAL_TIMEOUT", "600"))
LOCAL_MAX_TOKENS = int(os.environ.get("RENTWISE_LOCAL_MAX_TOKENS", "2048"))

# Below this, tool definitions get silently truncated out of the prompt.
MIN_SAFE_CONTEXT = 8192


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


class LocalModelUnavailable(RuntimeError):
    """No local server reachable, or the requested model is not pulled."""


class LocalBackend:
    """A local model behind an OpenAI-compatible /chat/completions endpoint."""

    name = "local"

    def __init__(
        self,
        model: str = LOCAL_MODEL,
        base_url: str = LOCAL_BASE_URL,
        max_tokens: int = LOCAL_MAX_TOKENS,
        timeout: int = LOCAL_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ---------- startup checks ----------

    def probe(self) -> dict:
        """Verify the server is up, the model exists, and the context is big enough.

        Called at startup so a misconfigured local model fails on boot with an
        actionable message, rather than halfway through a demo query.
        """
        info: dict = {"model": self.model, "base_url": self.base_url}

        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=10) as resp:
                listed = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise LocalModelUnavailable(
                f"No local model server at {self.base_url}. Start one:\n"
                f"    OLLAMA_CONTEXT_LENGTH=16384 ollama serve"
            ) from exc

        available = [m.get("id") for m in listed.get("data", [])]
        info["available"] = available
        if available and self.model not in available:
            raise LocalModelUnavailable(
                f"Model {self.model!r} is not available. Pulled models: "
                f"{', '.join(available) or 'none'}.\n"
                f"    ollama pull {self.model}"
            )

        info["context_length"] = self._context_length()
        if info["context_length"] and info["context_length"] < MIN_SAFE_CONTEXT:
            log.warning(
                "Local model context is %s tokens, below the %s needed for the tool "
                "schemas and retrieved cards. Tool definitions will be silently "
                "truncated and the agent will stop calling tools. Restart with: "
                "OLLAMA_CONTEXT_LENGTH=16384 ollama serve",
                info["context_length"],
                MIN_SAFE_CONTEXT,
            )
        return info

    def _context_length(self) -> int | None:
        """Read the context window ollama is *actually serving*.

        `/api/show` reports the model's trained maximum (262144 for Qwen3), which is
        useless for this check: it reads the same whether the server was started with
        a 4096-token window or a 32k one, so it would fail to warn in precisely the
        case this exists to catch. `/api/ps` reports the real per-model runtime value,
        but only once the model is loaded -- so warm it first.

        ollama-specific and best-effort. Returns None on any other server, which is
        not an error.
        """
        root = self.base_url.removesuffix("/v1")

        def served() -> int | None:
            try:
                with urllib.request.urlopen(f"{root}/api/ps", timeout=10) as resp:
                    running = json.loads(resp.read())
            except Exception:  # noqa: BLE001 - not ollama, or an older version
                return None
            for entry in running.get("models") or []:
                if entry.get("model") == self.model and entry.get("context_length"):
                    return int(entry["context_length"])
            return None

        if (length := served()) is not None:
            return length

        # Not resident yet. A minimal completion loads it, which also means the first
        # real question does not pay the model-load latency mid-demo.
        log.info("warming %s ...", self.model)
        try:
            self._post(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                }
            )
        except LocalModelUnavailable:
            return None
        return served()

    # ---------- generation ----------

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500]
            raise LocalModelUnavailable(
                f"Local model returned HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LocalModelUnavailable(
                f"Lost the local model server at {self.base_url}: {exc}"
            ) from exc

    def start(self, system: str, user: str, tools: list[dict]) -> list[Any]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def step(self, system: str, messages: list[Any], tools: list[dict]) -> Turn:
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        # Omit `tools` entirely when empty rather than sending []. Some
        # OpenAI-compatible servers reject an empty array, and the headline pass
        # deliberately runs without tools.
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]
        data = self._post(payload)

        choices = data.get("choices") or []
        if not choices:
            raise LocalModelUnavailable(f"Local model returned no choices: {data}")
        message = choices[0].get("message") or {}

        calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            raw_args = fn.get("arguments")
            calls.append(ToolCall(id=tc.get("id") or name, name=name, arguments=_parse_args(raw_args)))

        usage = data.get("usage") or {}
        return Turn(
            text=(message.get("content") or "").strip(),
            tool_calls=calls,
            raw=message,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        )

    def append_turn(self, messages: list[Any], turn: Turn) -> None:
        entry: dict = {"role": "assistant", "content": turn.text}
        if turn.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in turn.tool_calls
            ]
        messages.append(entry)

    def append_results(self, messages: list[Any], results: list[tuple[ToolCall, str]]) -> None:
        for call, output in results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": output,
                }
            )


class NullBackend:
    """No model. Retrieval runs; the answer is templated from what was retrieved."""

    name = "none"

    def probe(self) -> dict:
        return {"model": None}

    def start(self, system: str, user: str, tools: list[dict]) -> list[Any]:
        return [{"role": "user", "content": user}]

    def step(self, system: str, messages: list[Any], tools: list[dict]) -> Turn:
        return Turn(text="", tool_calls=[])

    def append_turn(self, messages: list[Any], turn: Turn) -> None:
        pass

    def append_results(self, messages: list[Any], results: list[tuple[ToolCall, str]]) -> None:
        pass


def _parse_args(raw: Any) -> dict[str, Any]:
    """Coerce a tool-call argument payload into a dict.

    Small local models are noticeably less reliable here than hosted ones: arguments
    arrive as a JSON string, as an already-decoded dict, as double-encoded JSON, or as
    something malformed. Recovering instead of raising costs one retry turn rather
    than the whole request.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            log.warning("local model emitted invalid tool-call JSON: %r", text[:200])
            return {}
        # Double-encoded: json.loads gave us another JSON string.
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError:
                return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def build_backend(kind: str | None = None) -> LLMBackend:
    kind = (kind or os.environ.get("RENTWISE_LLM") or "local").lower()

    if kind in {"none", "null", "off"}:
        return NullBackend()

    if kind in {"claude", "anthropic", "openai"}:
        raise ValueError(
            f"RENTWISE_LLM={kind!r} is not supported -- RentWise runs entirely on "
            "local models. Use RENTWISE_LLM=local (default) or 'none'."
        )

    if kind != "local":
        raise ValueError(f"unknown RENTWISE_LLM={kind!r} (expected local|none)")

    backend = LocalBackend()
    try:
        info = backend.probe()
    except LocalModelUnavailable as exc:
        log.warning("Local model unavailable, falling back to retrieval-only.\n%s", exc)
        return NullBackend()
    log.info(
        "local model ready: %s (context %s)",
        info.get("model"),
        info.get("context_length") or "unknown",
    )
    return backend
