"""OpenAI-compatible chat client with retries, timeouts and token accounting.

Works with Groq, Gemini (OpenAI endpoint), OpenRouter, Ollama, OpenAI — only base_url/model change.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a response after retries (or is not configured)."""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.seconds += other.seconds


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{id, name, arguments(str), extra_content}]
    usage: Usage = field(default_factory=Usage)


class ChatModel(Protocol):
    model: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None,
             temperature: float = 0.0, max_tokens: int = 1024) -> ChatResult: ...


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: float = 60, max_retries: int = 3):
        if not api_key:
            raise LLMError("LLM_API_KEY is not set")
        from openai import OpenAI

        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024) -> ChatResult:
        import openai

        retryable = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                     openai.InternalServerError)
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t = time.time()
            try:
                kwargs: dict[str, Any] = dict(model=self.model, messages=messages, temperature=temperature,
                                              max_tokens=max_tokens)
                if tools:
                    kwargs.update(tools=tools, tool_choice="auto")
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                u = resp.usage
                usage = Usage(1, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0,
                              time.time() - t)
                # Keep provider extras (Gemini 3 puts a required thought_signature in extra_content);
                # they must be echoed back with the tool call or the next request is rejected.
                calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}",
                          "extra_content": (tc.model_extra or {}).get("extra_content")}
                         for tc in (msg.tool_calls or [])]
                return ChatResult(content=msg.content, tool_calls=calls, usage=usage)
            except retryable as exc:
                last = exc
                if attempt == self.max_retries:
                    break
                delay = min(20.0, 1.5 * 2 ** attempt) + random.random()
                retry_after = getattr(getattr(exc, "response", None), "headers", {}) or {}
                if "retry-after" in retry_after:
                    try:
                        delay = min(30.0, float(retry_after["retry-after"]))
                    except ValueError:
                        pass
                log.warning("LLM retryable error (%s), retry %d in %.1fs", type(exc).__name__, attempt + 1, delay)
                time.sleep(delay)
            except openai.BadRequestError as exc:
                # e.g. model produced a malformed tool call; not worth retrying identically
                raise LLMError(f"LLM rejected request: {exc}") from exc
            except openai.APIError as exc:
                raise LLMError(f"LLM API error: {exc}") from exc
        raise LLMError(f"LLM unavailable after {self.max_retries + 1} attempts: {last}")
