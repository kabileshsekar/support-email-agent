"""Provider-agnostic LLM client.

Gemini and OpenRouter both expose an OpenAI-compatible Chat Completions API, so one
client covers both - only the base URL, key and model differ. A third provider,
`mock`, is a deterministic rule-based stand-in that lets the whole pipeline (and the
test-suite) run offline without an API key.

Every call is recorded as an `llm` span with latency and token usage.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import tracing
from .config import Settings, get_settings
from .observability import get_logger, log_event

log = get_logger("llm")

PROVIDERS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    raw_tool_calls: list[dict] = field(default_factory=list)  # provider payload, replayed verbatim

    def as_message(self) -> dict:
        """The assistant message to append to the conversation history."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.raw_tool_calls:
            msg["tool_calls"] = self.raw_tool_calls
        elif self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in self.tool_calls
            ]
        return msg


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """Every configured model has used up its (daily) quota."""


def _retry_after_seconds(exc: Exception) -> float:
    """Read the server's suggested wait from a 429 (e.g. "retryDelay': '64574s'" or "retry in 1h2m3s")."""
    text = str(exc)
    m = re.search(r"retryDelay'?:\s*'?(\d+(?:\.\d+)?)s", text)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", text)
    if m and any(m.groups()):
        h, mi, se = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mi * 60 + se
    return 0.0


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.provider = self.settings.llm_provider
        if self.provider == "mock":
            from .mock_llm import MockLLM
            self._mock = MockLLM()
            self.model = "mock-rules-v1"
            return
        if self.provider not in PROVIDERS:
            raise LLMError(f"Unknown LLM_PROVIDER '{self.provider}'. Use gemini, openrouter or mock.")
        key = self.settings.gemini_api_key if self.provider == "gemini" else self.settings.openrouter_api_key
        if not key:
            raise LLMError(f"No API key set for provider '{self.provider}'. Add it to .env "
                           f"(or set LLM_PROVIDER=mock to run offline).")
        from openai import OpenAI
        extra_headers = {}
        if self.provider == "openrouter":
            extra_headers = {"HTTP-Referer": "http://localhost", "X-Title": "Support Email Agent"}
        self._client = OpenAI(api_key=key, base_url=PROVIDERS[self.provider],
                              default_headers=extra_headers, max_retries=0, timeout=90)
        if self.provider == "gemini":
            primary, fallbacks = self.settings.gemini_model, self.settings.gemini_fallback_models
        else:
            primary, fallbacks = self.settings.openrouter_model, self.settings.openrouter_fallback_models
        # Free-tier quotas are per model, so when one runs out we move down this list.
        self.models = list(dict.fromkeys([primary, *fallbacks]))
        self.model = self.models[0]

    def _next_model(self, reason: str) -> bool:
        i = self.models.index(self.model)
        if i + 1 >= len(self.models):
            return False
        log_event(log, "switching model", old=self.model, new=self.models[i + 1], reason=reason[:200])
        self.model = self.models[i + 1]
        return True

    # ------------------------------------------------------------------ public API
    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None, *,
             purpose: str = "chat", temperature: float = 0.2, json_mode: bool = False,
             tool_choice: str = "auto") -> LLMResponse:
        with tracing.span(f"llm:{purpose}", kind="llm", input=messages[-1].get("content"),
                          provider=self.provider, model=self.model,
                          tools=[t["function"]["name"] for t in tools] if tools else []) as sp:
            if self.provider == "mock":
                resp = self._mock.chat(messages, tools if tool_choice != "none" else None, purpose=purpose)
            else:
                resp = self._call_with_retry(messages, tools, temperature, json_mode, tool_choice)
            sp.set(prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens, model=self.model)
            sp.set_output({"content": resp.content,
                           "tool_calls": [{"name": t.name, "arguments": t.arguments} for t in resp.tool_calls]})
            return resp

    def chat_json(self, messages: list[dict], *, purpose: str, temperature: float = 0.0) -> dict:
        """Ask for a JSON object and parse it robustly."""
        resp = self.chat(messages, purpose=purpose, temperature=temperature, json_mode=True)
        return parse_json(resp.content)

    # ------------------------------------------------------------------ internals
    def _call_with_retry(self, messages, tools, temperature, json_mode, tool_choice="auto", attempts: int = 5) -> LLMResponse:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        kwargs: dict[str, Any] = {"messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}

        delay, attempt = 3.0, 0
        while True:
            kwargs["model"] = self.model
            try:
                r = self._client.chat.completions.create(**kwargs)
                return self._parse(r)
            except RateLimitError as exc:
                wait = _retry_after_seconds(exc)
                if wait > 90:
                    # A long reset means a daily quota, not a per-minute burst: waiting is pointless.
                    if self._next_model(f"quota exhausted, resets in {wait / 3600:.1f} h"):
                        continue
                    raise QuotaExhausted(
                        f"{self.provider} free-tier quota is used up for all configured models "
                        f"({', '.join(self.models)}). It resets in about {wait / 3600:.1f} hours. "
                        f"Use LLM_PROVIDER=mock or another provider until then.") from exc
                err, delay = exc, max(delay, wait)
            except (APIConnectionError, APITimeoutError) as exc:
                err = exc
            except APIStatusError as exc:
                # Some free models reject response_format: retry once without it.
                if exc.status_code == 400 and "response_format" in kwargs:
                    kwargs.pop("response_format")
                    continue
                if exc.status_code == 404 and self._next_model(f"model unavailable: {exc.message}"):
                    continue
                # The model is overloaded ("high demand"): after one retry, try the next model instead.
                if exc.status_code == 503 and attempt >= 1 and self._next_model("model overloaded (503)"):
                    attempt = 0
                    continue
                if exc.status_code < 500:
                    raise LLMError(f"{self.provider} API error {exc.status_code}: {exc.message}") from exc
                err = exc
            attempt += 1
            log_event(log, "llm retry", attempt=attempt, model=self.model, error=str(err)[:300], wait_s=delay)
            if attempt >= attempts:
                raise LLMError(f"{self.provider} failed after {attempts} attempts: {err}") from err
            time.sleep(delay)
            delay = min(delay * 2, 60)

    def _parse(self, r) -> LLMResponse:
        if not r.choices:
            raise LLMError(f"Empty response from {self.provider}")
        msg = r.choices[0].message
        calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id or f"call_{i}", name=tc.function.name, arguments=args or {}))
        raw_calls = [tc.model_dump(exclude_none=True) for tc in (msg.tool_calls or [])]
        for i, rc in enumerate(raw_calls):
            rc.setdefault("id", calls[i].id)
            rc.setdefault("type", "function")
        usage = getattr(r, "usage", None)
        return LLMResponse(
            content=msg.content or "",
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=getattr(r, "model", self.model) or self.model,
            raw_tool_calls=raw_calls,
        )


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences and chatter."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Model did not return valid JSON: {text[:300]}")
