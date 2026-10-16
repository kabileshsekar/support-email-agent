"""Observability: structured logging + metrics computed from traces.

* Logs are written as JSON lines to runtime/agent.log (and plain text to the console),
  so they can be grepped or shipped to any log platform.
* Metrics (latency percentiles, token usage, tool error rate, escalation rate, intent
  mix...) are aggregated from the trace store, which is the single source of truth.
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from typing import Iterable

from . import config


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


_configured = False


def get_logger(name: str = "agent") -> logging.Logger:
    global _configured
    if not _configured:
        root = logging.getLogger("agent")
        root.setLevel(logging.INFO)
        fh = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
        fh.setFormatter(_JsonFormatter())
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
        root.addHandler(ch)
        root.propagate = False
        _configured = True
    return logging.getLogger(name if name.startswith("agent") else f"agent.{name}")


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(p / 100 * (len(values) - 1)))))
    return round(values[k], 1)


def compute_metrics(traces: Iterable[dict]) -> dict:
    traces = [t for t in traces if t.get("name") == "process_email"]
    n = len(traces)
    latencies = [t["duration_ms"] for t in traces]
    actions = Counter((t.get("output") or {}).get("action", "unknown") for t in traces)
    intents = Counter((t.get("output") or {}).get("intent", "unknown") for t in traces)

    llm_spans = [s for t in traces for s in t["spans"] if s["kind"] == "llm"]
    tool_spans = [s for t in traces for s in t["spans"] if s["kind"] == "tool"]
    guard_spans = [s for t in traces for s in t["spans"] if s["kind"] == "guardrail"]

    prompt_tokens = sum(s["attributes"].get("prompt_tokens", 0) or 0 for s in llm_spans)
    completion_tokens = sum(s["attributes"].get("completion_tokens", 0) or 0 for s in llm_spans)
    tool_errors = sum(1 for s in tool_spans if s["status"] == "error" or
                      (isinstance(s.get("output"), dict) and "error" in s["output"]))
    input_blocks = sum(1 for s in guard_spans if s["name"] == "input_guardrails" and s["attributes"].get("passed") is False)
    output_fail = sum(1 for s in guard_spans if s["name"] == "output_guardrails" and s["attributes"].get("passed") is False)

    return {
        "emails_processed": n,
        "errors": sum(1 for t in traces if t["status"] == "error"),
        "latency_ms_p50": _pct(latencies, 50),
        "latency_ms_p95": _pct(latencies, 95),
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "llm_calls": len(llm_spans),
        "llm_latency_ms_mean": round(statistics.mean([s["duration_ms"] for s in llm_spans]), 1) if llm_spans else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_email": round((prompt_tokens + completion_tokens) / n, 1) if n else 0.0,
        "tool_calls": len(tool_spans),
        "tool_error_rate": round(tool_errors / len(tool_spans), 3) if tool_spans else 0.0,
        "tool_usage": dict(Counter(s["name"].removeprefix("tool:") for s in tool_spans)),
        "guardrail_checks": len(guard_spans),
        "input_guardrail_blocks": input_blocks,      # spam / prompt injection caught before the LLM
        "output_guardrail_failures": output_fail,    # drafts rejected (incl. ones fixed by self-correction)
        "actions": dict(actions),
        "escalation_rate": round(actions.get("escalated", 0) / n, 3) if n else 0.0,
        "auto_resolved_rate": round(actions.get("sent", 0) / n, 3) if n else 0.0,
        "intents": dict(intents),
    }
