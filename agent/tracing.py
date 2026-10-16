"""Lightweight tracing.

Every processed email gets one Trace. Each step inside it (planning, an LLM call, a
tool call, a guardrail check...) is recorded as a Span with inputs, outputs, timing
and attributes such as token usage. Spans nest, so the trace forms a tree that the
UI can render. Finished traces are appended as one JSON line to runtime/traces.jsonl.

The active trace is kept in a ContextVar so any module can open a span with
`tracing.span(...)` without having the trace object passed around.
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from . import config

_current_trace: ContextVar[Optional["Trace"]] = ContextVar("current_trace", default=None)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe(value: Any, limit: int = 4000) -> Any:
    """Make a value JSON-serialisable and cap very long strings."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        value = repr(value)
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [truncated {len(value) - limit} chars]"
    return value


@dataclass
class Span:
    name: str
    kind: str                      # step | llm | tool | guardrail | memory | email
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    start_time: str = field(default_factory=_now_iso)
    duration_ms: float = 0.0
    status: str = "ok"             # ok | error
    input: Any = None
    output: Any = None
    attributes: dict = field(default_factory=dict)
    error: Optional[str] = None

    def set_output(self, output: Any) -> None:
        self.output = _safe(output)

    def set(self, **attrs: Any) -> None:
        self.attributes.update({k: _safe(v) for k, v in attrs.items()})


class Trace:
    def __init__(self, name: str, metadata: Optional[dict] = None, path: Path = config.TRACES_PATH):
        self.trace_id = uuid.uuid4().hex[:16]
        self.name = name
        self.metadata = metadata or {}
        self.start_time = _now_iso()
        self._t0 = time.perf_counter()
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        self.path = path
        self.status = "ok"
        self.output: Any = None
        self.duration_ms = 0.0

    @contextmanager
    def span(self, name: str, kind: str = "step", input: Any = None, **attrs: Any) -> Iterator[Span]:
        sp = Span(name=name, kind=kind, input=_safe(input),
                  parent_id=self._stack[-1].span_id if self._stack else None)
        sp.set(**attrs)
        self.spans.append(sp)
        self._stack.append(sp)
        t0 = time.perf_counter()
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
            raise
        finally:
            sp.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            self._stack.pop()

    def finish(self, status: str = "ok", output: Any = None) -> dict:
        self.status = status
        self.output = _safe(output)
        self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        record = self.to_dict()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
            "output": self.output,
            "spans": [asdict(s) for s in self.spans],
        }


@contextmanager
def start_trace(name: str, metadata: Optional[dict] = None, persist: bool = True) -> Iterator[Trace]:
    trace = Trace(name, metadata, path=config.TRACES_PATH if persist else None)
    token = _current_trace.set(trace)
    try:
        yield trace
    finally:
        _current_trace.reset(token)


def current_trace() -> Optional[Trace]:
    return _current_trace.get()


@contextmanager
def span(name: str, kind: str = "step", input: Any = None, **attrs: Any) -> Iterator[Span]:
    """Open a span on the active trace (or a throw-away span if no trace is active)."""
    trace = _current_trace.get()
    if trace is None:
        yield Span(name=name, kind=kind)
        return
    with trace.span(name, kind, input, **attrs) as sp:
        yield sp


def load_traces(path: Path = config.TRACES_PATH, limit: Optional[int] = None) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.reverse()  # newest first
    return rows[:limit] if limit else rows
