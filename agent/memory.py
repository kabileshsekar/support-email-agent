"""Agent memory.

* WorkingMemory (short-term): the scratchpad for one email - plan, tool calls and the
  evidence collected so far. Lives only for the duration of one run.
* ConversationMemory (long-term): every handled email is stored in SQLite with a short
  summary, so a follow-up email (same thread or same customer) is answered with the
  context of earlier conversations.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import MEMORY_DB_PATH


@dataclass
class WorkingMemory:
    plan: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)   # {"name", "arguments", "result"}
    notes: list[str] = field(default_factory=list)

    def record_tool(self, name: str, arguments: dict, result: dict) -> None:
        self.tool_calls.append({"name": name, "arguments": arguments, "result": result})

    def evidence_text(self) -> str:
        """All facts the agent has seen this run - used by the grounding guardrail."""
        return "\n".join(json.dumps(t["result"], default=str) for t in self.tool_calls)

    @property
    def tools_used(self) -> list[str]:
        return [t["name"] for t in self.tool_calls]


class ConversationMemory:
    def __init__(self, path: Path = MEMORY_DB_PATH):
        self.path = path
        with closing(sqlite3.connect(self.path)) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, email_id TEXT, thread_id TEXT, customer_email TEXT,
                ts TEXT, subject TEXT, intents TEXT, summary TEXT, reply TEXT, action TEXT)""")
            con.commit()

    def save(self, *, email_id: str, thread_id: str, customer_email: str, subject: str,
             intents: list[str], summary: str, reply: str, action: str) -> None:
        with closing(sqlite3.connect(self.path)) as con:
            con.execute(
                "INSERT INTO interactions (email_id, thread_id, customer_email, ts, subject, intents, summary, reply,"
                " action) VALUES (?,?,?,?,?,?,?,?,?)",
                (email_id, thread_id, customer_email.lower(), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 subject, json.dumps(intents), summary, reply, action))
            con.commit()

    def history(self, customer_email: str, thread_id: Optional[str] = None, limit: int = 3) -> list[dict]:
        """Most recent interactions: same thread first, then other emails from the same customer."""
        with closing(sqlite3.connect(self.path)) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM interactions WHERE customer_email = ? OR thread_id = ? "
                "ORDER BY (thread_id = ?) DESC, id DESC LIMIT ?",
                (customer_email.lower(), thread_id or "", thread_id or "", limit)).fetchall()
        return [dict(r) for r in rows]

    def all(self, limit: int = 200) -> list[dict]:
        with closing(sqlite3.connect(self.path)) as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,))]

    def clear(self) -> None:
        with closing(sqlite3.connect(self.path)) as con:
            con.execute("DELETE FROM interactions")
            con.commit()


def format_history(rows: list[dict]) -> str:
    if not rows:
        return "No previous interactions."
    return "\n".join(f"- [{r['ts']}] (thread {r['thread_id']}) Subject: {r['subject']} | {r['summary']}" for r in rows)
