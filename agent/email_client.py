"""Email I/O: read incoming emails, send replies, queue cases for human review.

MockMailbox  - reads data/inbox.json, writes replies to runtime/outbox.jsonl (default)
GmailMailbox - real Gmail over IMAP/SMTP using an App Password (EMAIL_BACKEND=gmail)
"""
from __future__ import annotations

import email
import imaplib
import json
import smtplib
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Optional

from . import config


@dataclass
class Email:
    id: str
    from_email: str
    subject: str
    body: str
    from_name: str = ""
    thread_id: str = ""
    received_at: str = ""
    in_reply_to: Optional[str] = None

    def __post_init__(self):
        self.thread_id = self.thread_id or self.id

    @classmethod
    def from_dict(cls, d: dict) -> "Email":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()][::-1]


class ReviewQueue:
    """Cases handed to a human: escalations, guardrail failures, AUTO_SEND=false drafts."""

    def __init__(self, path: Path = config.ESCALATIONS_PATH):
        self.path = path

    def add(self, email_obj: Email, reason: str, priority: str, draft: str, trace_id: str) -> dict:
        rec = {"ticket_id": f"TKT-{uuid.uuid4().hex[:6].upper()}", "created_at": _now(),
               "email_id": email_obj.id, "from": email_obj.from_email, "subject": email_obj.subject,
               "reason": reason, "priority": priority, "draft_reply": draft, "trace_id": trace_id}
        _append_jsonl(self.path, rec)
        return rec

    def all(self) -> list[dict]:
        return read_jsonl(self.path)


class MockMailbox:
    def __init__(self, inbox_path: Path = config.INBOX_PATH, outbox_path: Path = config.OUTBOX_PATH,
                 state_path: Path = config.MAILBOX_STATE_PATH):
        self.inbox_path, self.outbox_path, self.state_path = inbox_path, outbox_path, state_path

    def _processed(self) -> set[str]:
        if self.state_path.exists():
            return set(json.loads(self.state_path.read_text(encoding="utf-8")))
        return set()

    def all_emails(self) -> list[Email]:
        return [Email.from_dict(d) for d in json.loads(self.inbox_path.read_text(encoding="utf-8"))]

    def fetch_unread(self) -> list[Email]:
        done = self._processed()
        return [e for e in self.all_emails() if e.id not in done]

    def mark_processed(self, email_id: str) -> None:
        done = self._processed() | {email_id}
        self.state_path.write_text(json.dumps(sorted(done)), encoding="utf-8")

    def reset(self) -> None:
        for p in (self.state_path, self.outbox_path):
            if p.exists():
                p.unlink()

    def send(self, to: str, subject: str, body: str, in_reply_to: Optional[str] = None) -> dict:
        rec = {"message_id": f"MSG-{uuid.uuid4().hex[:8]}", "sent_at": _now(), "to": to,
               "subject": subject, "body": body, "in_reply_to": in_reply_to}
        _append_jsonl(self.outbox_path, rec)
        return rec

    def sent(self) -> list[dict]:
        return read_jsonl(self.outbox_path)


class GmailMailbox:
    """Minimal Gmail adapter. Requires 2FA + an App Password: https://myaccount.google.com/apppasswords"""

    IMAP_HOST, SMTP_HOST = "imap.gmail.com", "smtp.gmail.com"

    def __init__(self, address: str, app_password: str):
        if not (address and app_password):
            raise ValueError("Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env to use EMAIL_BACKEND=gmail")
        self.address, self.password = address, app_password

    def fetch_unread(self, limit: int = 20) -> list[Email]:
        out = []
        with imaplib.IMAP4_SSL(self.IMAP_HOST) as imap:
            imap.login(self.address, self.password)
            imap.select("INBOX")
            _, data = imap.search(None, "UNSEEN")
            for uid in data[0].split()[:limit]:
                _, msg_data = imap.fetch(uid, "(BODY.PEEK[])")   # PEEK: stays unread until processed
                msg = email.message_from_bytes(msg_data[0][1])
                name, addr = parseaddr(msg.get("From", ""))
                out.append(Email(
                    id=uid.decode(), from_email=addr, from_name=str(make_header(decode_header(name or ""))),
                    subject=str(make_header(decode_header(msg.get("Subject", "")))), body=self._text(msg),
                    thread_id=(msg.get("References") or msg.get("Message-ID") or uid.decode()).split()[0],
                    received_at=msg.get("Date", ""), in_reply_to=msg.get("Message-ID")))
        return out

    @staticmethod
    def _text(msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                    return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
            return ""
        return msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", "replace")

    def mark_processed(self, email_id: str) -> None:
        with imaplib.IMAP4_SSL(self.IMAP_HOST) as imap:
            imap.login(self.address, self.password)
            imap.select("INBOX")
            imap.store(email_id, "+FLAGS", "\\Seen")

    def send(self, to: str, subject: str, body: str, in_reply_to: Optional[str] = None) -> dict:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.address, to, subject
        if in_reply_to:
            msg["In-Reply-To"] = msg["References"] = in_reply_to
        msg.set_content(body)
        with smtplib.SMTP_SSL(self.SMTP_HOST, 465) as smtp:
            smtp.login(self.address, self.password)
            smtp.send_message(msg)
        rec = {"sent_at": _now(), "to": to, "subject": subject, "body": body, "in_reply_to": in_reply_to}
        _append_jsonl(config.OUTBOX_PATH, rec)
        return rec


def get_mailbox(settings: Optional[config.Settings] = None):
    settings = settings or config.get_settings()
    if settings.email_backend == "gmail":
        return GmailMailbox(settings.gmail_address, settings.gmail_app_password)
    return MockMailbox()


__all__ = ["Email", "MockMailbox", "GmailMailbox", "ReviewQueue", "get_mailbox", "read_jsonl", "asdict"]
