"""Central configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)


def env_version() -> float:
    """Last-modified time of .env, so long-running processes (the UI) can notice edits."""
    return ENV_PATH.stat().st_mtime if ENV_PATH.exists() else 0.0

DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "kb"
RUNTIME_DIR = ROOT / "runtime"          # everything the agent writes at runtime
DB_PATH = RUNTIME_DIR / "support.db"    # customers / orders / devices / warranty
MEMORY_DB_PATH = RUNTIME_DIR / "memory.db"
TRACES_PATH = RUNTIME_DIR / "traces.jsonl"
LOG_PATH = RUNTIME_DIR / "agent.log"
OUTBOX_PATH = RUNTIME_DIR / "outbox.jsonl"
ESCALATIONS_PATH = RUNTIME_DIR / "escalations.jsonl"
MAILBOX_STATE_PATH = RUNTIME_DIR / "mailbox_state.json"
INBOX_PATH = DATA_DIR / "inbox.json"


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").lower())
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
    gemini_fallback_models: list = field(default_factory=lambda: _list(
        "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-flash-lite-latest,gemini-3.1-flash-lite"))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openrouter/free")
    )
    openrouter_fallback_models: list = field(default_factory=lambda: _list(
        "OPENROUTER_FALLBACK_MODELS", "nvidia/nemotron-3-super-120b-a12b:free,google/gemma-4-31b-it:free"))
    email_backend: str = field(default_factory=lambda: os.getenv("EMAIL_BACKEND", "mock").lower())
    gmail_address: str = field(default_factory=lambda: os.getenv("GMAIL_ADDRESS", ""))
    gmail_app_password: str = field(default_factory=lambda: os.getenv("GMAIL_APP_PASSWORD", ""))
    auto_send: bool = field(default_factory=lambda: _bool("AUTO_SEND", True))
    max_tool_steps: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_STEPS", "6")))
    min_plan_confidence: float = field(default_factory=lambda: float(os.getenv("MIN_PLAN_CONFIDENCE", "0.5")))
    support_address: str = field(
        default_factory=lambda: os.getenv("SUPPORT_ADDRESS", "support@nimbus-electronics.example")
    )


def get_settings(**overrides) -> Settings:
    # Re-read .env every time so edits (e.g. a newly pasted API key) apply without a restart.
    load_dotenv(ENV_PATH, override=True)
    s = Settings()
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, v)
    return s


RUNTIME_DIR.mkdir(exist_ok=True)
