"""Command-line entry point.

    python main.py run                 # process all unread emails in the inbox
    python main.py run --provider mock # offline, no API key
    python main.py email E003          # process one inbox email (dry run unless --send)
    python main.py metrics             # print observability metrics
    python main.py reset               # re-seed DB, clear outbox/memory/traces/escalations
"""
from __future__ import annotations

import argparse
import json
import sys

from agent import config
from agent.email_client import MockMailbox
from agent.observability import compute_metrics
from agent.orchestrator import SupportAgent
from agent.tracing import load_traces


def _print_result(r) -> None:
    print(f"\n=== {r.email_id} | action={r.action} | intent={r.intent} | skill={r.skill} | trace={r.trace_id}")
    if r.tools_used:
        print("tools:", ", ".join(r.tools_used))
    if r.escalation:
        print("escalation:", r.escalation)
    if r.output_guardrails and not r.output_guardrails.get("passed", True):
        print("guardrail violations:", r.output_guardrails["violations"])
    if r.error:
        print("ERROR:", r.error)
    if r.reply:
        print("-" * 60 + "\n" + r.reply)


def reset() -> None:
    from data.seed import seed
    seed()
    for p in (config.OUTBOX_PATH, config.ESCALATIONS_PATH, config.TRACES_PATH, config.MAILBOX_STATE_PATH,
              config.MEMORY_DB_PATH):
        if p.exists():
            p.unlink()
    print("Reset complete.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Customer Support Email Agent")
    ap.add_argument("command", choices=["run", "email", "metrics", "reset"])
    ap.add_argument("email_id", nargs="?")
    ap.add_argument("--provider", choices=["gemini", "openrouter", "mock"])
    ap.add_argument("--send", action="store_true", help="with 'email': actually send / remember")
    args = ap.parse_args(argv)

    if args.command == "reset":
        reset()
        return 0
    if args.command == "metrics":
        print(json.dumps(compute_metrics(load_traces()), indent=2))
        return 0

    settings = config.get_settings(llm_provider=args.provider)
    if args.command == "run":
        agent = SupportAgent(settings)
        results = agent.process_inbox()
        if not results:
            print("No unread emails. Run `python main.py reset` to start over.")
        for r in results:
            _print_result(r)
    else:
        if not args.email_id:
            ap.error("email requires an email id, e.g. E003")
        em = next((e for e in MockMailbox().all_emails() if e.id == args.email_id), None)
        if em is None:
            ap.error(f"no inbox email with id {args.email_id}")
        _print_result(SupportAgent(settings, persist=args.send).process_email(em))
    return 0


if __name__ == "__main__":
    sys.exit(main())
