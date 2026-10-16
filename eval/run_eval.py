"""Offline evaluation of the support agent.

Runs every case in eval/dataset.json through the agent in dry-run mode (nothing is
sent, nothing is remembered) and scores:

  intent_accuracy   primary intent is one of the accepted intents
  action_accuracy   sent / escalated / ignored matches expectation
  tool_recall       expected tools were called ("a|b" = either is fine)
  fact_recall       required facts appear in the reply ("a|b" = any alternative)
  safety_pass       none of the forbidden strings appear in the reply (data leaks, false promises)
  guardrail_pass    reply passed the output guardrails
  judge (optional)  LLM-as-judge 1-5 for helpfulness / accuracy / tone, given the tool evidence

    python -m eval.run_eval                     # uses LLM_PROVIDER from .env
    python -m eval.run_eval --provider mock     # offline baseline
    python -m eval.run_eval --judge --delay 4   # add LLM judge, pause between cases (free-tier rate limits)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from agent import config
from agent.email_client import Email, MockMailbox
from agent.llm import LLMClient, LLMError
from agent.orchestrator import SupportAgent

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE / "reports"

JUDGE_PROMPT = """You are grading a customer support email reply written by an AI agent for Nimbus Electronics.
You get the customer email, the facts the agent retrieved (tool results), and the reply.
Score 1-5 (5 = best):
- helpfulness: does it fully address every customer question with clear next steps?
- accuracy: are all statements supported by the retrieved facts (no invented details)?
- tone: polite, empathetic, professional and concise?
Return JSON only: {"helpfulness": n, "accuracy": n, "tone": n, "comment": "one sentence"}"""


def _any(alternatives: str, text: str) -> bool:
    return any(a.strip().lower() in text.lower() for a in alternatives.split("|"))


def load_cases(path: Path = HERE / "dataset.json") -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    inbox = {e.id: e for e in MockMailbox().all_emails()}
    for c in cases:
        if "inbox_id" in c:
            c["_email"] = inbox[c["inbox_id"]]
        else:
            c["_email"] = Email(id=f"EVAL-{c['id']}", thread_id=f"EVAL-{c['id']}", **c["email"])
    return cases


def judge(llm: LLMClient, em: Email, evidence: list[dict], reply: str) -> dict:
    user = (f"Customer email:\nSubject: {em.subject}\n{em.body}\n\nTool results:\n"
            f"{json.dumps([t['result'] for t in evidence], default=str)[:6000]}\n\nAgent reply:\n{reply}")
    try:
        out = llm.chat_json([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
                            purpose="judge")
        return {k: out.get(k) for k in ("helpfulness", "accuracy", "tone", "comment")}
    except LLMError as exc:
        return {"error": str(exc)}


def score_case(case: dict, result, trace: dict | None) -> dict:
    exp, reply = case["expect"], result.reply or ""
    tools_used = set(result.tools_used)
    tools_ok = [_any_tool(t, tools_used) for t in exp.get("tools", [])]
    facts_ok = [_any(f, reply) for f in exp.get("must_include", [])]
    leaks = [f for f in exp.get("must_not_include", []) if _any(f, reply)]
    llm_spans = [s for s in (trace or {}).get("spans", []) if s["kind"] == "llm"]
    return {
        "id": case["id"],
        "intent": result.intent,
        "action": result.action,
        "intent_ok": result.intent in exp["intents"],
        "action_ok": result.action in exp["actions"],
        "tool_recall": sum(tools_ok) / len(tools_ok) if tools_ok else 1.0,
        "fact_recall": sum(facts_ok) / len(facts_ok) if facts_ok else 1.0,
        "safety_ok": not leaks,
        "leaks": leaks,
        "guardrail_ok": result.output_guardrails.get("passed", True) if result.output_guardrails else True,
        "revisions": result.revisions,
        "tools_used": result.tools_used,
        "latency_ms": (trace or {}).get("duration_ms", 0.0),
        "llm_calls": len(llm_spans),
        "tokens": sum((s["attributes"].get("prompt_tokens") or 0) + (s["attributes"].get("completion_tokens") or 0)
                      for s in llm_spans),
        "error": result.error,
        "reply": reply,
    }


def _any_tool(spec: str, used: set[str]) -> bool:
    return any(t in used for t in spec.split("|"))


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    mean = lambda k: round(sum(float(r[k]) for r in rows) / n, 3) if n else 0.0  # noqa: E731
    summary = {
        "cases": n,
        "intent_accuracy": mean("intent_ok"),
        "action_accuracy": mean("action_ok"),
        "tool_recall": mean("tool_recall"),
        "fact_recall": mean("fact_recall"),
        "safety_pass_rate": mean("safety_ok"),
        "guardrail_pass_rate": mean("guardrail_ok"),
        "errors": sum(1 for r in rows if r["error"]),
        "latency_ms_mean": round(statistics.mean(r["latency_ms"] for r in rows), 1) if n else 0.0,
        "tokens_mean": round(statistics.mean(r["tokens"] for r in rows), 1) if n else 0.0,
    }
    judged = [r["judge"] for r in rows if isinstance(r.get("judge"), dict) and "helpfulness" in r["judge"]]
    for k in ("helpfulness", "accuracy", "tone"):
        vals = [j[k] for j in judged if isinstance(j.get(k), (int, float))]
        if vals:
            summary[f"judge_{k}"] = round(statistics.mean(vals), 2)
    return summary


def run(provider: str | None = None, use_judge: bool = False, delay: float = 0.0, only: list[str] | None = None,
        progress=None) -> dict:
    settings = config.get_settings(llm_provider=provider)
    agent = SupportAgent(settings, persist=False)
    judge_llm = LLMClient(settings) if use_judge else None
    cases = [c for c in load_cases() if not only or c["id"] in only]
    rows, stopped_early = [], None
    for i, case in enumerate(cases):
        result = agent.process_email(case["_email"])
        if result.error and "QuotaExhausted" in result.error:
            print(f"Stopping early: {result.error}", flush=True)
            stopped_early = result.error
            break
        row = score_case(case, result, agent.last_trace)
        if judge_llm and result.reply:
            row["judge"] = judge(judge_llm, case["_email"], result.tool_calls, result.reply)
        rows.append(row)
        if progress:
            progress(i + 1, len(cases), row)
        if delay and i < len(cases) - 1:
            time.sleep(delay)
    report = {"created_at": datetime.now().isoformat(timespec="seconds"), "provider": agent.llm.provider,
              "model": agent.llm.model, "summary": summarise(rows), "cases": rows,
              "stopped_early": stopped_early}
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in (f"report_{agent.llm.provider}_{stamp}.json", "latest.json"):
        (REPORT_DIR / name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _print(report: dict) -> None:
    print(f"\nProvider: {report['provider']} ({report['model']})")
    print(f"{'case':42} {'intent':18} {'action':10} int act tool fact safe guard")
    for r in report["cases"]:
        ok = lambda b: " ok" if b else " XX"  # noqa: E731
        print(f"{r['id'][:42]:42} {r['intent'][:18]:18} {r['action']:10}{ok(r['intent_ok'])} {ok(r['action_ok'])}"
              f" {r['tool_recall']:4.2f} {r['fact_recall']:4.2f} {ok(r['safety_ok'])} {ok(r['guardrail_ok'])}"
              + (f"  leaks={r['leaks']}" if r["leaks"] else "") + (f"  ERROR {r['error']}" if r["error"] else ""))
    print("\nSummary:")
    for k, v in report["summary"].items():
        print(f"  {k:22} {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["gemini", "openrouter", "mock"])
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge scoring")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between cases (free-tier rate limits)")
    ap.add_argument("--only", nargs="*", help="run only these case ids")
    a = ap.parse_args()
    _print(run(a.provider, a.judge, a.delay, a.only,
               progress=lambda i, n, r: print(f"[{i}/{n}] {r['id']}: {r['action']}", flush=True)))
