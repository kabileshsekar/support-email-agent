"""Orchestration: the end-to-end pipeline for one email.

    input guardrails -> verify sender + load memory -> plan -> route
        -> skill loop (LLM <-> tools, bounded) -> output guardrails (+1 self-correction)
        -> send reply | escalate to human | ignore spam
        -> save memory, finish trace

Every stage runs inside a span of the email's trace.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import guardrails, tracing
from .config import Settings, get_settings
from .db import SupportDB
from .email_client import Email, ReviewQueue, get_mailbox
from .llm import LLMClient, LLMError
from .memory import ConversationMemory, WorkingMemory, format_history
from .observability import get_logger, log_event
from .planner import make_plan
from .skills import AGENT_SYSTEM_PROMPT, compose
from .tools import ToolContext, execute, schemas, to_message

log = get_logger("orchestrator")


@dataclass
class AgentResult:
    email_id: str
    action: str                      # sent | escalated | review | ignored | error
    intent: str = ""
    plan: dict = field(default_factory=dict)
    skill: str = ""
    reply: str = ""
    tools_used: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    input_guardrails: dict = field(default_factory=dict)
    output_guardrails: dict = field(default_factory=dict)
    escalation: Optional[dict] = None
    actions: list[dict] = field(default_factory=list)
    revisions: int = 0
    customer_verified: bool = False
    trace_id: str = ""
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


class SupportAgent:
    def __init__(self, settings: Optional[Settings] = None, mailbox=None, persist: bool = True):
        self.settings = settings or get_settings()
        self.llm = LLMClient(self.settings)
        self.db = SupportDB()
        self.memory = ConversationMemory()
        self.mailbox = mailbox or get_mailbox(self.settings)
        self.review_queue = ReviewQueue()
        self.persist = persist          # False = dry run (evaluation): nothing is sent or remembered
        self._known_emails = self.db.all_customer_emails()
        self.last_trace: Optional[dict] = None   # full trace of the most recent email (UI + evaluation)

    # ------------------------------------------------------------------ public
    def process_inbox(self) -> list[AgentResult]:
        results = []
        for em in self.mailbox.fetch_unread():
            results.append(self.process_email(em))
        return results

    def process_email(self, em: Email) -> AgentResult:
        meta = {"email_id": em.id, "thread_id": em.thread_id, "from": em.from_email, "subject": em.subject,
                "provider": self.llm.provider, "model": self.llm.model}
        with tracing.start_trace("process_email", meta, persist=self.persist) as trace:
            result = AgentResult(email_id=em.id, action="error", trace_id=trace.trace_id)
            try:
                self._run(em, result)
                status = "ok"
            except Exception as exc:
                log_event(log, "pipeline error", email_id=em.id, error=str(exc))
                result.error = f"{type(exc).__name__}: {exc}"
                result.action = "error"
                status = "error"
                if self.persist:
                    self.review_queue.add(em, f"agent error: {result.error}", "high", result.reply, trace.trace_id)
            self.last_trace = trace.finish(status, {"action": result.action, "intent": result.intent,
                                  "tools_used": result.tools_used, "escalated": result.action == "escalated"})
            if self.persist and result.action != "error" and hasattr(self.mailbox, "mark_processed"):
                self.mailbox.mark_processed(em.id)
            log_event(log, "email processed", email_id=em.id, action=result.action, intent=result.intent,
                      trace_id=trace.trace_id, ms=trace.duration_ms)
            return result

    # ------------------------------------------------------------------ pipeline
    def _run(self, em: Email, result: AgentResult) -> None:
        s = self.settings
        customer = self.db.customer_by_email(em.from_email)
        result.customer_verified = customer is not None

        # 1. Input guardrails
        with tracing.span("input_guardrails", kind="guardrail") as sp:
            body, _ = guardrails.redact_pii(em.body[:guardrails.MAX_EMAIL_CHARS])
            gin = guardrails.check_input(em.subject, em.body, sender_is_customer=customer is not None)
            result.input_guardrails = gin.as_dict()
            sp.set(passed=gin.passed, **{k: v for k, v in gin.flags.items() if not isinstance(v, list)})
            sp.set_output(gin.as_dict())

        if gin.flags["spam"]:
            result.action, result.intent = "ignored", "spam"
            return

        # 2. Memory: who is this and what have we discussed before?
        with tracing.span("load_memory", kind="memory") as sp:
            history_rows = self.memory.history(em.from_email, em.thread_id)
            history = format_history(history_rows)
            sender_ctx = (f"verified customer {customer['name']} (id {customer['customer_id']}, tier {customer['tier']})"
                          if customer else f"UNVERIFIED - {em.from_email} is not a registered customer email")
            sp.set(customer_verified=customer is not None, history_items=len(history_rows))
            sp.set_output({"sender": sender_ctx, "history": history})

        # 3. Prompt injection => never hand the email to the tool loop; a human reviews it.
        if gin.flags["prompt_injection"]:
            result.intent = "security"
            self._escalate(em, result, "Possible prompt-injection / data-exfiltration attempt", "high", draft="")
            return

        # 4. Plan
        work = WorkingMemory()
        plan = make_plan(self.llm, em.subject, body, sender_ctx, history)
        work.plan = plan
        result.plan, result.intent = plan, plan["intents"][0]
        if "spam" in plan["intents"] and customer is None:
            result.action = "ignored"
            return

        # 5. Skill loop
        skill = compose([i for i in plan["intents"] if i != "spam"])
        if plan["requires_human"] and "human_escalation" not in skill.name:
            skill = compose(plan["intents"] + ["human_escalation"])
        result.skill = skill.name
        ctx = ToolContext(sender_email=em.from_email, customer=customer, db=self.db, dry_run=not self.persist)
        messages = self._initial_messages(em, body, skill, plan, sender_ctx, history, customer)
        reply = self._skill_loop(messages, skill, ctx, work)

        # 6. Output guardrails with one self-correction attempt
        gout = self._check_output(reply, work, em)
        if not gout.passed:
            result.revisions = 1
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": guardrails.feedback_for(gout)}]
            reply = self._skill_loop(messages, skill, ctx, work)
            gout = self._check_output(reply, work, em)

        result.reply, result.output_guardrails = reply, gout.as_dict()
        result.tools_used, result.tool_calls, result.actions = work.tools_used, work.tool_calls, ctx.actions

        # 7. Decide: send, escalate or hold for review
        if not gout.passed:
            reasons = "; ".join(v["detail"] for v in gout.violations)
            self._escalate(em, result, f"Reply failed guardrails after revision: {reasons}", "normal", reply)
        elif plan["confidence"] < s.min_plan_confidence:
            self._escalate(em, result, f"Low planner confidence ({plan['confidence']:.2f})", "normal", reply)
        elif ctx.escalation:
            # The acknowledgement passed the guardrails, so the customer gets it while a human takes over.
            self._send(em, reply)
            self._escalate(em, result, ctx.escalation["reason"], ctx.escalation.get("priority", "normal"), reply,
                           ack_sent=True)
        elif not s.auto_send:
            self._escalate(em, result, "AUTO_SEND disabled - draft awaiting approval", "low", reply, action="review")
        else:
            self._send(em, reply)
            result.action = "sent"

        # 8. Long-term memory
        if self.persist:
            with tracing.span("save_memory", kind="memory"):
                self.memory.save(email_id=em.id, thread_id=em.thread_id, customer_email=em.from_email,
                                 subject=em.subject, intents=plan["intents"],
                                 summary=f"Goal: {plan['customer_goal']} Outcome: {result.action}; tools: "
                                         f"{', '.join(work.tools_used) or 'none'}. Reply: {reply[:200]}",
                                 reply=reply, action=result.action)

    # ------------------------------------------------------------------ helpers
    def _initial_messages(self, em, body, skill, plan, sender_ctx, history, customer) -> list[dict]:
        system = AGENT_SYSTEM_PROMPT.format(skill_name=skill.name, skill_instructions=skill.instructions)
        first_name = customer["name"].split()[0] if customer else (em.from_name.split()[0] if em.from_name else "")
        user = (f"Sender: {sender_ctx}\nCustomer first name: {first_name or 'unknown'}\n"
                f"Previous interactions:\n{history}\n\n"
                f"Plan:\n<plan>{json.dumps(plan)}</plan>\n\n"
                f"<email>\nSubject: {em.subject}\n\n{body}\n</email>\n\n"
                "Use the tools you need, then write the reply.")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _skill_loop(self, messages: list[dict], skill, ctx: ToolContext, work: WorkingMemory) -> str:
        tool_defs = schemas(list(skill.tools))
        with tracing.span(f"skill:{skill.name}", kind="step", tools=list(skill.tools)) as sp:
            for step in range(self.settings.max_tool_steps):
                resp = self.llm.chat(messages, tool_defs, purpose="skill")
                if not resp.tool_calls:
                    sp.set(steps=step + 1)
                    return resp.content.strip()
                messages.append(resp.as_message())
                for tc in resp.tool_calls:
                    out = execute(tc.name, tc.arguments, ctx, allowed=list(skill.tools))
                    work.record_tool(tc.name, tc.arguments, out)
                    messages.append(to_message(tc.id, tc.name, out))
            # Step budget exhausted: force a final answer without tools.
            sp.set(steps=self.settings.max_tool_steps, budget_exhausted=True)
            messages.append({"role": "user", "content": "Tool budget reached. Write the final reply now."})
            return self.llm.chat(messages, tool_defs, purpose="skill", tool_choice="none").content.strip()

    def _check_output(self, reply: str, work: WorkingMemory, em: Email):
        with tracing.span("output_guardrails", kind="guardrail", input=reply) as sp:
            g = guardrails.check_output(reply, work.evidence_text(), f"{em.subject}\n{em.body}", em.from_email,
                                        self.settings.support_address, self._known_emails)
            sp.set(passed=g.passed, violations=len(g.violations))
            sp.set_output(g.as_dict())
            return g

    def _send(self, em: Email, reply: str) -> None:
        if not self.persist:
            return
        with tracing.span("send_email", kind="email", input={"to": em.from_email}):
            subject = em.subject if em.subject.lower().startswith("re:") else f"Re: {em.subject}"
            self.mailbox.send(em.from_email, subject, reply, in_reply_to=em.id)

    def _escalate(self, em, result: AgentResult, reason: str, priority: str, draft: str,
                  ack_sent: bool = False, action: str = "escalated") -> None:
        result.action = action
        result.escalation = {"reason": reason, "priority": priority, "acknowledgement_sent": ack_sent}
        if self.persist:
            with tracing.span("escalate", kind="email", input=result.escalation):
                ticket = self.review_queue.add(em, reason, priority, draft, result.trace_id)
                result.escalation["ticket_id"] = ticket["ticket_id"]


def build_agent(**settings_overrides) -> SupportAgent:
    return SupportAgent(get_settings(**settings_overrides))


__all__ = ["SupportAgent", "AgentResult", "build_agent", "LLMError"]
