"""Deterministic, rule-based stand-in for an LLM (LLM_PROVIDER=mock).

It speaks the same interface as the real client (messages + tools in, content or
tool calls out) so the full pipeline - planning, tool calling, guardrails, memory,
tracing, evaluation - can run offline and in tests. Its answers are templated, so it
demonstrates the plumbing, not language quality: use Gemini/OpenRouter for that.
"""
from __future__ import annotations

import json
import re
import uuid

from .llm import LLMResponse, ToolCall

INTENT_RULES = [
    ("human_escalation", r"\b(compensation|lawyer|legal action|dispute with my bank|chargeback|sue|unacceptable|ridiculous)\b"),
    ("returns_refunds", r"\b(refund|return(ed)?|cancel(led|lation)?|money back)\b"),
    ("warranty", r"\b(warranty|guarantee|replace(ment|d)?|repair)\b"),
    ("order_status", r"\b(where is|when will|arrive|delivery|delivered|shipping|shipped|tracking|delayed?|order status|status of order)\b"),
    ("troubleshooting", r"\b(error|not working|won'?t|doesn'?t|stopped|fix|keeps|drop(ping|s)? off|disconnect|battery|dead|slow|crash|issue|problem)\b"),
    ("product_info", r"\b(thinking of buying|does it (work|support)|compatible|specs?|how much|price|can it be|how fast|how many devices)\b"),
    ("general_faq", r"\b(account|password|email address|payment|contact|hours)\b"),
    ("spam", r"\b(won|winner|btc|crypto|lottery|prize)\b"),
]
RULES = dict(INTENT_RULES)
PRODUCT_WORDS = {
    "Echo Speaker 200": r"\b(echo|speaker)\b", "AirBuds Pro": r"\b(air ?buds|earbuds?)\b",
    "Pulse Watch 3": r"\b(pulse|watch)\b", "Mesh Router AX6": r"\b(router|mesh)\b",
    "RoboClean X1": r"\b(roboclean|vacuum)\b", "Slate 10": r"\b(slate|tablet)\b", "Guard Cam 2K": r"\b(guard ?cam|camera)\b",
}


def _between(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"mock_{uuid.uuid4().hex[:8]}", name=name, arguments=args)


class MockLLM:
    def chat(self, messages: list[dict], tools, *, purpose: str) -> LLMResponse:
        if purpose == "plan":
            out = LLMResponse(content=json.dumps(self._plan(messages[-1]["content"])))
        elif purpose == "judge":
            out = LLMResponse(content=json.dumps({"helpfulness": 4, "accuracy": 4, "tone": 4,
                                                  "comment": "mock judge - scores are placeholders"}))
        else:
            out = self._skill(messages, tools)
        out.model = "mock-rules-v1"
        out.prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        out.completion_tokens = len(out.content) // 4 + 20 * len(out.tool_calls)
        return out

    # ------------------------------------------------------------------ planning
    def _plan(self, prompt: str) -> dict:
        email = _between(prompt, "email") or prompt
        subject, _, body = email.partition("\n")
        # For replies, the subject repeats the old request: classify on the new body only.
        low = (body if subject.lower().startswith("subject: re:") else email).lower()
        intents = [name for name, pat in INTENT_RULES if re.search(pat, low)]
        if "spam" in intents and len(intents) > 1 and "http" not in low:
            intents.remove("spam")
        if "spam" in intents:
            intents = ["spam"]
        if "returns_refunds" in intents and "order_status" in intents:
            intents.remove("order_status")
        if "warranty" in intents and "troubleshooting" in intents:
            intents.remove("troubleshooting")
            intents.insert(1, "troubleshooting")
        intents = intents[:2] or ["general_faq"]
        products = [p for p, pat in PRODUCT_WORDS.items() if re.search(pat, low)]
        human = "human_escalation" in intents
        return {
            "intents": intents,
            "entities": {"order_ids": re.findall(r"ORD-\d+", email), "serial_numbers": re.findall(r"SN-[A-Z]{3}-\d+", email),
                         "products": products},
            "customer_goal": email.splitlines()[0].replace("Subject:", "").strip()[:120],
            "sentiment": "angry" if human else "neutral", "urgency": "high" if human else "normal",
            "requires_human": human, "human_reason": "customer demands compensation / threatens dispute" if human else None,
            "steps": [f"handle {i}" for i in intents], "confidence": 0.8,
        }

    # ------------------------------------------------------------------ skill loop
    def _skill(self, messages: list[dict], tools) -> LLMResponse:
        available = {t["function"]["name"] for t in tools or []}
        user = next(m["content"] for m in messages if m["role"] == "user")
        plan = json.loads(_between(user, "plan") or "{}")
        email = _between(user, "email")
        done = [m for m in messages if m["role"] == "tool"]
        if not done and available:
            calls = self._choose_tools(plan, email, available)
            if calls:
                return LLMResponse(tool_calls=calls)
        if done and available and not any(m.get("name") in ("get_device_info", "check_warranty") for m in done):
            follow = self._follow_up(plan, done, available)
            if follow:
                return LLMResponse(tool_calls=follow)
        return LLMResponse(content=self._compose(user, done))

    def _choose_tools(self, plan: dict, email: str, available: set[str]) -> list[ToolCall]:
        intents, ents = plan.get("intents", []), plan.get("entities", {})
        orders, serials, products = ents.get("order_ids", []), ents.get("serial_numbers", []), ents.get("products", [])
        subject = email.splitlines()[0].replace("Subject:", "").strip()
        calls: list[ToolCall] = []
        for intent in intents:
            if intent == "order_status":
                calls += [_call("get_order_status", order_id=o) for o in orders] or [_call("list_customer_orders")]
            elif intent == "warranty":
                calls += [_call("check_warranty", serial_number=s) for s in serials] or [_call("list_customer_devices")]
                calls.append(_call("search_knowledge_base", query="warranty claim out-of-warranty"))
            elif intent == "troubleshooting":
                # Only the products mentioned in the sentence that describes the problem.
                problem = next((s for s in re.split(r"(?<=[.?!])\s+", email) if re.search(RULES["troubleshooting"], s.lower())), email)
                prods = [p for p, pat in PRODUCT_WORDS.items() if re.search(pat, problem.lower())] or products
                symptom = " ".join(re.findall(r"error \w+|battery|wi-?fi|offline|disconnect\w*|charg\w*|sync\w*|dock|one (side|bud)|left|right",
                                              problem.lower()))
                calls.append(_call("search_knowledge_base", query=f"{' '.join(prods)} {symptom or problem[:120]}"))
                if not serials and "list_customer_devices" in available:
                    calls.append(_call("list_customer_devices"))
                calls += [_call("get_device_info", serial_number=s) for s in serials]
            elif intent == "product_info":
                calls += [_call("get_product_info", product=p) for p in products] or [_call("get_product_info", product=subject)]
            elif intent == "returns_refunds":
                calls += [_call("get_order_status", order_id=o) for o in orders]
                if re.search(r"(please|can you|could you|want to|like to)\s+(please\s+)?cancel", email.lower()) and orders:
                    calls.append(_call("request_order_cancellation", order_id=orders[0], reason="customer request"))
                calls.append(_call("search_knowledge_base",
                                   query="cancelling an order" if "cancel" in email.lower() else "refund timing return"))
            elif intent == "general_faq":
                calls.append(_call("search_knowledge_base", query=subject))
            elif intent == "human_escalation":
                calls += [_call("get_order_status", order_id=o) for o in orders]
                calls.append(_call("escalate_to_human", reason=plan.get("human_reason") or "sensitive case", priority="high"))
        seen, unique = set(), []
        for c in calls:
            key = (c.name, json.dumps(c.arguments, sort_keys=True))
            if c.name in available and key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def _follow_up(self, plan: dict, done: list[dict], available: set[str]) -> list[ToolCall]:
        """After list_customer_devices, inspect the device that matches the product in the email."""
        for m in done:
            if m.get("name") != "list_customer_devices":
                continue
            devices = json.loads(m["content"]).get("devices", [])
            wanted = [p.lower() for p in plan.get("entities", {}).get("products", [])]
            match = [d for d in devices if any(w.split()[0] in d["product"].lower() for w in wanted)] or devices[:1]
            if match:
                tool = "check_warranty" if "warranty" in plan.get("intents", []) else "get_device_info"
                if tool in available:
                    return [_call(tool, serial_number=match[0]["serial_number"])]
        return []

    # ------------------------------------------------------------------ reply
    def _compose(self, user: str, done: list[dict]) -> str:
        m = re.search(r"Customer first name: (\S+)", user)
        name = m.group(1) if m and m.group(1) != "unknown" else ""
        parts = []
        for msg in done:
            try:
                parts.append(self._render(msg.get("name", ""), json.loads(msg["content"])))
            except (ValueError, KeyError):
                continue
        parts = [p for p in dict.fromkeys(parts) if p]
        if not parts:
            parts = ["Thank you for your message. A member of our team will look into it and get back to you."]
        greeting = f"Hi {name}," if name else "Hello,"
        return greeting + "\n\n" + "\n\n".join(parts) + "\n\nBest regards,\nNimbus Support"

    @staticmethod
    def _render(tool: str, r: dict) -> str:
        err = r.get("error")
        if err == "sender_not_verified":
            return ("For your security, we can only share order and account details with the email address "
                    "registered on the account. Please write to us from that address and we will be happy to help.")
        if err == "not_found":
            return "I couldn't find that order or device on your account. Could you double-check the number?"
        if err == "not_cancellable":
            return (f"This order can no longer be cancelled because its status is '{r['status']}'. "
                    "You can return it within 30 days of delivery instead.")
        if err:
            return ""
        if tool == "get_order_status":
            items = ", ".join(f"{i['quantity']} x {i['product']}" for i in r["items"])
            s = f"Your order {r['order_id']} ({items}) is currently {r['status'].replace('_', ' ')}."
            if r.get("carrier") and r.get("tracking_number"):
                s += f" It is with {r['carrier']}, tracking number {r['tracking_number']}."
            if r.get("estimated_delivery") and r["status"] not in ("delivered", "cancelled", "returned"):
                s += f" The estimated delivery date is {r['estimated_delivery']}."
            if r.get("note"):
                s += f" {r['note']}"
            return s
        if tool == "list_customer_orders":
            return "Orders on your account: " + "; ".join(f"{o['order_id']} ({o['status']})" for o in r["orders"]) + "."
        if tool == "check_warranty":
            if r["status"] == "active":
                return (f"Your {r['product']} ({r['serial_number']}) is covered by the {r['plan']} until "
                        f"{r['end_date']}.")
            return (f"The warranty for your {r['product']} ({r['serial_number']}) ended on {r['end_date']}, so it is "
                    "no longer covered.")
        if tool == "get_device_info":
            if not r["firmware_up_to_date"]:
                return (f"Your {r['product']} is running firmware {r['firmware_installed']}; the latest version is "
                        f"{r['firmware_latest']}. Please update it first in the Nimbus app, as updates fix known issues.")
            return f"Your {r['product']} is on the latest firmware ({r['firmware_installed']})."
        if tool == "search_knowledge_base" and r.get("results"):
            top = r["results"][0]
            return f"From our guide \"{top['section']}\":\n{top['content']}"
        if tool == "get_product_info":
            specs = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in list(r["specs"].items())[:6])
            return f"The {r['name']} costs ${r['price']:.2f} and has a {r['warranty_months']}-month warranty.\n{specs}"
        if tool == "request_order_cancellation":
            return f"I've submitted a cancellation request for order {r['order_id']}. {r['message']}"
        if tool == "escalate_to_human":
            return ("I'm sorry for the trouble. I've passed your message to a specialist on our customer care team, "
                    "who will reply within 2 business days.")
        return ""
