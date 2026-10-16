"""Guardrails - deterministic checks around the LLM.

Input guardrails (before the LLM sees the email)
  * spam / phishing detection
  * prompt-injection detection (attempts to override instructions or extract data)
  * PII redaction (card numbers, SSNs) so they are never sent to the LLM or logs
  * size limit

Output guardrails (before a reply is sent)
  * grounding: every order ID, serial, tracking number, ISO date and $ amount in the
    reply must appear in the evidence (tool results or the customer's own email)
  * policy: no claims of actions the agent cannot take (refunds issued, replacements
    approved, compensation, guarantees)
  * privacy: no email addresses other than the sender's or the support address
  * hygiene: no placeholders, internal jargon or empty/oversized replies

Authorization (whose data may be read) is enforced inside the tools themselves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_EMAIL_CHARS = 8000
MAX_REPLY_CHARS = 2500

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |your )?(previous |prior )?(instructions|rules|guidelines)",
    r"you are now (in )?\w+",
    r"\b(admin|developer|debug|god) mode\b",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"system prompt",
    r"\b(list|send|give|dump|export)\b.{0,40}\b(all|every|other)\b.{0,20}\b(customers?|users?|emails|accounts)\b",
    r"act as (an? )?(admin|administrator|developer|system)",
    r"</?(system|assistant|instructions?)>",
]
SPAM_PATTERNS = [
    r"\b(you('ve)? won|winner|congratulations)\b", r"\b(btc|bitcoin|crypto|lottery|prize)\b",
    r"\bclick (here|the link|now)\b|https?://\S+/claim", r"\b(act now|limited time|100% free)\b",
    r"\bviagra|casino|loan approval\b",
]
PII_PATTERNS = {
    "CARD": r"\b(?:\d[ -]?){13,16}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
}

FORBIDDEN_CLAIMS = [
    (r"\b(i|we)\s*('ve|\s+have)?\s+(issued|processed|approved|initiated|authori[sz]ed)\s+(a|your|the)?\s*"
     r"(full\s+|partial\s+)?(refund|replacement|compensation|credit|return)", "claims an action the agent cannot take"),
    (r"\b(i|we)\s*('ll|\s+will)\s+(refund|compensate|reimburse|replace|send you a (new|replacement))\b",
     "promises a refund/replacement/compensation"),
    (r"\bguarantee[ds]?\b", "makes a guarantee"),
    (r"\b(\$\d+|\d+%)\s+(credit|voucher|discount code|compensation)\b", "offers ad-hoc compensation"),
]
INTERNAL_TERMS = [r"system prompt", r"\bjson\b", r"function call", r"tool call", r"as an ai\b",
                  r"language model", r"sender_not_verified", r"not_found\b", r"<email>"]
PLACEHOLDERS = [r"\[(your|customer|agent|insert)[^\]]*\]", r"\{\{?\s*\w+\s*\}?\}", r"\bXXX+\b", r"lorem ipsum"]


@dataclass
class GuardResult:
    passed: bool
    violations: list[dict] = field(default_factory=list)
    flags: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "violations": self.violations, "flags": self.flags}


def _matches(patterns, text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, flags=re.IGNORECASE)]


# ------------------------------------------------------------------ input
def redact_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, pat in PII_PATTERNS.items():
        if re.search(pat, text):
            found.append(label)
            text = re.sub(pat, f"[REDACTED {label}]", text)
    return text, found


def check_input(subject: str, body: str, sender_is_customer: bool) -> GuardResult:
    text = f"{subject}\n{body}"
    injection = _matches(INJECTION_PATTERNS, text)
    spam = _matches(SPAM_PATTERNS, text)
    _, pii = redact_pii(body)
    flags = {
        "prompt_injection": bool(injection),
        "injection_patterns": injection,
        # Known customers are never auto-classified as spam by heuristics alone.
        "spam": len(spam) >= 2 and not sender_is_customer,
        "spam_signals": len(spam),
        "pii_redacted": pii,
        "too_long": len(body) > MAX_EMAIL_CHARS,
    }
    violations = []
    if flags["prompt_injection"]:
        violations.append({"check": "prompt_injection", "detail": f"matched {len(injection)} pattern(s)"})
    if flags["spam"]:
        violations.append({"check": "spam", "detail": f"{len(spam)} spam signals"})
    return GuardResult(passed=not violations, violations=violations, flags=flags)


# ------------------------------------------------------------------ output
_ID_PATTERNS = {
    "order_id": r"\bORD-\d{4,}\b",
    "serial_number": r"\bSN-[A-Z]{3}-\d{4,}\b",
    "tracking_number": r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{12,22}\b",
    "date": r"\b\d{4}-\d{2}-\d{2}\b",
}


def _amounts(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in re.findall(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)", text)}


def _numbers(text: str) -> set[float]:
    return {float(m) for m in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", text)}


def check_output(reply: str, evidence: str, customer_email_text: str, sender_email: str,
                 support_address: str, known_emails: set[str] | None = None) -> GuardResult:
    violations: list[dict] = []
    reply = reply or ""
    source = f"{evidence}\n{customer_email_text}"

    if len(reply.strip()) < 20:
        violations.append({"check": "empty_reply", "detail": "reply is empty or too short"})
    if len(reply) > MAX_REPLY_CHARS:
        violations.append({"check": "too_long", "detail": f"{len(reply)} chars > {MAX_REPLY_CHARS}"})

    # Grounding: identifiers and figures must come from evidence.
    for kind, pat in _ID_PATTERNS.items():
        for value in set(re.findall(pat, reply)):
            if value.lower() not in source.lower():
                violations.append({"check": "ungrounded_fact", "detail": f"{kind} '{value}' not found in tool results"})
    allowed_amounts = _amounts(source) | _numbers(source)
    for amt in _amounts(reply):
        if amt not in allowed_amounts:
            violations.append({"check": "ungrounded_fact", "detail": f"amount ${amt:g} not found in tool results"})

    for pat, why in FORBIDDEN_CLAIMS:
        m = re.search(pat, reply, flags=re.IGNORECASE)
        if m:
            violations.append({"check": "policy", "detail": f"{why}: '{m.group(0)}'"})

    allowed = {sender_email.lower(), support_address.lower()}
    for addr in set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", reply)):
        if addr.lower().rstrip(".") not in allowed:
            detail = "another customer's email address" if known_emails and addr.lower() in known_emails else addr
            violations.append({"check": "privacy", "detail": f"reply contains {detail}"})

    for pat in _matches(INTERNAL_TERMS, reply):
        violations.append({"check": "internal_leak", "detail": f"mentions internal term /{pat}/"})
    for pat in _matches(PLACEHOLDERS, reply):
        violations.append({"check": "placeholder", "detail": f"contains placeholder /{pat}/"})

    return GuardResult(passed=not violations, violations=violations)


def feedback_for(result: GuardResult) -> str:
    lines = "\n".join(f"- {v['check']}: {v['detail']}" for v in result.violations)
    return ("Your draft reply failed automated quality checks:\n" + lines +
            "\nRewrite the reply to fix every issue. Only state facts that appear in tool results, copy "
            "identifiers/dates/amounts exactly, and do not promise refunds, replacements or compensation. "
            "Return only the corrected email body.")
