# Customer Support Email Agent

An AI agent that reads customer emails, finds the information it needs in a **knowledge base** (FAQs, manuals,
troubleshooting guides) and a **device & order database** (customers, orders, devices, specs, warranty), and
sends a helpful, grounded reply. Anything risky or outside its authority goes to a human.

The demo company is a fictional electronics brand, **Nimbus Electronics**, with sample data included.

```
             ┌──────────────── trace (every step is a span) ─────────────────┐
 Inbox ──▶ Input guardrails ──▶ Memory ──▶ Planner ──▶ Skill loop ──▶ Output guardrails ──▶ Send
           spam · injection     who is     intents,    LLM ⇄ tools     grounding · policy   │
           PII redaction        this?      entities,   (bounded,       privacy · hygiene    ├─▶ Escalate
                                history    plan        authorized)     +1 self-correction   └─▶ Ignore
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env  -> then add your GEMINI_API_KEY
python main.py reset              # creates the sample database
streamlit run app.py              # demo UI
```

Get a free Gemini key at <https://aistudio.google.com/apikey>, or set `LLM_PROVIDER=openrouter` and use a free
OpenRouter key (<https://openrouter.ai/keys>) with any `:free` model. **No key?** Use `LLM_PROVIDER=mock`: a
rule-based stand-in that runs the whole pipeline offline. It's useful for checking the plumbing, but its replies
are templated.

### CLI

```bash
python main.py run                      # process every unread email in data/inbox.json
python main.py email E005               # one email, dry run (add --send to send + remember)
python main.py run --provider mock      # offline
python main.py metrics                  # observability metrics from the trace store
python -m eval.run_eval --judge --delay 4   # evaluation + LLM judge (~125 requests, spread over the fallback models)
pytest -q                               # unit + integration tests (offline)
```

## How each required component is implemented

| Component | Where | What it does |
|---|---|---|
| **Planning** | `agent/planner.py` | One LLM call returns JSON with intents (several allowed), entities (order IDs, serials, products), sentiment, urgency, ordered steps, whether a human is needed, and a confidence score. Order IDs and serial numbers are also pulled out with regex and merged in, so they never depend on the model getting them right. If planning fails, the email gets a low-confidence plan and goes to human review. |
| **Skills** | `agent/skills.py` | Seven playbooks: `order_status`, `warranty`, `troubleshooting`, `product_info`, `returns_refunds`, `general_faq`, `human_escalation`. Each one has its own instructions and **only the tools it needs**. Emails with several requests get the matching skills combined. |
| **Tools** | `agent/tools.py`, `agent/kb.py`, `agent/db.py` | `search_knowledge_base`, `get_customer_profile`, `list_customer_orders`, `get_order_status`, `list_customer_devices`, `get_device_info`, `check_warranty`, `get_product_info`, `request_order_cancellation`, `escalate_to_human`, all exposed through JSON-schema function calling. KB search uses TF-IDF over markdown sections. |
| **Orchestration** | `agent/orchestrator.py` | The pipeline shown above. The tool loop has a step limit (`MAX_TOOL_STEPS`). Routing chooses send, escalate, hold for review, or ignore. The email is sent by the orchestrator only after the output checks pass; sending is **not** a tool the LLM can call. |
| **Memory** | `agent/memory.py` | *Working memory*: plan, tool calls and evidence for the current email. *Long-term memory*: every interaction is stored in SQLite with a summary, and later emails in the same thread or from the same customer get that history (see inbox email E014, a follow-up to E001). |
| **Guardrails** | `agent/guardrails.py`, `agent/tools.py` | **Input**: spam, prompt injection (sent straight to a human, never into the tool loop), PII redaction, size limit. **Authorization**: the sender's email is checked against the customer record, and account tools only return that customer's records. Someone else's order looks exactly like a non-existent one. **Output**: every order ID, serial, tracking number, ISO date and $ amount must appear in tool results or the customer's email. No claimed refunds, replacements, compensation or guarantees. No other people's email addresses, no internal jargon, no placeholders. A failed draft gets one rewrite with feedback, then goes to a human. |
| **Tracing** | `agent/tracing.py` | One trace per email. Each step is a span (`step`, `llm`, `tool`, `guardrail`, `memory`, `email`) with input, output, latency, tokens and errors, kept as a tree and stored in `runtime/traces.jsonl`. The UI shows the full tree. |
| **Observability** | `agent/observability.py` | JSON-line logs in `runtime/agent.log`. Metrics computed from traces: p50/p95 latency, LLM calls, tokens per email, tool usage and error rate, input blocks, output-guardrail failures, auto-resolve and escalation rates, intent mix. Shown on the **Dashboard** tab and by `python main.py metrics`. |
| **Evaluation** | `eval/` | 25 labelled cases covering the 13 real support emails in the inbox plus 12 more: edge cases, injection, PII, and a customer asking about someone else's order. Scores intent accuracy, action accuracy, tool recall, fact recall, safety (forbidden strings such as leaked data or false promises) and guardrail pass rate, plus an optional LLM-as-judge score for helpfulness, accuracy and tone. Reports go to `eval/reports/`. |

## Project layout

```
agent/        config, llm (Gemini/OpenRouter/mock), planner, skills, tools, kb, db,
              memory, guardrails, orchestrator, tracing, observability, email_client
data/         seed.py (sample DB), kb/*.md (12 knowledge-base docs), inbox.json (14 sample emails)
eval/         dataset.json, run_eval.py, reports/
tests/        pytest suite (offline)
app.py        Streamlit UI       main.py   CLI
runtime/      created at run time: support.db, memory.db, traces.jsonl, agent.log, outbox, escalations
```

## Sample inbox: what each email exercises

| Email | Scenario |
|---|---|
| E001, E002 | Order status: shipped, and delayed with a carrier note |
| E003 | Watch battery problem; the agent spots outdated firmware |
| E004 | Warranty check on a device whose warranty has expired |
| E005 | Two requests in one email: vacuum error code and order status |
| E006 | Unknown sender asks about someone else's order; no data is shared |
| E007 | Prompt injection asking for a refund and a list of all customers; sent to a human |
| E008 | Refund status for a returned item |
| E009 | Pre-sales product question from someone who isn't a customer |
| E010 | Compensation demand and chargeback threat; acknowledgement sent, escalated |
| E011 | Cancellation request (an action tool, allowed only while the order is `processing`) |
| E012 | Spam; ignored |
| E013 | Account FAQ |
| E014 | Follow-up in E001's thread; uses memory |

## Evaluation results

The 25-case test set, run on Gemini (free tier) with the LLM-as-judge. The full report is in `eval/reports/latest.json`
and can be viewed in the **🧪 Evaluation** tab.

| Metric | Gemini | Offline mock (baseline) |
|---|---|---|
| Intent accuracy | 100% | 96% |
| Action accuracy (send / escalate / ignore) | 100% | 100% |
| Tool recall | 100% | 96% |
| Fact recall | 100% | 86% |
| Safety (no leaks / false promises) | 100% | 100% |
| Output guardrail pass rate | 100% | 100% |
| Judge: helpfulness / accuracy / tone | 5.0 / 4.88 / 4.94 | n/a |

**What the evaluation caught and what was fixed.** The first Gemini run had two failures:
1. The agent escalated a fixable earbud problem to the warranty team before the customer had tried any
   troubleshooting. The warranty and troubleshooting skills now tell it to give the steps first and only escalate if
   the customer has already tried them.
2. Google returned "503 model overloaded" five times in a row. The client now switches to the next model.

Both cases, plus three related ones, were re-run after the fixes and passed.

**Notes on these numbers**
* Most cases ran on the fallback model `gemini-flash-lite-latest`, because the main model's free quota ran out (see
  below).
* Average time per email was about 20 s. Each email needs 3–5 LLM calls, and the free tier adds waiting on top.
* 25 cases is a small test set. It checks the main behaviours and safety rules but doesn't prove quality at scale.

### Free-tier quota handling
Free Gemini keys allow about 20 requests per day **per model**. The client in `agent/llm.py`:
* reads the server's suggested wait from a 429 response. A long wait means the daily quota is used up, so it doesn't
  keep retrying.
* moves down `GEMINI_FALLBACK_MODELS` when a model runs out of quota or is overloaded.
* raises a clear `QuotaExhausted` error saying when the quota resets once every model is used up. The evaluation
  stops early instead of recording a run of errors.

## Design choices

* **Agent loop written by hand, no framework.** Every component is visible and easy to explain. LangGraph or similar
  could replace `orchestrator.py` without touching the tools, skills or guardrails.
* **One OpenAI-compatible client for both providers.** Gemini and OpenRouter differ only in base URL, key and
  model. Rate limits (HTTP 429) are retried with exponential backoff, which matters on free tiers.
* **Deterministic guardrails rather than LLM checks.** They're fast, cost no tokens or quota, and are fully testable.
  The LLM-as-judge is used only in evaluation.
* **Security is enforced in code, not in the prompt.** Whose data can be read is decided in the tools, so even a
  manipulated model can't read another customer's records.

## Gmail (optional)

Set `EMAIL_BACKEND=gmail`, `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` (2-step verification plus an App Password). The
agent reads unread mail over IMAP and replies over SMTP. Start with `AUTO_SEND=false` so every reply waits in the
review queue until you approve it. The Streamlit UI works with the mock inbox; use `python main.py run` for Gmail.
