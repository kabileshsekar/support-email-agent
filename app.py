"""Streamlit demo UI for the Customer Support Email Agent.

    streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import config
from agent.email_client import Email, MockMailbox, ReviewQueue, read_jsonl
from agent.kb import get_kb
from agent.llm import LLMError
from agent.memory import ConversationMemory
from agent.observability import compute_metrics
from agent.orchestrator import SupportAgent
from agent.tracing import load_traces

st.set_page_config(page_title="Support Email Agent", page_icon="📧", layout="wide")

SERIES = "#2a78d6"
ACTION_BADGE = {"sent": "🟢 Sent", "escalated": "🟠 Escalated to human", "review": "🔵 Held for review",
                "ignored": "⚪ Ignored (spam)", "error": "🔴 Error"}
KIND_ICON = {"step": "⚙️", "llm": "🧠", "tool": "🛠️", "guardrail": "🛡️", "memory": "💾", "email": "✉️"}


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    providers = ["gemini", "openrouter", "mock"]
    default = config.get_settings().llm_provider
    provider = st.selectbox("LLM provider", providers, index=providers.index(default) if default in providers else 0,
                            help="mock = offline rule-based stand-in (no API key)")
    auto_send = st.checkbox("Auto-send replies", value=config.get_settings().auto_send,
                            help="Off = every reply is queued for human approval")
    dry_run = st.checkbox("Dry run (don't send or remember)", value=False)
    st.caption(f"Email backend: **{config.get_settings().email_backend}**")
    if st.button("Reset demo data", help="Re-seed DB, clear outbox, memory, traces and review queue"):
        from main import reset
        reset()
        st.cache_resource.clear()
        st.success("Demo data reset.")


@st.cache_resource(show_spinner=False)
def get_agent(provider: str, auto_send: bool, persist: bool, env_version: float) -> SupportAgent:
    # env_version is part of the cache key: editing .env builds a fresh agent with the new settings.
    return SupportAgent(config.get_settings(llm_provider=provider, auto_send=auto_send), persist=persist)


def agent_or_error():
    try:
        return get_agent(provider, auto_send, not dry_run, config.env_version())
    except LLMError as exc:
        st.error(str(exc))
        st.stop()


# ------------------------------------------------------------------ rendering helpers
def render_trace(trace: dict) -> None:
    spans = trace["spans"]
    depth, by_id = {}, {s["span_id"]: s for s in spans}
    for s in spans:
        d, p = 0, s["parent_id"]
        while p:
            d, p = d + 1, by_id[p]["parent_id"]
        depth[s["span_id"]] = d
    st.caption(f"Trace `{trace['trace_id']}` · {trace['duration_ms']:.0f} ms · status {trace['status']} · "
               f"{len(spans)} spans")
    rows = [{"span": " " * depth[s["span_id"]] + f"{KIND_ICON.get(s['kind'], '•')} {s['name']}",
             "kind": s["kind"], "ms": s["duration_ms"], "status": s["status"],
             "tokens": (s["attributes"].get("prompt_tokens") or 0) + (s["attributes"].get("completion_tokens") or 0)
             if s["kind"] == "llm" else None} for s in spans]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for s in spans:
        with st.expander(f"{KIND_ICON.get(s['kind'], '•')} {s['name']} — {s['duration_ms']:.0f} ms"):
            c1, c2 = st.columns(2)
            c1.markdown("**Input**")
            c1.write(s["input"] if s["input"] is not None else "—")
            c2.markdown("**Output**")
            c2.write(s["output"] if s["output"] is not None else "—")
            if s["attributes"]:
                st.json(s["attributes"])
            if s["error"]:
                st.error(s["error"])


def render_result(result, trace: dict | None) -> None:
    st.subheader(ACTION_BADGE.get(result.action, result.action))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intent", result.intent or "—")
    c2.metric("Skill", result.skill or "—")
    c3.metric("Tool calls", len(result.tools_used))
    c4.metric("Latency", f"{trace['duration_ms'] / 1000:.1f} s" if trace else "—")
    if not result.customer_verified:
        st.warning("Sender is not a registered customer, so account tools are locked.")
    if result.escalation:
        st.info(f"**Escalation:** {result.escalation['reason']} (priority {result.escalation['priority']})")
    if result.error:
        st.error(result.error)

    tabs = st.tabs(["✉️ Reply", "🗺️ Plan", "🛠️ Tool calls", "🛡️ Guardrails", "🔎 Trace"])
    with tabs[0]:
        if result.reply:
            st.text_area("Reply", result.reply, height=320, label_visibility="collapsed")
        else:
            st.write("No reply drafted.")
        if result.actions:
            st.write("Actions requested:", result.actions)
    with tabs[1]:
        st.json(result.plan or {})
    with tabs[2]:
        if not result.tool_calls:
            st.write("No tools called.")
        for i, tc in enumerate(result.tool_calls, 1):
            ok = "error" not in tc["result"]
            with st.expander(f"{i}. {'✅' if ok else '⚠️'} {tc['name']}({json.dumps(tc['arguments'])})"):
                st.json(tc["result"])
    with tabs[3]:
        st.markdown("**Input checks**")
        st.json(result.input_guardrails)
        st.markdown("**Output checks**" + (f" (revised {result.revisions}×)" if result.revisions else ""))
        st.json(result.output_guardrails or {"note": "not reached"})
    with tabs[4]:
        if trace:
            render_trace(trace)


def bar(data: dict, title: str, x_label: str = "count"):
    if not data:
        st.caption(f"{title}: no data yet")
        return
    df = pd.DataFrame(sorted(data.items(), key=lambda kv: kv[1]), columns=["name", x_label])
    fig = px.bar(df, x=x_label, y="name", orientation="h", title=title, text=x_label)
    fig.update_traces(marker_color=SERIES, textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x}<extra></extra>")
    fig.update_layout(height=60 + 34 * len(df), margin=dict(l=10, r=30, t=40, b=10), yaxis_title=None,
                      xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", title=None),
                      bargap=0.35, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ pages
st.title("📧 Customer Support Email Agent")
st.caption("Nimbus Electronics demo · planning → skills → tools → guardrails → send/escalate · "
           f"provider: **{provider}**")

page_inbox, page_out, page_traces, page_dash, page_eval, page_mem = st.tabs(
    ["📥 Inbox", "📤 Outbox & Review", "🔎 Traces", "📊 Dashboard", "🧪 Evaluation", "🧠 Memory & KB"])

with page_inbox:
    mailbox = MockMailbox()
    left, right = st.columns([1, 2])
    with left:
        unread = mailbox.fetch_unread()
        st.markdown(f"**Unread: {len(unread)}**")
        if st.button("▶️ Process all unread", disabled=not unread, use_container_width=True):
            agent = agent_or_error()
            bar_ = st.progress(0.0)
            for i, em in enumerate(unread, 1):
                r = agent.process_email(em)
                st.write(f"{em.id}: {ACTION_BADGE.get(r.action, r.action)} · {r.intent}")
                bar_.progress(i / len(unread))
            st.success("Done. See Outbox, Traces and Dashboard.")
        all_emails = mailbox.all_emails()
        labels = {f"{'🆕' if e in unread else '✔️'} {e.id} · {e.from_name or e.from_email} · {e.subject}": e
                  for e in all_emails}
        choice = st.radio("Emails", list(labels), label_visibility="collapsed")
        with st.expander("✍️ Compose a test email"):
            with st.form("compose"):
                frm = st.text_input("From", "priya.sharma@example.com")
                subj = st.text_input("Subject", "My speaker keeps disconnecting")
                body = st.text_area("Body", "Hi, my Echo Speaker 200 drops off Wi-Fi every evening. Help?")
                submitted = st.form_submit_button("Run agent on this email")
            if submitted:
                st.session_state["custom_email"] = Email(id=f"CUSTOM-{abs(hash((frm, subj, body))) % 10**6}",
                                                         from_email=frm, from_name="", subject=subj, body=body)
    with right:
        em = st.session_state.pop("custom_email", None) or labels[choice]
        st.markdown(f"**From:** {em.from_name} &lt;{em.from_email}&gt;  \n**Subject:** {em.subject}  \n"
                    f"**Thread:** {em.thread_id}")
        st.text_area("Body", em.body, height=140, disabled=True, label_visibility="collapsed")
        run_now = st.button("🤖 Run agent on this email", type="primary") or em.id.startswith("CUSTOM-")
        if run_now:
            agent = agent_or_error()
            with st.spinner("Agent working: planning, calling tools, checking guardrails..."):
                result = agent.process_email(em)
            st.session_state["last"] = (result, agent.last_trace)
        if "last" in st.session_state and st.session_state["last"][0].email_id == em.id:
            render_result(*st.session_state["last"])

with page_out:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sent replies")
        sent = read_jsonl(config.OUTBOX_PATH)
        if not sent:
            st.caption("Nothing sent yet.")
        for m in sent:
            with st.expander(f"→ {m['to']} · {m['subject']} · {m['sent_at']}"):
                st.text(m["body"])
    with c2:
        st.subheader("Human review queue")
        queue = ReviewQueue().all()
        if not queue:
            st.caption("Queue is empty.")
        for t in queue:
            with st.expander(f"{t['ticket_id']} · [{t['priority']}] {t['from']} · {t['subject']}"):
                st.write(f"**Reason:** {t['reason']}")
                st.write(f"**Trace:** `{t['trace_id']}`")
                if t["draft_reply"]:
                    st.text_area("Draft reply", t["draft_reply"], height=200, key=t["ticket_id"])

with page_traces:
    traces = load_traces(limit=200)
    if not traces:
        st.caption("No traces yet. Process an email first.")
    else:
        opts = {f"{t['start_time'][11:19]} · {t['metadata'].get('email_id')} · {t['metadata'].get('subject', '')[:40]} · "
                f"{(t.get('output') or {}).get('action', t['status'])}": t for t in traces}
        render_trace(opts[st.selectbox("Trace", list(opts))])

with page_dash:
    m = compute_metrics(load_traces())
    if not m["emails_processed"]:
        st.caption("No data yet. Process some emails first.")
    else:
        k = st.columns(6)
        k[0].metric("Emails processed", m["emails_processed"])
        k[1].metric("Auto-resolved", f"{m['auto_resolved_rate']:.0%}")
        k[2].metric("Escalation rate", f"{m['escalation_rate']:.0%}")
        k[3].metric("p50 / p95 latency", f"{m['latency_ms_p50'] / 1000:.1f}s / {m['latency_ms_p95'] / 1000:.1f}s")
        k[4].metric("Tokens / email", f"{m['tokens_per_email']:.0f}")
        k[5].metric("Tool error rate", f"{m['tool_error_rate']:.0%}")
        k2 = st.columns(4)
        k2[0].metric("LLM calls", m["llm_calls"])
        k2[1].metric("Tool calls", m["tool_calls"])
        k2[2].metric("Input blocks (spam/injection)", m["input_guardrail_blocks"])
        k2[3].metric("Output guardrail failures", m["output_guardrail_failures"])
        c1, c2, c3 = st.columns(3)
        with c1:
            bar(m["intents"], "Emails by intent")
        with c2:
            bar(m["actions"], "Emails by outcome")
        with c3:
            bar(m["tool_usage"], "Tool usage")
        with st.expander("Raw metrics"):
            st.json(m)

with page_eval:
    from eval.run_eval import REPORT_DIR, run
    st.write("Runs the labelled test set in `eval/dataset.json` (dry run: nothing is sent or remembered).")
    c1, c2, c3 = st.columns(3)
    use_judge = c1.checkbox("LLM-as-judge scoring")
    delay = c2.number_input("Delay between cases (s)", 0.0, 30.0, 0.0 if provider == "mock" else 4.0,
                            help="Free tiers have requests-per-minute limits")
    if c3.button("Run evaluation", type="primary"):
        prog = st.progress(0.0)
        try:
            run(provider, use_judge, delay, progress=lambda i, n, r: prog.progress(i / n, f"{r['id']}"))
        except LLMError as exc:
            st.error(str(exc))
    latest = REPORT_DIR / "latest.json"
    if latest.exists():
        rep = json.loads(latest.read_text(encoding="utf-8"))
        st.caption(f"Latest report: {rep['created_at']} · {rep['provider']} ({rep['model']})")
        s = rep["summary"]
        cols = st.columns(6)
        for col, key in zip(cols, ["intent_accuracy", "action_accuracy", "tool_recall", "fact_recall",
                                   "safety_pass_rate", "guardrail_pass_rate"]):
            col.metric(key.replace("_", " ").title(), f"{s[key]:.0%}")
        if "judge_helpfulness" in s:
            j = st.columns(3)
            for col, key in zip(j, ["judge_helpfulness", "judge_accuracy", "judge_tone"]):
                col.metric(key.replace("_", " ").title(), f"{s[key]:.2f} / 5")
        df = pd.DataFrame(rep["cases"])
        show = ["id", "intent", "action", "intent_ok", "action_ok", "tool_recall", "fact_recall", "safety_ok",
                "guardrail_ok", "latency_ms", "tokens"]
        st.dataframe(df[show], use_container_width=True, hide_index=True)
        pick = st.selectbox("Inspect a case", df["id"])
        row = df[df["id"] == pick].iloc[0]
        st.text_area("Reply", row["reply"], height=220)
        if row.get("leaks"):
            st.error(f"Forbidden content found: {row['leaks']}")

with page_mem:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Conversation memory")
        rows = ConversationMemory().all()
        if rows:
            st.dataframe(pd.DataFrame(rows)[["ts", "customer_email", "thread_id", "subject", "action", "summary"]],
                         use_container_width=True, hide_index=True)
        else:
            st.caption("Empty.")
    with c2:
        st.subheader("Knowledge base search")
        q = st.text_input("Query", "watch battery drains")
        for r in get_kb().search(q, top_k=4):
            with st.expander(f"{r['score']:.2f} · {r['title']} › {r['section']}"):
                st.write(r["content"])
