"""Tools the agent can call, with JSON schemas for LLM function calling.

Security model: the LLM never decides *whose* data it may read. Every tool receives
a ToolContext holding the customer verified from the sender's email address, and
account-level tools only return records that belong to that customer. A request for
someone else's order gets the same "not found" answer as a non-existent order, so
the agent cannot be tricked into leaking data or confirming that it exists.

Sending email is deliberately NOT a tool exposed to the LLM: the orchestrator sends
the reply only after the output guardrails pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional

from . import tracing
from .db import SupportDB
from .kb import get_kb
from .observability import get_logger, log_event

log = get_logger("tools")


@dataclass
class ToolContext:
    sender_email: str
    customer: Optional[dict]            # verified customer record, or None if sender unknown
    db: SupportDB
    dry_run: bool = False
    escalation: Optional[dict] = None   # set by escalate_to_human
    actions: list[dict] = field(default_factory=list)  # side-effect requests (e.g. cancellations)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    requires_verified_customer: bool = False

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


NOT_VERIFIED = {
    "error": "sender_not_verified",
    "message": ("The sender's email address is not registered to any customer account. Do not reveal or "
                "confirm any order, device or account details. Ask the customer to write from the email "
                "address registered on their Nimbus account."),
}
NOT_FOUND = {"error": "not_found",
             "message": "No matching record was found on this customer's account. Ask them to double-check it."}

TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict, required: list[str] | None = None,
         verified: bool = False):
    def wrap(fn):
        TOOLS[name] = Tool(name, description,
                           {"type": "object", "properties": properties, "required": required or []},
                           fn, verified)
        return fn
    return wrap


def _owned(record: Optional[dict], ctx: ToolContext) -> bool:
    return bool(record) and record.get("customer_id") == ctx.customer["customer_id"]


# ---------------------------------------------------------------- knowledge base
@tool("search_knowledge_base",
      "Search the Nimbus knowledge base (FAQs, user manuals, troubleshooting guides, policies). "
      "Use specific queries, e.g. 'Pulse Watch 3 battery drain' or 'refund timing after return'.",
      {"query": {"type": "string", "description": "What to look for"},
       "top_k": {"type": "integer", "description": "Number of sections to return (1-5)", "default": 3}},
      ["query"])
def search_knowledge_base(ctx: ToolContext, query: str, top_k: int = 3):
    results = get_kb().search(query, top_k=max(1, min(int(top_k or 3), 5)))
    return {"results": results} if results else {"results": [], "message": "No relevant articles found."}


# ---------------------------------------------------------------- customer & orders
@tool("get_customer_profile", "Get the verified sender's own customer profile (name, tier, member since).",
      {}, verified=True)
def get_customer_profile(ctx: ToolContext):
    c = ctx.customer
    return {"customer_id": c["customer_id"], "name": c["name"], "email": c["email"],
            "tier": c["tier"], "member_since": c["member_since"]}


@tool("list_customer_orders", "List all orders on the verified sender's account, newest first.", {},
      verified=True)
def list_customer_orders(ctx: ToolContext):
    orders = ctx.db.orders_for_customer(ctx.customer["customer_id"])
    return {"orders": [{"order_id": o["order_id"], "placed_date": o["placed_date"], "status": o["status"],
                        "items": [f"{i['quantity']} x {i['product']}" for i in o["items"]]} for o in orders]}


@tool("get_order_status",
      "Get status, carrier, tracking number, estimated delivery date and items for one order on the sender's account.",
      {"order_id": {"type": "string", "description": "Order ID, e.g. ORD-10421"}}, ["order_id"], verified=True)
def get_order_status(ctx: ToolContext, order_id: str):
    o = ctx.db.order(order_id)
    if not _owned(o, ctx):
        return NOT_FOUND
    return {"order_id": o["order_id"], "status": o["status"], "placed_date": o["placed_date"],
            "shipping_method": o["shipping_method"], "carrier": o["carrier"],
            "tracking_number": o["tracking_number"], "estimated_delivery": o["estimated_delivery"],
            "note": o["note"], "total": o["total"],
            "items": [{"product": i["product"], "quantity": i["quantity"]} for i in o["items"]]}


# ---------------------------------------------------------------- devices & warranty
@tool("list_customer_devices", "List the devices registered to the verified sender, with serial numbers.", {},
      verified=True)
def list_customer_devices(ctx: ToolContext):
    devs = ctx.db.devices_for_customer(ctx.customer["customer_id"])
    return {"devices": [{"serial_number": d["serial_number"], "product": d["product"],
                         "purchase_date": d["purchase_date"]} for d in devs]}


@tool("get_device_info",
      "Get details of one of the sender's devices: product, purchase date, installed vs latest firmware.",
      {"serial_number": {"type": "string", "description": "Serial number, e.g. SN-WCH-30455"}},
      ["serial_number"], verified=True)
def get_device_info(ctx: ToolContext, serial_number: str):
    d = ctx.db.device(serial_number)
    if not _owned(d, ctx):
        return NOT_FOUND
    return {"serial_number": d["serial_number"], "product": d["product"], "category": d["category"],
            "purchase_date": d["purchase_date"], "firmware_installed": d["firmware_version"],
            "firmware_latest": d["latest_firmware"],
            "firmware_up_to_date": d["firmware_version"] == d["latest_firmware"]}


@tool("check_warranty",
      "Check warranty status for one of the sender's devices: plan, end date, active or expired, coverage.",
      {"serial_number": {"type": "string", "description": "Serial number, e.g. SN-SPK-40877"}},
      ["serial_number"], verified=True)
def check_warranty(ctx: ToolContext, serial_number: str):
    d = ctx.db.device(serial_number)
    if not _owned(d, ctx):
        return NOT_FOUND
    w = ctx.db.warranty(serial_number)
    end = date.fromisoformat(w["end_date"])
    days_left = (end - date.today()).days
    return {"serial_number": d["serial_number"], "product": d["product"],
            "plan": "NimbusCare+ (extended)" if w["plan"] == "nimbuscare_plus" else "Standard warranty",
            "start_date": w["start_date"], "end_date": w["end_date"],
            "status": "active" if days_left >= 0 else "expired",
            "days_remaining": max(days_left, 0), "claims_used": w["claims_used"], "coverage": w["coverage"]}


# ---------------------------------------------------------------- products (public)
@tool("get_product_info", "Get price, warranty length, latest firmware and full specifications for a product. "
      "Public information - works for anyone.",
      {"product": {"type": "string", "description": "Product name or SKU, e.g. 'Guard Cam 2K'"}}, ["product"])
def get_product_info(ctx: ToolContext, product: str):
    p = ctx.db.find_product(product)
    if not p:
        return {"error": "not_found", "message": f"No product matches '{product}'.",
                "available_products": [x["name"] for x in ctx.db.products()]}
    return {"sku": p["sku"], "name": p["name"], "category": p["category"], "price": p["price"],
            "warranty_months": p["warranty_months"], "latest_firmware": p["latest_firmware"], "specs": p["specs"]}


# ---------------------------------------------------------------- actions
@tool("request_order_cancellation",
      "Submit a cancellation request for an order on the sender's account. Only possible while the order "
      "status is 'processing'.",
      {"order_id": {"type": "string"}, "reason": {"type": "string"}}, ["order_id"], verified=True)
def request_order_cancellation(ctx: ToolContext, order_id: str, reason: str = ""):
    o = ctx.db.order(order_id)
    if not _owned(o, ctx):
        return NOT_FOUND
    if o["status"] != "processing":
        return {"error": "not_cancellable", "status": o["status"],
                "message": "Only orders in 'processing' status can be cancelled. Shipped orders must be returned."}
    action = {"type": "order_cancellation", "order_id": o["order_id"], "reason": reason,
              "customer_id": ctx.customer["customer_id"]}
    ctx.actions.append(action)
    return {"result": "cancellation_request_submitted", "order_id": o["order_id"],
            "message": "Request logged. The warehouse confirms cancellations within 24 hours and the refund "
                       "goes to the original payment method in 5-7 business days."}


@tool("escalate_to_human",
      "Hand the case to a human specialist. Use for refund exceptions, compensation, legal threats, "
      "warranty claims needing approval, abusive messages, or anything you cannot resolve with the tools.",
      {"reason": {"type": "string"},
       "priority": {"type": "string", "enum": ["low", "normal", "high"], "default": "normal"}},
      ["reason"])
def escalate_to_human(ctx: ToolContext, reason: str, priority: str = "normal"):
    ctx.escalation = {"reason": reason, "priority": priority}
    return {"result": "escalated",
            "message": "A specialist will reply within 2 business days. Tell the customer this and do not "
                       "promise any specific outcome."}


# ---------------------------------------------------------------- execution
def schemas(names: list[str]) -> list[dict]:
    return [TOOLS[n].schema() for n in names if n in TOOLS]


def execute(name: str, arguments: dict, ctx: ToolContext, allowed: Optional[list[str]] = None) -> dict:
    with tracing.span(f"tool:{name}", kind="tool", input=arguments) as sp:
        if name not in TOOLS or (allowed is not None and name not in allowed):
            result = {"error": "unknown_tool", "message": f"Tool '{name}' is not available."}
        elif TOOLS[name].requires_verified_customer and ctx.customer is None:
            result = NOT_VERIFIED
        else:
            try:
                args = {k: v for k, v in (arguments or {}).items() if not k.startswith("_")}
                result = TOOLS[name].fn(ctx, **args)
            except TypeError as exc:
                result = {"error": "bad_arguments", "message": str(exc)}
            except Exception as exc:  # a broken tool must not crash the agent
                log_event(log, "tool failed", tool=name, error=str(exc))
                result = {"error": "tool_failure", "message": f"{type(exc).__name__}: {exc}"}
        sp.set_output(result)
        sp.set(ok="error" not in result)
        return result


def to_message(tool_call_id: str, name: str, result: dict) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name,
            "content": json.dumps(result, ensure_ascii=False, default=str)}
