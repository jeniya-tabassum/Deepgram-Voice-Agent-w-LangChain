"""
The LangChain "brain" for the Deepgram Voice Agent.

THIS FILE IS THE WHOLE POINT OF THE DEMO: it shows how to plug LangChain into a
Deepgram Voice Agent. Deepgram handles speech-to-text, text-to-speech, and
conversation flow. Whenever the customer asks a real question, Deepgram calls a
single function `ask_support_brain`, and that function is handled here.

To showcase **multi-step agent reasoning**, the brain is a LangChain
**tool-calling agent** (`create_agent`, a LangGraph agent⇄tools loop). The agent
breaks a complex query into sequential steps and decides *which tools to call, in
what order* — chaining them until it can answer. It has five domain tools:

  1. search_dealership_info   -> RAG over the Ascent Automotive Group knowledge base
  2. check_inventory          -> look up vehicle availability + starting price
  3. calculate_monthly_payment-> a financing CALCULATOR (loan amortization)
  4. estimate_trade_in        -> value a customer's current car
  5. book_appointment         -> book a test drive / service / financing consult

Example of the multi-step reasoning this enables — "I have a 2018 Civic with 60k
miles to trade in; what's the monthly payment on a new RAV4?":

    ① estimate_trade_in(Civic, 2018, 60000)  -> ~$5,200
    ② check_inventory(Toyota, RAV4, new)      -> from $27,990
    ③ calculate_monthly_payment(27990, down=5200) -> ~$450/mo
    -> one short spoken answer

The agent decides that sequence itself. Swap ChatOpenAI for any LangChain chat
model, the TFIDFRetriever for a real vector store, and the mock backends for real
inventory / DMS / lender APIs — the Deepgram integration never changes.
"""

import json
import os
import uuid
from datetime import date
from typing import Optional

from langchain.agents import create_agent  # LangChain v1, LangGraph agent⇄tools loop
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from .knowledge_base import FAQ_METADATAS, FAQ_TEXTS

# --------------------------------------------------------------------------- #
# 1. Knowledge base retriever (RAG)
# --------------------------------------------------------------------------- #
# TFIDFRetriever keeps the demo dependency-light: no embeddings API, no vector
# DB. For production, swap this for an embeddings-backed vector store retriever.
_retriever = TFIDFRetriever.from_texts(FAQ_TEXTS, metadatas=FAQ_METADATAS)
_retriever.k = 3

_CURRENT_YEAR = date.today().year

# Rough starting prices used by the mock inventory engine.
_NEW_FROM = 27990
_CPO_FROM = 21490
_USED_FROM = 15990


# --------------------------------------------------------------------------- #
# 2. Domain tools — the agent chains these to answer complex queries
# --------------------------------------------------------------------------- #
@tool
def search_dealership_info(query: str) -> str:
    """Search the Ascent Automotive Group knowledge base for general information about
    financing, warranties, trade-in process, returns policy, hours, and delivery.
    Use this for any 'how does X work' or policy question."""
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant dealership information was found."
    return "\n\n---\n\n".join(d.page_content for d in docs)


@tool
def check_inventory(
    make: str,
    model: str,
    max_budget: Optional[int] = None,
    condition: Optional[str] = None,
) -> str:
    """Check whether a specific vehicle is available and get its starting price.
    Provide make and model, and optionally a max budget and condition
    (new, used, or cpo). Returns a JSON object including `price_from`, which you
    can feed into calculate_monthly_payment. Call this before quoting any price."""
    cond = (condition or "").lower()
    if "new" in cond:
        starting, label = _NEW_FROM, "new"
    elif "cpo" in cond or "certified" in cond:
        starting, label = _CPO_FROM, "certified pre-owned"
    elif "used" in cond or "pre-owned" in cond:
        starting, label = _USED_FROM, "used"
    else:
        starting, label = _CPO_FROM, "in-stock"

    seed = len((make or "") + (model or ""))
    count = 2 + (seed % 6)  # deterministic 2..7 units (repeatable demo)

    try:
        budget = int(max_budget) if max_budget is not None else None
    except (TypeError, ValueError):
        budget = None
    within_budget = budget is None or budget >= starting

    return json.dumps({
        "make": make, "model": model, "condition": label,
        "matches": count, "price_from": starting,
        "max_budget": budget, "within_budget": within_budget,
        "available": count > 0,
    })


@tool
def calculate_monthly_payment(
    price: float,
    down_payment: float = 0.0,
    apr_percent: float = 6.9,
    term_months: int = 60,
) -> str:
    """Financing calculator. Compute the monthly loan payment for a vehicle using
    standard amortization. Provide the price (use price_from from check_inventory),
    the down payment (a trade-in value can be the down payment), the APR percent,
    and the term in months. ALWAYS use this for any payment math — never compute
    it yourself. Returns a JSON object with `monthly_payment`."""
    try:
        principal = max(0.0, float(price) - float(down_payment))
        n = int(term_months)
        r = float(apr_percent) / 100.0 / 12.0
    except (TypeError, ValueError):
        return json.dumps({"error": "invalid numeric input"})
    if n <= 0:
        return json.dumps({"error": "term_months must be positive"})
    monthly = principal / n if r == 0 else principal * r / (1 - (1 + r) ** (-n))
    return json.dumps({
        "price": round(float(price), 2),
        "down_payment": round(float(down_payment), 2),
        "amount_financed": round(principal, 2),
        "apr_percent": apr_percent,
        "term_months": n,
        "monthly_payment": round(monthly, 2),
        "total_of_payments": round(monthly * n, 2),
    })


@tool
def estimate_trade_in(
    make: str,
    model: str,
    year: int,
    mileage: int,
    condition: str = "good",
) -> str:
    """Estimate the trade-in value of a customer's current vehicle from its make,
    model, year, mileage, and condition (excellent, good, fair, or poor). Returns
    a JSON object with `estimated_value`, which can be used as a down payment."""
    try:
        age = max(0, _CURRENT_YEAR - int(year))
        miles = max(0, int(mileage))
    except (TypeError, ValueError):
        age, miles = 8, 100000
    factor = {"excellent": 1.1, "good": 1.0, "fair": 0.85, "poor": 0.7}.get(
        (condition or "good").lower(), 1.0
    )
    value = (22000 - age * 1500 - (miles / 1000.0) * 80) * factor
    value = max(500, round(value / 100.0) * 100)  # floor + round to nearest $100
    return json.dumps({
        "make": make, "model": model, "year": year, "mileage": miles,
        "condition": condition, "estimated_value": value,
    })


def _looks_negative(reply) -> bool:
    """True if the caller's confirmation reply reads as 'no / not yet / change it'.

    Deliberately conservative: we only BLOCK the booking on a clear negative, so an
    ambiguous 'yeah ok sure' proceeds. 'yes' anywhere wins immediately.
    """
    text = str(reply or "").strip().lower()
    if not text:
        return False  # empty resume value -> treat as go-ahead
    if text.startswith(("yes", "yep", "yeah", "sure", "correct", "go ahead", "confirm")):
        return False
    negatives = ("no", "nope", "don't", "do not", "cancel", "stop", "wait",
                 "not yet", "hold on", "change", "different", "instead")
    return any(word in text for word in negatives)


@tool
def book_appointment(
    appointment_type: str,
    make: str = "",
    model: str = "",
    contact_phone: str = "",
    preferred_time: str = "",
) -> str:
    """Book an appointment once the customer is ready. appointment_type is one of
    test_drive, service, or financing. Provide the vehicle make/model, a callback
    phone number, and a preferred time. Returns a JSON confirmation with a
    reference number (APT-XXXXXX)."""
    type_label = {
        "test_drive": "test drive",
        "service": "service appointment",
        "financing": "financing consultation",
    }.get((appointment_type or "").lower(), "appointment")
    when = preferred_time or "the next available slot"
    vehicle = f"{make} {model}".strip() or "your vehicle"

    # --- Human-in-the-loop checkpoint --------------------------------------- #
    # Pause the graph and CONFIRM with the caller BEFORE committing the booking.
    # interrupt() snapshots graph state to the checkpointer and unwinds; when the
    # caller answers, the graph RESUMES *at this node* (prior tool results are
    # loaded from the checkpoint — the graph does NOT restart) with `decision` set
    # to their spoken reply. This is the "resume from the same node" behavior.
    decision = interrupt({
        "action": "confirm_booking",
        "ui": "booking",  # which graph node the UI should render as paused
        "appointment_type": type_label,
        "vehicle": vehicle,
        "preferred_time": when,
        # `prompt` is what the voice agent should say to the caller.
        "prompt": (
            f"I'm about to book a {type_label} for the {vehicle} at {when}"
            + (f", and I'll use {contact_phone} as your callback number" if contact_phone else "")
            + ". Should I go ahead and book it?"
        ),
    })

    # Reached only after the caller responds and the graph resumes here.
    if _looks_negative(decision):
        return json.dumps({
            "appointment_type": type_label, "vehicle": vehicle,
            "status": "Not booked",
            "message": (
                f"No problem — I won't book the {type_label} yet. "
                "Just tell me what you'd like to change."
            ),
        })

    ref = "APT-" + uuid.uuid4().hex[:6].upper()
    return json.dumps({
        "reference": ref, "appointment_type": type_label, "vehicle": vehicle,
        "contact_phone": contact_phone or "your number on file",
        "preferred_time": when, "status": "Booked",
        "message": f"Your {type_label} for the {vehicle} is booked ({ref}) for {when}.",
    })


TOOLS = [
    search_dealership_info,
    check_inventory,
    calculate_monthly_payment,
    estimate_trade_in,
    book_appointment,
]

# Maps tool name -> the UI node id used by the live agent panel (frontend/app.js).
TOOL_UI_IDS = {
    "search_dealership_info": "search",
    "check_inventory": "inventory",
    "calculate_monthly_payment": "calculator",
    "estimate_trade_in": "tradein",
    "book_appointment": "booking",
}


# --------------------------------------------------------------------------- #
# 3. The agent
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are the customer-care brain for Ascent Automotive Group, a car dealership. You answer questions "
    "coming from a live phone/voice conversation, so your FINAL answer must be SHORT and "
    "conversational — one or two sentences, no markdown, no lists.\n\n"
    "You have tools and you REASON IN MULTIPLE STEPS: break a complex request into a sequence of "
    "tool calls and chain them. Rules:\n"
    "- Use search_dealership_info for general policy questions (financing, warranty, trade-in, "
    "returns, hours, delivery).\n"
    "- Use check_inventory to confirm a vehicle is available and to get its price BEFORE quoting "
    "any price.\n"
    "- For ANY monthly-payment or financing math, ALWAYS call calculate_monthly_payment — never do "
    "arithmetic yourself. Get the price from check_inventory first.\n"
    "- Use estimate_trade_in when the customer mentions trading in a car; its estimated_value can "
    "be used as the down_payment for calculate_monthly_payment.\n"
    "- Use book_appointment only when the customer wants to book.\n"
    "- Always refer to a vehicle by the EXACT make and model the customer named, and pass those exact "
    "values to the tools. Never substitute, guess, or drift to a different model (e.g. do not answer "
    "about a Highlander when the customer asked about a RAV4).\n"
    "Never invent prices, inventory, payments, or policies — ground every number in a tool result. "
    "If a needed detail is missing, ask one brief question."
)


def build_support_agent():
    """Construct the LangChain multi-step tool-calling agent (a compiled LangGraph
    agent⇄tools loop). Called once at startup; returns a compiled graph exposing
    the `.astream(...)` / `.ainvoke(...)` interface the bridge uses.

    A checkpointer is required for human-in-the-loop: it persists graph state at
    each step so a call that hits interrupt() (see book_appointment) can be
    RESUMED from that node on a later turn instead of restarting. State is keyed by
    the `thread_id` the bridge passes per call (see run_support_brain). InMemorySaver
    keeps the demo dependency-light; swap for a PostgresSaver/RedisSaver to persist
    across process restarts and scale horizontally."""
    model = os.environ.get("LANGCHAIN_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0, max_tokens=500)
    return create_agent(
        llm, tools=TOOLS, system_prompt=_SYSTEM_PROMPT, checkpointer=InMemorySaver()
    )


# --------------------------------------------------------------------------- #
# 4. Helpers + public entry point
# --------------------------------------------------------------------------- #
def _extract_text(content) -> str:
    """The message content may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(p for p in parts if p)
    return str(content or "")


def _short(value, limit: int = 90) -> str:
    s = value if isinstance(value, str) else json.dumps(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"


async def run_support_brain(
    agent, question: str, thread_id: str, resume: bool = False, on_event=None
) -> dict:
    """Run the multi-step agent for one customer turn.

    Returns {"content": <text the voice agent should speak>, "interrupted": bool}.
    When `interrupted` is True, `content` is a confirmation question and the graph
    is PAUSED at a node (see book_appointment); the bridge should send the caller's
    next reply back with resume=True to continue from that node.

    `thread_id` scopes the graph's persisted state so each phone call has its own
    conversation/checkpoint. `resume=False` starts a fresh turn from the user's
    message; `resume=True` feeds `question` (the caller's reply) into the paused
    interrupt() via Command(resume=...), resuming the graph in place.

    If `on_event` is provided, it is awaited for each reasoning step so a UI can
    visualize the agent's tool-calling loop live. We stream with
    `stream_mode="updates"` and classify the new messages each node emits:
      - AIMessage with tool_calls  -> the agent decided to call tool(s)
      - ToolMessage                -> a tool returned a result
      - AIMessage without tool_calls -> the final spoken answer
      - a "__interrupt__" chunk    -> the graph paused for human confirmation
    Event shapes: {"kind": "tool_call", "step", "name", "ui", "args"},
                  {"kind": "tool_result", "name", "ui", "result"},
                  {"kind": "interrupt", "prompt"}.
    """
    async def emit(payload: dict) -> None:
        if on_event is not None:
            try:
                await on_event(payload)
            except Exception:  # noqa: BLE001 - UI streaming must never break the answer
                pass

    config = {"configurable": {"thread_id": thread_id}}
    graph_input = Command(resume=question) if resume else {
        "messages": [{"role": "user", "content": question}]
    }

    try:
        answer, step, trace = "", 0, []
        interrupt_prompt = None
        async for chunk in agent.astream(graph_input, config=config, stream_mode="updates"):
            # A pause surfaces as a top-level "__interrupt__" key whose value is a
            # tuple of Interrupt objects; .value is the dict book_appointment passed.
            if "__interrupt__" in chunk:
                intr = chunk["__interrupt__"][0]
                val = getattr(intr, "value", None)
                if isinstance(val, dict):
                    interrupt_prompt = val.get("prompt")
                    interrupt_ui = val.get("ui")
                else:
                    interrupt_prompt = str(val) if val is not None else None
                    interrupt_ui = None
                trace.append("interrupt")
                await emit({"kind": "interrupt", "prompt": interrupt_prompt, "ui": interrupt_ui})
                continue
            for _node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                for m in update.get("messages", []) or []:
                    if isinstance(m, AIMessage):
                        tool_calls = getattr(m, "tool_calls", None) or []
                        if tool_calls:
                            for tc in tool_calls:
                                step += 1
                                name = tc.get("name", "?")
                                trace.append(name)
                                await emit({
                                    "kind": "tool_call", "step": step, "name": name,
                                    "ui": TOOL_UI_IDS.get(name, name),
                                    "args": _short(tc.get("args", {})),
                                })
                        else:
                            txt = _extract_text(m.content).strip()
                            if txt:
                                answer = txt
                    elif isinstance(m, ToolMessage):
                        await emit({
                            "kind": "tool_result",
                            "name": getattr(m, "name", None),
                            "ui": TOOL_UI_IDS.get(getattr(m, "name", ""), None),
                            "result": _short(_extract_text(m.content)),
                        })
        if trace:
            tail = "paused" if interrupt_prompt is not None else "answer"
            print(f"[brain] steps: {' -> '.join(trace)} -> {tail}")
        if interrupt_prompt is not None:
            return {"content": interrupt_prompt, "interrupted": True}
        return {
            "content": answer or "I'm sorry, I wasn't able to find an answer to that.",
            "interrupted": False,
        }
    except Exception as exc:  # noqa: BLE001 - surface a speakable fallback
        print(f"[langchain_brain] error: {exc!r}")
        return {
            "content": (
                "I'm having trouble pulling that up right now. "
                "I can connect you with a sales specialist who can follow up with you."
            ),
            "interrupted": False,
        }
