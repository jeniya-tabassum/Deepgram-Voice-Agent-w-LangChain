"""
Deepgram Voice Agent connection + configuration.

We talk to the Voice Agent v1 API over a WebSocket:
    wss://agent.deepgram.com/v1/agent/converse

The key idea for the LangChain integration is the `functions` array in the
Settings message below. We declare ONE function, `ask_support_brain`, with NO
`endpoint` field — that makes it a *client-side* function. When the agent decides
to call it, Deepgram sends us a `FunctionCallRequest` over this same socket, our
backend runs the LangChain agent, and we reply with a `FunctionCallResponse`.
"""

import os
import random

import websockets

DEEPGRAM_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"

# Audio formats exchanged with Deepgram (must match what the browser sends/plays).
INPUT_SAMPLE_RATE = 48000   # mic PCM the browser streams up
OUTPUT_SAMPLE_RATE = 24000  # TTS PCM Deepgram streams back

# --- End-of-utterance (turn-taking) tuning for Flux -------------------------- #
# These live inside agent.listen.provider and are Flux/v2-only. They control WHEN
# Deepgram decides the caller has finished their turn, which is exactly the
# "don't cut slower/natural speakers off" requirement.
#
#   eot_threshold   (0.5–0.9, default 0.7): confidence required to END the turn.
#                   HIGHER = wait for stronger evidence the caller is really done,
#                   so mid-sentence pauses don't trigger a premature end-of-turn.
#   eot_timeout_ms  (default 5000): hard cap — the turn ends this long after
#                   speech even if confidence stays low. RAISE it to tolerate long
#                   pauses (e.g. a caller reading a VIN or thinking out loud).
#   eager_eot_threshold (optional, must be <= eot_threshold): starts the response
#                   BEFORE the caller fully finishes. It lowers latency but risks
#                   interrupting, so it's OFF by default here — set DG_EAGER_EOT_
#                   THRESHOLD to opt in. Docs: developers.deepgram.com/docs/flux/configuration
# All three can also be changed mid-call via an UpdateListen message.
EOT_THRESHOLD = float(os.environ.get("DG_EOT_THRESHOLD", "0.8"))
EOT_TIMEOUT_MS = int(os.environ.get("DG_EOT_TIMEOUT_MS", "8000"))
_EAGER_EOT = os.environ.get("DG_EAGER_EOT_THRESHOLD")  # opt-in; omit to favor not interrupting


async def connect_deepgram():
    """Open the Voice Agent WebSocket, authenticating with the Deepgram API key.

    Handles the `additional_headers` (websockets >= 14) vs `extra_headers`
    (older websockets) naming difference so this works across versions.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set (see .env.example).")

    headers = {"Authorization": f"Token {api_key}"}
    try:
        return await websockets.connect(
            DEEPGRAM_AGENT_URL, additional_headers=headers, max_size=None
        )
    except TypeError:
        return await websockets.connect(
            DEEPGRAM_AGENT_URL, extra_headers=headers, max_size=None
        )


# The Ascent Automotive Group voice persona / call flow. Note the appended "#Tools" section:
# it tells the think-LLM to route every dealership question through the
# client-side `ask_support_brain` function, which is handled by LangChain.
_SYSTEM_PROMPT = (
    "#Role\n"
    "You are a virtual customer-care assistant for Ascent Automotive Group, a car dealership, speaking to "
    "customers over the phone. Your task is to help them with financing, warranties, trade-ins, "
    "vehicle availability, and booking test drives or service.\n\n"
    "#General Guidelines\n"
    "Be warm, helpful, and professional.\n"
    "Speak clearly and naturally in plain language.\n"
    "Always refer to a vehicle by the exact make and model the caller stated; never substitute or "
    "guess a different model when relaying an answer.\n"
    "Keep most responses to 1–2 sentences and under 120 characters unless the caller asks for "
    "more detail (max: 300 characters).\n"
    "Do not use markdown formatting, including code blocks, quotes, bold, links, or italics.\n"
    "Use line breaks for lists.\n"
    "Avoid repeating phrasing.\n"
    "If a message is unclear, ask for clarification.\n"
    "If the user’s message is empty, respond with an empty message.\n"
    "If asked how you're doing, respond kindly and briefly.\n\n"
    "#Voice-Specific Instructions\n"
    "Speak in a conversational tone—your responses will be spoken aloud.\n"
    "Pause briefly after questions to allow replies.\n"
    "Confirm unclear inputs with the customer.\n"
    "Do not interrupt.\n\n"
    "#Style\n"
    "Use a friendly, approachable, professional tone.\n"
    "Keep language simple and reassuring.\n"
    "Mirror the customer’s tone if they use formal or technical language.\n\n"
    "#Call Flow Objective\n"
    "Greet the caller and welcome them to Ascent Automotive Group. Ask how you can help.\n"
    "If they ask about a general topic (financing, warranty, trade-in, returns, hours, delivery), "
    "route it to ask_support_brain and relay the answer.\n"
    "If they are shopping for a vehicle, ask:\n"
    "“What make and model are you interested in, and do you have a budget in mind?”\n"
    "Then check availability via ask_support_brain.\n"
    "If they want to come in, offer to book a test drive or service appointment, and collect the "
    "make and model, a callback number, and a preferred time.\n\n"
    "#Off-Scope Questions\n"
    "If asked about a specific lienholder payoff, state-specific legal paperwork, or complex "
    "insurance claims:\n"
    "“I recommend speaking with one of our sales or finance representatives for that.”\n\n"
    "#Customer Considerations\n"
    "Callers may be comparison shopping or budget-conscious. Stay patient, helpful, and low-pressure.\n\n"
    "#Closing\n"
    "Always ask:\n"
    "“Is there anything else I can help you with today?”\n"
    "Then thank them and say:\n"
    "“Thanks for calling Ascent Automotive Group. We look forward to seeing you!”\n\n"
    "#Tools\n"
    "You have a tool called ask_support_brain, backed by the Ascent Automotive Group knowledge base, "
    "inventory system, and appointment booking. Whenever the caller asks about financing, "
    "warranties, trade-ins, returns, hours, or delivery — or when checking whether a specific "
    "vehicle is available, or booking a test drive or service — call ask_support_brain with the "
    "caller's full question (include any make, model, year, budget, condition, and preferred time "
    "they gave) and relay its answer. Do not rely on memorized details; always ground answers in "
    "ask_support_brain.\n"
    "\n"
    "#Confirmations (important)\n"
    "Sometimes ask_support_brain replies with a confirmation question — for example, asking whether "
    "to go ahead and book an appointment. When that happens, read the question to the caller, then "
    "send their reply — even a simple 'yes', 'no', or a correction like 'make it 3pm' — straight "
    "back to ask_support_brain by calling it again with their answer as the question. Never book, "
    "cancel, or change an appointment on your own; let ask_support_brain finalize it."
)

_GREETING = (
    "Hi! Thanks for calling Ascent Automotive Group customer care. I can help with financing, trade-ins, "
    "finding a vehicle, or booking a test drive. Try the scenarios listed below. How may I help?"
)


# --- "Thinking" filler while the LangChain brain runs -------------------------- #
# The brain may chain several LLM/tool steps (2–5s). Rather than dead air, we speak
# a short filler phrase via InjectAgentMessage so the call feels responsive. Gated
# by a delay in main.py so quick answers don't get an unnecessary "let me look
# that up". Set AGENT_FILLER_DELAY_MS=0 to always speak it; a large value disables.
FILLER_DELAY_MS = int(os.environ.get("AGENT_FILLER_DELAY_MS", "500"))

# Filler phrases grouped by what the caller is asking for, so the "thinking" line
# matches the action (checking the calendar vs. inventory vs. running numbers)
# instead of a generic "let me look that up" every time.
_FILLER_PHRASES = {
    "calendar": [
        "Please wait while I check our calendar for you.",
        "One moment while I check our schedule.",
        "Let me pull up our calendar and find a slot.",
    ],
    "inventory": [
        "Let me check our inventory for you.",
        "One moment while I see what we have in stock.",
        "Let me look that up in our inventory.",
    ],
    "finance": [
        "Let me pull up those numbers for you.",
        "One moment while I work out the figures.",
        "Let me run that calculation for you.",
    ],
    "default": [
        "Let me look that up for you.",
        "One moment while I check that.",
        "Give me just a second.",
    ],
}


def _filler_category(question: str) -> str:
    """Pick a filler category from keywords in the caller's question."""
    q = (question or "").lower()
    if any(w in q for w in ("book", "appointment", "schedul", "calendar", "test drive",
                            "test-drive", "service", "reschedul", "availab", "slot", "time")):
        return "calendar"
    if any(w in q for w in ("inventory", "in stock", "available", "do you have",
                            "vehicle", "model", "trim", "color")):
        return "inventory"
    if any(w in q for w in ("payment", "financ", "loan", "apr", "trade", "down payment",
                            "monthly", "price", "cost", "quote", "estimate")):
        return "finance"
    return "default"


def choose_filler(question: str = "", last_category: str = None, avoid: str = None):
    """Pick a context-matched 'thinking' filler for the InjectAgentMessage.

    Returns (message_dict, category). The category is derived from keywords in the
    question with ZERO added latency (it's data we already have). If the question
    is contentless — e.g. the caller just said "yes" on a confirmation turn — we
    reuse `last_category` so the filler stays on-topic (still calendar, not
    generic). `avoid` skips the last phrase so it never repeats back-to-back.

    behavior="default" => Deepgram speaks it only during silence and harmlessly
    refuses (InjectionRefused) if a turn is already active, so it can never talk
    over the caller. Docs: developers.deepgram.com/docs/voice-agent-inject-agent-message
    """
    category = _filler_category(question)
    if category == "default" and last_category:
        category = last_category  # carry context across a "yes"/"no" confirmation turn
    choices = _FILLER_PHRASES[category]
    pool = [p for p in choices if p != avoid] or choices
    msg = {
        "type": "InjectAgentMessage",
        "message": random.choice(pool),
        "behavior": "default",
    }
    return msg, category


def turn_taking_summary() -> dict:
    """The active Flux end-of-utterance settings, for display in the UI so the
    demo can point at exactly how turn-taking is tuned."""
    return {
        "model": "flux-general-en",
        "eot_threshold": EOT_THRESHOLD,
        "eot_timeout_ms": EOT_TIMEOUT_MS,
        "eager_eot_threshold": float(_EAGER_EOT) if _EAGER_EOT else None,
    }


def build_settings() -> dict:
    """The Settings message sent immediately after the socket opens.

    Configures audio formats and the listen (STT) / think (LLM) / speak (TTS)
    providers, plus — most importantly — the client-side `ask_support_brain`
    function that routes customer questions to the LangChain brain.

    The STT/TTS/LLM provider blocks below mirror the config you provided
    (Flux STT v2, Aura-2 Odysseus, Gemini 3.1 Flash Lite). The `functions`
    array is what wires LangChain in.
    """
    # Flux listen provider + end-of-utterance tuning (see constants above). Built
    # as a dict so the optional eager-EOT param is only sent when configured.
    listen_provider = {
        "type": "deepgram",
        "version": "v2",
        "model": "flux-general-en",
        "eot_threshold": EOT_THRESHOLD,
        "eot_timeout_ms": EOT_TIMEOUT_MS,
    }
    if _EAGER_EOT:
        listen_provider["eager_eot_threshold"] = float(_EAGER_EOT)

    return {
        "type": "Settings",
        "audio": {
            "input": {
                "encoding": "linear16",
                "sample_rate": INPUT_SAMPLE_RATE,
            },
            "output": {
                "encoding": "linear16",
                "sample_rate": OUTPUT_SAMPLE_RATE,
                "container": "none",
            },
        },
        "agent": {
            "greeting": _GREETING,
            "listen": {"provider": listen_provider},
            "think": {
                # The LLM Deepgram uses to drive the conversation and decide WHEN
                # to call our function. The actual dealership answers come
                # from the LangChain brain behind ask_support_brain.
                "provider": {
                    "type": "google",
                    "model": "gemini-3.1-flash-lite",
                },
                "prompt": _SYSTEM_PROMPT,
                "functions": [
                    {
                        "name": "ask_support_brain",
                        "description": (
                            "Send the customer's question to the Ascent Automotive Group customer-care brain "
                            "(dealership knowledge base + inventory availability checker + "
                            "appointment booking) and get back an accurate, ready-to-speak answer. "
                            "Use this for every financing, warranty, trade-in, returns, hours, or "
                            "delivery question, whenever checking whether a specific vehicle is "
                            "available, and whenever booking a test drive or service appointment."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": (
                                        "The customer's full question or request, in natural "
                                        "language. Include any vehicle details they gave, such as "
                                        "the make, model, year, budget, condition (new/used/cpo), "
                                        "and any preferred time for an appointment."
                                    ),
                                }
                            },
                            "required": ["question"],
                        },
                        # NOTE: no "endpoint" key => this is a CLIENT-SIDE function.
                        # Deepgram will send us a FunctionCallRequest to handle.
                    }
                ],
            },
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "model": "aura-2-odysseus-en",
                }
            },
        },
    }
