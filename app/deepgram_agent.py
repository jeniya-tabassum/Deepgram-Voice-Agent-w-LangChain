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

import websockets

DEEPGRAM_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"

# Audio formats exchanged with Deepgram (must match what the browser sends/plays).
INPUT_SAMPLE_RATE = 48000   # mic PCM the browser streams up
OUTPUT_SAMPLE_RATE = 24000  # TTS PCM Deepgram streams back


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


# The Summit Motors voice persona / call flow. Note the appended "#Tools" section:
# it tells the think-LLM to route every dealership question through the
# client-side `ask_support_brain` function, which is handled by LangChain.
_SYSTEM_PROMPT = (
    "#Role\n"
    "You are a virtual customer-care assistant for Summit Motors, a car dealership, speaking to "
    "customers over the phone. Your task is to help them with financing, warranties, trade-ins, "
    "vehicle availability, and booking test drives or service.\n\n"
    "#General Guidelines\n"
    "Be warm, helpful, and professional.\n"
    "Speak clearly and naturally in plain language.\n"
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
    "Greet the caller and welcome them to Summit Motors. Ask how you can help.\n"
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
    "“Thanks for calling Summit Motors. We look forward to seeing you!”\n\n"
    "#Tools\n"
    "You have a tool called ask_support_brain, backed by the Summit Motors knowledge base, "
    "inventory system, and appointment booking. Whenever the caller asks about financing, "
    "warranties, trade-ins, returns, hours, or delivery — or when checking whether a specific "
    "vehicle is available, or booking a test drive or service — call ask_support_brain with the "
    "caller's full question (include any make, model, year, budget, condition, and preferred time "
    "they gave) and relay its answer. Do not rely on memorized details; always ground answers in "
    "ask_support_brain."
)

_GREETING = (
    "Hi! Thanks for calling Summit Motors customer care. I can help with financing, trade-ins, "
    "finding a vehicle, or booking a test drive. Try the scenarios listed below. How may I help?"
)


def build_settings() -> dict:
    """The Settings message sent immediately after the socket opens.

    Configures audio formats and the listen (STT) / think (LLM) / speak (TTS)
    providers, plus — most importantly — the client-side `ask_support_brain`
    function that routes customer questions to the LangChain brain.

    The STT/TTS/LLM provider blocks below mirror the config you provided
    (Flux STT v2, Aura-2 Odysseus, Gemini 3.1 Flash Lite). The `functions`
    array is what wires LangChain in.
    """
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
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "version": "v2",
                    "model": "flux-general-en",
                }
            },
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
                            "Send the customer's question to the Summit Motors customer-care brain "
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
