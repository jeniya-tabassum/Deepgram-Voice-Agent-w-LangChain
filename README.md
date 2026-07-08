# Build a Voice Agent with a Multi-Step LangChain Brain

> A hands-on tutorial: talk to a car-dealership customer-care agent in your
> browser. **Deepgram** handles the voice; a **multi-step LangChain agent**
> does the thinking — chaining tools (RAG, a financing calculator, inventory
> lookups) to answer complex questions. You watch the agent reason **live**,
> tool by tool, on screen.

<p align="center"><em>Two panels: a live call transcript on the left, and the agent's tool-calling loop lighting up on the right.</em></p>

---

## What you'll learn

1. How to connect a **Deepgram Voice Agent** to your own logic with **one
   client-side function** — no public tunnel, all on `localhost`.
2. How to build a **multi-step LangChain agent** that breaks a complex request
   into a *sequence* of tool calls and chains them into one answer.
3. How to build **domain-specific tools** (RAG, a calculator, custom functions)
   the agent can compose.
4. How to **stream the agent's reasoning** to a browser and visualize it live.

The whole point: **the voice layer stays simple, and the "brain" can grow
arbitrarily complex — without ever changing the Deepgram integration.**

---

## The idea in one picture

```
 Browser mic ──PCM16──▶ FastAPI ──WS──▶ Deepgram Voice Agent (STT + router LLM + TTS)
 Browser 🔊  ◀──PCM16── FastAPI ◀──WS──            │
                                                   │  needs a real answer?
                                                   ▼
                                     FunctionCallRequest: ask_support_brain
                                                   │
                                                   ▼
                              🧠 LangChain agent  (agent ⇄ tools loop, OpenAI)
                                 reasons step-by-step, chaining tools:
                                   • search_dealership_info     (RAG)
                                   • check_inventory            (availability + price)
                                   • calculate_monthly_payment  (financing calculator)
                                   • estimate_trade_in          (custom function)
                                   • book_appointment           (custom workflow)
                                                   │
                                                   ▼
                                     FunctionCallResponse ──▶ Deepgram speaks it
```

**Two LLMs, two jobs:**
- **Gemini** (inside Deepgram) is the *router* — decides *when* to call the brain and speaks the result.
- **OpenAI** (inside LangChain) is the *reasoning agent* — decides *which tools to call and in what order*.

---

## Prerequisites

- **Python 3.10+**
- A **Deepgram API key** — https://console.deepgram.com/
- An **OpenAI API key** — https://platform.openai.com/
- A browser with microphone access (Chrome recommended)

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/jeniya-tabassum/Deepgram-Voice-Agent-w-LangChain.git
cd Deepgram-Voice-Agent-w-LangChain

# 2. Create a virtual environment + install deps
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env
#   then edit .env and paste your DEEPGRAM_API_KEY and OPENAI_API_KEY

# 4. Run
uvicorn app.main:app --port 8000
```

Open **http://localhost:8000**, click **Start call**, allow the microphone, and
talk. The right-hand panel lights up as the agent reasons.

> **Tip:** on startup the server prints the agent's graph (`model ⇄ tools` loop)
> as ASCII in the terminal — a nice way to see the shape of the brain.

---

## Talk to it — try these

The three asks are ordered so each triggers **more tool calls** than the last —
watch the trace on the right grow:

| Say this | The agent chains | Tools |
|---|---|---|
| *"How does financing work?"* | one RAG lookup | `search_dealership_info` |
| *"What's the monthly payment on a RAV4 with $5,000 down?"* | look up price → calculate | `check_inventory` → `calculate_monthly_payment` |
| *"I have a 2018 Civic with 60k miles to trade in — what's the monthly on a new RAV4?"* | value trade-in → look up price → calculate | `estimate_trade_in` → `check_inventory` → `calculate_monthly_payment` |

That last one is the payoff: **the agent decides the 3-step sequence itself** —
nothing is hard-coded.

---

## How it works — a guided tour

### Part 1 — The Deepgram Voice Agent (`app/deepgram_agent.py`)

Deepgram's Voice Agent handles all the hard real-time audio (speech-to-text with
Flux, text-to-speech with Aura-2, turn-taking, barge-in). We connect to it over a
WebSocket and send one **Settings** message that configures the STT/LLM/TTS
providers — and declares **exactly one function**, `ask_support_brain`:

```python
"functions": [
    {
        "name": "ask_support_brain",
        "description": "Send the customer's question to the Ascent Automotive Group brain …",
        "parameters": { "type": "object", "properties": { "question": {"type": "string"} }, "required": ["question"] },
        # NOTE: no "endpoint" key  =>  this is a CLIENT-SIDE function.
    }
]
```

Because there's **no `endpoint`**, this is a *client-side* function: when the
model decides to call it, Deepgram sends the call back to us over the **same
WebSocket**. That's why the whole demo runs on `localhost` with no ngrok/tunnel.

The system prompt tells Gemini to *always* route real questions through
`ask_support_brain` and never invent answers — so Gemini is purely a router.

### Part 2 — The multi-step agent (`app/langchain_brain.py`)

This is the heart of the tutorial. The brain is a LangChain **tool-calling
agent**, built with `create_agent`:

```python
from langchain.agents import create_agent

def build_support_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500)
    return create_agent(llm, tools=TOOLS, system_prompt=_SYSTEM_PROMPT)
```

`create_agent` compiles an **agent ⇄ tools loop**: the model runs, optionally
calls one or more tools, reads the results, and loops until it can answer. *That
loop is the multi-step reasoning.*

The five tools are ordinary Python functions decorated with `@tool`. Their
docstrings tell the agent when to use them — for example the calculator:

```python
@tool
def calculate_monthly_payment(price: float, down_payment: float = 0.0,
                              apr_percent: float = 6.9, term_months: int = 60) -> str:
    """Financing calculator … ALWAYS use this for payment math — never compute it
    yourself. Get the price from check_inventory first."""
    principal = max(0.0, float(price) - float(down_payment))
    r = float(apr_percent) / 100 / 12
    monthly = principal / term_months if r == 0 else principal * r / (1 - (1 + r) ** (-term_months))
    return json.dumps({"monthly_payment": round(monthly, 2), ...})
```

The system prompt instructs the agent to *break a complex request into a
sequence of tool calls and chain them* — e.g. get a trade-in value, use it as the
down payment, look up the vehicle price, then compute the monthly payment.

### Part 3 — Streaming the reasoning to the browser

To visualize the agent live, we stream it with LangGraph's `updates` mode and
classify each new message:

```python
async for chunk in agent.astream({"messages": [{"role": "user", "content": question}]},
                                 stream_mode="updates"):
    for _node, update in chunk.items():
        for m in update.get("messages", []):
            if isinstance(m, AIMessage) and m.tool_calls:
                # the agent decided to call a tool  -> emit "tool_call"
            elif isinstance(m, ToolMessage):
                # a tool returned                    -> emit "tool_result"
            elif isinstance(m, AIMessage):
                answer = m.content                   # the final spoken answer
```

Each event is forwarded to the browser, which lights up the corresponding tool
node and appends a row to the live trace (`frontend/app.js`).

### Part 4 — The bridge (`app/main.py`)

`main.py` is a thin FastAPI app that bridges the browser and Deepgram over two
WebSockets. The key handler: when Deepgram emits a `FunctionCallRequest` for
`ask_support_brain`, we run the agent (streaming its steps to the UI) and reply
with a `FunctionCallResponse` on the same socket:

```python
await _safe_send_text(browser, {"type": "graph_start"})
content = await run_support_brain(_support_agent, question, on_event=_on_event)
await _safe_send_text(browser, {"type": "graph_done"})
await dg.send(json.dumps({"type": "FunctionCallResponse", "id": call_id,
                          "name": name, "content": content}))
```

That round-trip **is** the entire integration.

---

## Extending it: add your own tool

Adding a capability is just writing a function and listing it:

```python
@tool
def check_incentives(make: str, model: str) -> str:
    """Look up current manufacturer rebates/incentives for a vehicle."""
    # ... call your real API here ...
    return json.dumps({"make": make, "model": model, "rebate": 1500})

TOOLS = [search_dealership_info, check_inventory, calculate_monthly_payment,
         estimate_trade_in, book_appointment, check_incentives]  # <- add it
```

The agent will start using it whenever the docstring is relevant — **no changes
to the Deepgram integration.** (To also light it up in the UI, add a node in
`frontend/index.html` and map its name in `TOOL_UI_IDS`.)

---

## From demo to production

Every mock is a clean swap; the voice integration never changes:

| Demo piece | Production swap |
|---|---|
| `TFIDFRetriever` over a hard-coded FAQ | Embeddings + a real vector store (pgvector, Pinecone, Weaviate) |
| Mock `check_inventory` / `calculate_monthly_payment` / `book_appointment` | Real inventory / DMS / lender + booking API calls inside the tools |
| `ChatOpenAI` (gpt-4o-mini) | Any LangChain chat model — keep it fast for low voice latency |
| Browser `ScriptProcessorNode` | `AudioWorkletNode` (modern, glitch-free capture) |

---

## Project layout

```
app/
  main.py             FastAPI: bridges browser <-> Deepgram, streams agent steps to the UI
  deepgram_agent.py   Voice Agent WebSocket + Settings message (declares ask_support_brain)
  langchain_brain.py  ⭐ the multi-step tool-calling agent + its 5 domain tools
  knowledge_base.py   demo dealership FAQ (swap for your real docs / vector store)
frontend/
  index.html          call UI + live transcript + live agent-reasoning panel
  app.js              mic capture, PCM streaming, TTS playback, barge-in, agent-step rendering
requirements.txt
.env.example          copy to .env and add your keys
```

---

## Troubleshooting

- **The right-hand panel stays grey.** Hard-refresh the tab (**Cmd/Ctrl+Shift+R**)
  to clear a cached `app.js`. Open the console — you should see
  `[agent-panel] agent_step …` lines when you speak.
- **401 / invalid API key.** The app loads `.env` with `override=True`, so `.env`
  wins over your shell environment. Make sure the key in `.env` is valid.
- **No answer / it can't hear you.** Check the mic permission and that you see a
  live transcript. Watch the terminal for `[brain] steps: …` to confirm the agent
  ran.

---

## Two ways to wire LangChain into a Voice Agent

- **(A) Client-side function calling** — what this tutorial uses. Simplest, fully
  local. Deepgram calls your function back over the existing socket.
- **(B) Custom LLM endpoint** — point `agent.think.provider` at your own
  OpenAI-compatible `/chat/completions` server backed by LangChain, so *every*
  turn is reasoned by LangChain. More powerful, but requires a publicly reachable
  URL (Deepgram's servers call it directly).

---

## License

MIT — use it, fork it, build on it.
