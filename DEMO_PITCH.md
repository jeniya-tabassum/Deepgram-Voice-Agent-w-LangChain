# Demo Pitch — LangChain Multi-Step Agent + Deepgram Voice Agent

**Goal of this demo:** show that a **Deepgram Voice Agent can drive a genuine
multi-step LangChain agent** — one that breaks a complex spoken request into a
*sequence of tool calls* (RAG, a financing calculator, inventory lookups, custom
functions) and chains them into one answer. The UI shows the agent's tool-calling
loop **live** as you talk. Use case: **Summit Motors**, a car dealership. No public
tunnel, all on localhost.

---

## The one-liner

> "Deepgram handles the *voice*. But it never makes up answers — when the caller
> asks something real, it calls one function, and behind it a **LangChain agent
> reasons in multiple steps**, calling tools in sequence until it can answer. You
> watch it happen on the right, tool by tool."

## The core idea to hammer

There are **two LLMs with two different jobs**:

- **Gemini (inside Deepgram)** = conversation *router*. Decides *when* to reach
  for the brain and speaks the result aloud.
- **OpenAI (inside LangChain)** = the *reasoning agent*. Given a question, it
  decides **which tools to call and in what order**, chaining them — e.g.
  trade-in value → inventory price → monthly payment — before answering.

> The voice layer stays dumb and fast; the agent can reason through as many steps
> and tools as the question needs — **and Deepgram never changes**, because it
> always just calls the one `ask_support_brain` function.

---

## Run it

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

Open **http://localhost:8000**, click **Start call**, allow the mic, and talk.
(Test mic/audio in the browser once *before* the live demo.) At startup the
server prints the agent's graph (a `model ⇄ tools` loop) as ASCII.

**Two panels:** left = live transcript, right = **live agent reasoning**: an
`agent` hub wired to 5 tools, each lighting up (with a step-order badge) as the
agent calls it, plus a step-by-step trace of `tool(args) → result`. That trace is
the money shot — it's the *real* streamed reasoning, not an animation.

---

## Demo flow — escalating tool chains

> The three asks are ordered so each one triggers **more tool calls** than the
> last. Watch the right-hand trace grow. The server log also prints
> `[brain] steps: … -> answer`.

### 1. One tool — RAG

**Ask:** *"How does financing work at your dealership?"*

**Chain:** `search_dealership_info` → answer.
The agent recognizes a policy question, does one RAG lookup, and speaks a grounded
answer. One row in the trace.

**Show:** `app/langchain_brain.py:67` (`search_dealership_info`) and `:52` (the
retriever).

---

### 2. Two tools — inventory feeds the calculator

**Ask:** *"What would my monthly payment be on a RAV4 with 5,000 dollars down?"*

**Chain:** `check_inventory(Toyota, RAV4)` → *gets `price_from` $27,990* →
`calculate_monthly_payment(27990, down=5000)` → answer.
**Point out:** the agent can't answer in one shot — it must first look up the
price, then feed it into the calculator. Two rows appear, in order.

**Show:** `app/langchain_brain.py:78` (`check_inventory`, returns `price_from`)
and `:116` (`calculate_monthly_payment` — real loan amortization; the agent is
told to *never* do the math itself).

---

### 3. Three tools — the "wow" (a full reasoning chain)

**Ask:** *"I have a 2018 Honda Civic with 60,000 miles to trade in — what would my
monthly payment be on a new Toyota RAV4?"*

**Chain:** `estimate_trade_in(Civic, 2018, 60k)` → *~$5,200* →
`check_inventory(Toyota, RAV4, new)` → *from $27,990* →
`calculate_monthly_payment(27990, down=5200)` → *~$450/mo* → answer.
**The agent picks this 3-step sequence itself** — nothing is hard-coded. Watch
three tools light up in order on the right, each with its result in the trace,
then one short spoken answer: *"About $450 a month, with your $5,200 trade-in."*

**Show:** `app/langchain_brain.py:148` (`estimate_trade_in`). Then point at the
system prompt (`:206`): *"break a complex request into a sequence of tool calls
and chain them."*

---

## The live panel — how it works (if asked)

The agent streams step-by-step with `astream(stream_mode="updates")`
(`app/langchain_brain.py:296`). We classify each new message: an `AIMessage` with
`tool_calls` = the agent decided to call a tool; a `ToolMessage` = a tool
returned. Each becomes an `agent_step` event pushed to the browser
(`app/main.py:77`), which lights the tool node and appends to the trace. So what
you see is the **real** reasoning loop.

---

## The handoff, in code (if an engineer asks "how does the call cross over?")

1. Deepgram emits `FunctionCallRequest` on the WebSocket. → `app/main.py:107`
2. Backend streams the agent, forwarding each step to the UI.
   → `app/main.py:80` calls `run_support_brain(..., on_event=...)`
3. Backend replies with `FunctionCallResponse` on the same socket. → `app/main.py:88`

That round-trip **is** the entire integration.

---

## The agent itself

**Show:** `app/langchain_brain.py:243` — `build_support_agent()`:

- LangChain v1 **`create_agent`** (`:249`) — a LangGraph **agent⇄tools loop**: the
  model runs, optionally calls tools, sees the results, and loops until it can
  answer. That loop *is* the multi-step reasoning.
- over `ChatOpenAI` (gpt-4o-mini, temperature 0).
- **five domain tools** (`:202`): `search_dealership_info` (RAG),
  `check_inventory`, `calculate_monthly_payment` (calculator), `estimate_trade_in`
  (custom function), `book_appointment` (custom workflow).

**The punchline:** the agent can grow arbitrarily complex — more tools, external
APIs, sub-agents, human-in-the-loop — and **Deepgram never changes.** It still
calls the same one `ask_support_brain` function.

---

## Anticipated questions

**"Is this real multi-step reasoning or just one function call?"**
Real. There are **two** layers: Deepgram calling `ask_support_brain` (one call),
and — inside it — the agent's own loop that may call *several* tools in sequence.
The 3-tool trade-in question proves it: three distinct tool calls, each feeding
the next, chosen by the model at runtime. Watch the trace.

**"Why an agent instead of hard-coded logic?"**
Because the *number and order* of steps depends on the question. "How does
financing work?" needs one tool; the trade-in question needs three. An agent
plans that per-query; hard-coded routing can't cover the combinatorial space.
You also get tool abstraction, RAG, tracing, and model-swapping for free.

**"Could tools call real systems?"**
Yes — each tool is an ordinary function. Swap `check_inventory` for a real
inventory/DMS API, `calculate_monthly_payment` for a lender quote API,
`book_appointment` for your scheduler. The agent and the Deepgram integration
don't change. (See the production table in `README.md`.)

**"Why no ngrok / public URL?"**
Client-side function calling — Deepgram calls back over the existing socket. The
alternative (a custom LLM endpoint reasoning *every* turn) needs a public URL; we
chose the simpler local path.

---

## Closing line

> "Deepgram gives you production-grade real-time voice; LangChain gives you an
> agent that reasons through as many steps and tools as the question demands. The
> agent on the right can get as sophisticated as you want — and the voice
> integration is still one function and one WebSocket message."
