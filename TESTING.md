# Testing Guide

How to verify this POC — both a **fast offline check** (no mic, ~30s) and the
**live browser demo** with exact things to say. Covers the two features added on
top of the base demo:

- **#3 — LangGraph human-in-the-loop** (graph pauses for confirmation, then
  **resumes from the same node**, not a restart)
- **#4 — Flux end-of-utterance tuning** (turn-taking that doesn't cut off
  slower/natural speech) + barge-in

> New to the architecture? Read `DEMO_PREP_GUIDE.md` first — it explains *why* each
> of these behaves the way it does.

---

## 0. Prerequisites

```bash
cd /Users/jeniya/Desktop/CODE/Prospect/Friedkin

# Both keys must be set (should print the two names)
grep -oE '^(DEEPGRAM_API_KEY|OPENAI_API_KEY)=' .env
```

- `DEEPGRAM_API_KEY` — needed for the live browser call (and module import).
- `OPENAI_API_KEY` — the LangGraph brain (`gpt-4o-mini`).
- Dependencies are already in `.venv` (see `requirements.txt`).

---

## A. Offline smoke test (no mic, ~30s)

Exercises the brain's HITL pause→resume and the EOT config **without** Deepgram or
a microphone. Best first check and safe to run right before a demo.

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_hitl.py
```

**Expected output (abridged):**

```
LISTEN provider: {... 'model': 'flux-general-en', 'version': 'v2',
                  'eot_threshold': 0.8, 'eot_timeout_ms': 8000}
OK: EOT params present, eager off by default

TURN 1 (book):        {'content': "...Should I go ahead and book it?", 'interrupted': True}
  interrupt event ->  [{'kind': 'interrupt', 'ui': 'booking', 'prompt': '...'}]
TURN 2 (resume yes):  {'content': "...booked ... (APT-XXXXXX)...", 'interrupted': False}
TURN 3 (book):        {... 'interrupted': True}
TURN 4 (resume no):   {... 'interrupted': False}   # not booked
TURN 5 (multi-step):  {...} | tools: ['estimate_trade_in', 'check_inventory', 'calculate_monthly_payment']

ALL CHECKS PASSED
```

**What each turn proves:**

| Turn | Proves |
|---|---|
| LISTEN line | #4 — Flux EOT params (`eot_threshold`, `eot_timeout_ms`) are in the Settings; eager off by default |
| 1 | #3 — a booking request **pauses** the graph (`interrupted: True`) and emits a UI `interrupt` event with `ui:"booking"` + prompt |
| 2 | #3 — replying "yes" **resumes from the paused node** and completes the booking (`APT-…`) |
| 3–4 | #3 — replying "no" resumes but produces **no** confirmed booking |
| 5 | no regression — the agent still chains 3 tools for the trade-in question |

> Note: the reasoning LLM is non-deterministic on *wording* and occasionally asks a
> clarifying question instead of chaining. The **mechanism** is stable; the test
> gives complete details so it chains reliably. If a run phrases things
> differently, that's expected.

---

## B. Live browser demo

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

Open **http://localhost:8000**, click **📞 Start call**, allow the mic.
(Test your mic/audio once *before* a live demo.)

The screen has two panels: **left** = live transcript + a turn-taking pill;
**right** = the live LangGraph reasoning panel (nodes light up as tools fire).

### Test #4 — turn-taking / end-of-utterance
Watch the **pill under the transcript** and the caption beside it.

| Say / do | Expect |
|---|---|
| *(just after connecting)* | Caption: **"Flux · ends turn at ≥0.80 confidence · waits up to 8s pause"**; pill = `👂 Listening` |
| *"How does financing work?"* | Pill cycles `🎙 Caller speaking` → `💭 Agent thinking` → `🔊 Agent speaking` → `👂 Listening` |
| Speak a sentence with a **2–3s pause mid-way** (e.g. *"I'm looking for… ummm… a RAV4"*) | Agent does **not** cut you off during the pause (raised `eot_timeout_ms`) |
| Start talking **while the agent is speaking** | Barge-in: agent audio stops instantly; pill flips to `🎙 Caller speaking` |

### Test #3 — human-in-the-loop resume-from-node
Watch the **`book_appointment` node** (right panel) and the banner below it.

1. **Say:** *"I'd like to book a test drive for a RAV4 this Saturday at 2pm, my
   number is 555-1234."*
   - **Expect:** `book_appointment` node **pulses amber**; amber banner
     *"⏸ Human-in-the-loop — graph paused at book_appointment, waiting for the
     caller to confirm"*; a `⏸ interrupt()` row in the trace; the agent **speaks a
     confirmation question**. Nothing is booked yet.
2. **Say:** *"Yes, go ahead."*
   - **Expect:** banner turns green *"▶ Resumed from checkpoint — same graph, same
     node (not restarted)"*; the trace **continues** (does not clear); a booking
     confirmation (`APT-XXXXXX`) is spoken.
3. **Repeat step 1, then say:** *"No, make it 3pm instead."*
   - **Expect:** the graph resumes but **no booking is confirmed** — the agent asks
     what to change.

> **The point to make to the customer at step 2:** the trace *continued from the
> paused node* rather than restarting — that's the resume-from-node behavior, the
> thing they said was hardest.

---

## C. Optional — prove the EOT tuning is live and configurable

Restart with different turn-taking settings and confirm the UI caption changes:

```bash
DG_EOT_THRESHOLD=0.6 DG_EOT_TIMEOUT_MS=4000 DG_EAGER_EOT_THRESHOLD=0.5 \
  PYTHONPATH=. .venv/bin/uvicorn app.main:app --port 8000
```

Reload the page → caption reads **"…≥0.60 confidence · waits up to 4s pause ·
eager 0.50"**. (Lower threshold = ends turns faster; eager = starts responding
before you fully finish.) **Revert to defaults for the real demo** (just restart
without the env vars).

| Env var | Default | Effect |
|---|---|---|
| `DG_EOT_THRESHOLD` | `0.8` | Confidence to end the turn (higher = more patient) |
| `DG_EOT_TIMEOUT_MS` | `8000` | Hard cap: end turn this long after speech |
| `DG_EAGER_EOT_THRESHOLD` | *(unset)* | Opt-in: respond before caller fully finishes (lower latency, more interruption risk) |

---

## D. Quick sanity checks (wiring didn't break)

```bash
# Frontend JS parses
node --check frontend/app.js

# Backend imports + EOT config helper works
PYTHONPATH=. .venv/bin/python -c \
  "import app.main; from app.deepgram_agent import turn_taking_summary; print(turn_taking_summary())"

# Server serves the page + new UI elements
.venv/bin/uvicorn app.main:app --port 8077 & sleep 6
curl -s http://localhost:8077/ | grep -oE 'id="(turnState|hitlBanner)"|app.js\?v=4'
kill %1
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| No audio or no transcript | Check the server terminal for a Deepgram `Error` status; check the browser console (`[status]`, `[agent-panel]` logs). Usually a mic permission or a stale `DEEPGRAM_API_KEY`. |
| Agent asks for details instead of pausing at booking | It only reaches `book_appointment` (and the interrupt) once it has an appointment type **and** a time. Provide them — correct behavior, not a bug. |
| Brain errors mid-call | It returns a safe spoken fallback ("I'm having trouble… I can connect you with a specialist"). The call won't crash. |
| Reasoning varies run-to-run | The LLM is non-deterministic on phrasing/step choice; the mechanism is stable. Don't promise identical wording. |
| Fastest recovery mid-demo | End the call and click **Start** again — each call gets a fresh `thread_id`, so state resets cleanly. |
| `ModuleNotFoundError: No module named 'app'` | Run from the project root and prefix with `PYTHONPATH=.` |
