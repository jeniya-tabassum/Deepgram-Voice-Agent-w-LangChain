"""Offline smoke test — exercises the LangGraph brain WITHOUT Deepgram or a mic.

Proves the two features added to the POC:
  #4  Flux end-of-utterance config is present in the Voice Agent Settings.
  #3  Human-in-the-loop resume-from-node: a booking PAUSES for confirmation,
      then RESUMES from the same node on the caller's reply (yes -> booked,
      no -> not booked), and multi-step reasoning still chains correctly.

Run from the project root:
    PYTHONPATH=. .venv/bin/python scripts/smoke_hitl.py
Requires DEEPGRAM_API_KEY (for import) and OPENAI_API_KEY (the brain) in .env.
"""
import asyncio

from dotenv import load_dotenv

load_dotenv(override=True)

from app.deepgram_agent import build_settings
from app.langchain_brain import build_support_agent, run_support_brain


def check_settings():
    s = build_settings()
    prov = s["agent"]["listen"]["provider"]
    print("LISTEN provider:", prov)
    assert prov["model"] == "flux-general-en" and prov["version"] == "v2"
    assert "eot_threshold" in prov and "eot_timeout_ms" in prov
    assert "eager_eot_threshold" not in prov  # off by default
    print("OK: EOT params present, eager off by default\n")


async def main():
    check_settings()
    agent = build_support_agent()

    # --- 1. Booking should PAUSE (interrupt), not book immediately -----------
    events = []
    async def cap(p):
        events.append(p)
    r1 = await run_support_brain(
        agent,
        "I'd like to book a test drive for a RAV4 this Saturday at 2pm, call me at 555-1234.",
        thread_id="call-A",
        on_event=cap,
    )
    print("TURN 1 (book):", r1)
    assert r1["interrupted"] is True, "expected graph to pause for confirmation"
    assert "book" in r1["content"].lower()
    intr = [e for e in events if e.get("kind") == "interrupt"]
    print("  interrupt event ->", intr)
    assert intr and intr[0].get("ui") == "booking" and intr[0].get("prompt"), "UI needs kind/ui/prompt"

    # --- 2. Resume same thread with 'yes' -> should book from the SAME node ---
    r2 = await run_support_brain(agent, "Yes, go ahead.", thread_id="call-A", resume=True)
    print("TURN 2 (resume yes):", r2)
    assert r2["interrupted"] is False
    assert "APT-" in r2["content"] or "booked" in r2["content"].lower()

    # --- 3. Different call: pause, then decline with a correction ------------
    r3 = await run_support_brain(
        agent,
        "Book me a service appointment for my Camry this Friday at 9am, my number is 555-9876.",
        thread_id="call-B",
    )
    print("TURN 3 (book):", r3)
    assert r3["interrupted"] is True
    r4 = await run_support_brain(agent, "No, make it the afternoon instead.", thread_id="call-B", resume=True)
    print("TURN 4 (resume no):", r4)
    assert r4["interrupted"] is False
    # Decline invariant: the graph resumed but did NOT produce a confirmed booking.
    assert "APT-" not in r4["content"] and "is booked" not in r4["content"].lower()

    # --- 4. Multi-step reasoning still works (no regression) -----------------
    steps = []
    async def on_event(p):
        if p.get("kind") == "tool_call":
            steps.append(p["name"])
    r5 = await run_support_brain(
        agent,
        "I have a 2018 Honda Civic in good condition with 60000 miles to trade in. Use that trade-in "
        "as the down payment and tell me the estimated monthly payment on a new Toyota RAV4.",
        thread_id="call-C",
        on_event=on_event,
    )
    print("TURN 5 (multi-step):", r5, "| tools:", steps)
    assert r5["interrupted"] is False
    assert len(steps) >= 2, f"expected multi-step chain, got {steps}"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
