"""Proof that LangGraph resumes from the PAUSED node, not from START.

Inspects the checkpoint's `.next` pointer after interrupt() and after resume.
Run:  PYTHONPATH=. .venv/bin/python scripts/prove_resume.py
"""
import asyncio

from dotenv import load_dotenv

load_dotenv(override=True)

from langgraph.types import Command
from app.langchain_brain import build_support_agent


async def main():
    agent = build_support_agent()
    cfg = {"configurable": {"thread_id": "proof-1"}}

    # Turn 1: a booking -> hits interrupt() and PAUSES
    async for _ in agent.astream(
        {"messages": [{"role": "user",
         "content": "Book a test drive for a RAV4 Saturday 2pm, phone 555-1234."}]},
        config=cfg, stream_mode="updates"):
        pass

    snap = await agent.aget_state(cfg)
    print("=== AFTER INTERRUPT (paused) ===")
    print("next node(s)         :", snap.next)                      # where it will resume
    print("is it START?         :", (snap.next == ()) or ("__start__" in snap.next))
    print("messages so far      :", len(snap.values.get("messages", [])))
    tasks = snap.tasks
    print("pending task         :", tasks[0].name if tasks else None)
    print("has pending interrupt:", bool(tasks and tasks[0].interrupts))
    msgs_before = len(snap.values.get("messages", []))

    # Turn 2: resume with Command(resume=...) on the SAME thread_id
    async for _ in agent.astream(Command(resume="yes"), config=cfg, stream_mode="updates"):
        pass
    snap2 = await agent.aget_state(cfg)
    print("\n=== AFTER RESUME ===")
    print("next node(s)         :", snap2.next, "(empty = graph finished)")
    print("messages now         :", len(snap2.values.get("messages", [])),
          "(grew — prior state kept, not reset to 0)")
    print("prior messages kept  :", len(snap2.values.get("messages", [])) >= msgs_before)


if __name__ == "__main__":
    asyncio.run(main())
