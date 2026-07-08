"""
FastAPI app that bridges the browser <-> Deepgram Voice Agent, and routes the
agent's function calls into the LangChain brain.

Data flow per call:

    Browser mic (PCM16) --WS--> FastAPI --WS--> Deepgram Voice Agent
    Browser speaker      <--WS-- FastAPI <--WS-- Deepgram (TTS PCM16 + events)

When Deepgram emits a FunctionCallRequest for `ask_support_brain`, we run the
LangChain agent and send back a FunctionCallResponse — all over the same socket.
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from websockets.exceptions import ConnectionClosed

from .deepgram_agent import (
    FILLER_DELAY_MS,
    build_settings,
    choose_filler,
    connect_deepgram,
    turn_taking_summary,
)
from .langchain_brain import build_support_agent, run_support_brain

# override=True so the project's .env is authoritative even if a (possibly stale
# or malformed) key of the same name is already exported in your shell.
load_dotenv(override=True)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Ascent Automotive Group — Deepgram + LangChain Voice Agent")

# Built once at startup and reused for every call.
_support_agent = None


@app.on_event("startup")
async def _startup() -> None:
    global _support_agent
    _support_agent = build_support_agent()
    print("[main] LangChain support agent ready.")
    # Print the LangGraph topology so the demo can show how complex the brain is.
    try:
        print("[main] LangGraph brain topology:")
        print(_support_agent.get_graph().draw_ascii())
    except Exception as exc:  # noqa: BLE001 - drawing is best-effort (needs grandalf)
        print(f"[main] (graph ascii unavailable: {exc!r})")


@app.get("/")
async def index() -> FileResponse:
    # no-cache so edits to the UI (and the ?v= bust on app.js) are always picked up.
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ---- the bridge --------------------------------------------------------------
async def _handle_function_call(fn: dict, dg, browser: WebSocket, session: dict) -> None:
    """Run one client-side function call through LangChain and reply to Deepgram.

    Spawned as a background task so audio keeps flowing while the brain thinks.

    `session` holds the per-call graph state: a stable `thread_id` (so the brain's
    checkpointed state follows this one phone call) and `pending_interrupt` (set
    when the graph paused for human confirmation). While a pause is pending, the
    caller's next reply RESUMES the graph from the interrupted node instead of
    starting a new turn.
    """
    name = fn.get("name")
    call_id = fn.get("id")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}

    # Let the UI show that the brain was invoked.
    await _safe_send_text(browser, {"type": "function_call", "name": name, "arguments": args})

    if name == "ask_support_brain":
        # Stream each agent reasoning step to the browser so the live panel updates.
        async def _on_event(payload: dict) -> None:
            await _safe_send_text(browser, {"type": "agent_step", **payload})

        # If the graph is paused awaiting confirmation, this turn resumes it.
        resume = session.get("pending_interrupt", False)
        session["pending_interrupt"] = False

        # Mask latency: if the brain takes longer than FILLER_DELAY_MS, speak a
        # short "let me look that up" filler so the caller isn't left in silence.
        # Cancelled the moment the answer is ready, so fast turns stay filler-free.
        async def _speak_filler() -> None:
            try:
                await asyncio.sleep(FILLER_DELAY_MS / 1000)
                # Match the filler to what was asked (carrying context across a
                # yes/no confirmation turn), and don't repeat the last one.
                msg, category = choose_filler(
                    args.get("question", ""),
                    last_category=session.get("filler_category"),
                    avoid=session.get("last_filler"),
                )
                session["last_filler"] = msg["message"]
                session["filler_category"] = category
                await dg.send(json.dumps(msg))
                await _on_event({"kind": "filler", "text": msg["message"]})
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - filler must never break the call
                pass

        filler_task = asyncio.create_task(_speak_filler())

        await _safe_send_text(browser, {"type": "graph_start", "resume": resume})
        try:
            result = await run_support_brain(
                _support_agent,
                args.get("question", ""),
                thread_id=session["thread_id"],
                resume=resume,
                on_event=_on_event,
            )
        finally:
            filler_task.cancel()
        await _safe_send_text(browser, {"type": "graph_done"})
        content = result["content"]
        # If the graph paused for human-in-the-loop, remember it so the caller's
        # next reply is routed back in as a resume rather than a fresh question.
        if result.get("interrupted"):
            session["pending_interrupt"] = True
    else:
        content = f"Unknown function '{name}'."

    await dg.send(
        json.dumps(
            {
                "type": "FunctionCallResponse",
                "id": call_id,
                "name": name,
                "content": content,
            }
        )
    )
    await _safe_send_text(browser, {"type": "function_result", "name": name, "content": content})


async def _handle_dg_event(message: str, dg, browser: WebSocket, session: dict) -> None:
    """Handle a JSON (text) event from Deepgram."""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return

    msg_type = data.get("type")

    if msg_type == "FunctionCallRequest":
        for fn in data.get("functions", []):
            # client_side defaults to True for functions with no endpoint.
            if fn.get("client_side", True):
                asyncio.create_task(_handle_function_call(fn, dg, browser, session))
        return

    if msg_type == "ConversationText":
        await _safe_send_text(
            browser,
            {"type": "transcript", "role": data.get("role"), "content": data.get("content")},
        )
    elif msg_type == "UserStartedSpeaking":
        # Barge-in: tell the browser to drop any queued/playing TTS audio.
        await _safe_send_text(browser, {"type": "user_started_speaking"})
    elif msg_type in ("Welcome", "SettingsApplied", "AgentAudioDone", "AgentThinking",
                      "AgentStartedSpeaking", "Error", "Warning"):
        await _safe_send_text(browser, {"type": "status", "event": msg_type, "detail": data})


async def _safe_send_text(browser: WebSocket, payload: dict) -> None:
    try:
        await browser.send_text(json.dumps(payload))
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket("/ws")
async def ws_endpoint(browser: WebSocket) -> None:
    await browser.accept()
    dg = await connect_deepgram()
    await dg.send(json.dumps(build_settings()))

    # Per-call state. thread_id scopes the LangGraph brain's checkpointed state to
    # THIS call; pending_interrupt tracks whether the brain is paused for human
    # confirmation (so the next reply resumes the graph). See _handle_function_call.
    session = {"thread_id": uuid.uuid4().hex, "pending_interrupt": False}

    # Tell the UI how turn-taking is tuned, so the demo can show the EOT settings.
    await _safe_send_text(browser, {"type": "config", "turn_taking": turn_taking_summary()})

    async def browser_to_deepgram() -> None:
        while True:
            msg = await browser.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            audio = msg.get("bytes")
            if audio:
                await dg.send(audio)

    async def deepgram_to_browser() -> None:
        async for message in dg:
            if isinstance(message, (bytes, bytearray)):
                # Binary frame = TTS audio for the speaker.
                try:
                    await browser.send_bytes(bytes(message))
                except (WebSocketDisconnect, RuntimeError):
                    break
            else:
                await _handle_dg_event(message, dg, browser, session)

    # Run both directions; as soon as either side closes, cancel the other so we
    # don't try to send on a closed socket.
    tasks = [asyncio.create_task(browser_to_deepgram()), asyncio.create_task(deepgram_to_browser())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except (WebSocketDisconnect, ConnectionClosed):
        pass
    finally:
        try:
            await dg.close()
        except Exception:  # noqa: BLE001
            pass


# Serve the rest of the static frontend (app.js, etc.). Mounted last so it does
# not shadow the routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
