/* Summit Motors — browser voice client.
 *
 * Responsibilities:
 *   - capture mic audio, downsample to 16 kHz PCM16, stream to the backend WS
 *   - receive TTS PCM16 (24 kHz) from the backend and play it back gaplessly
 *   - render the live transcript + a chip when the LangChain brain is called
 *   - light up the LangGraph panel live as each graph node fires
 *   - handle barge-in: flush queued audio when the user starts speaking
 */

const TARGET_INPUT_RATE = 48000; // must match deepgram_agent.INPUT_SAMPLE_RATE
const OUTPUT_RATE = 24000; // must match deepgram_agent.OUTPUT_SAMPLE_RATE

const callBtn = document.getElementById("callBtn");
const dot = document.getElementById("dot");
const statusText = document.getElementById("statusText");
const transcriptEl = document.getElementById("transcript");
const emptyHint = document.getElementById("emptyHint");
const graphPathEl = document.getElementById("graphPath");
const agentTraceEl = document.getElementById("agentTrace");

let ws = null;
let micStream = null;
let captureCtx = null;
let processor = null;
let sourceNode = null;
let playbackCtx = null;
let nextPlayTime = 0;
let activeSources = [];
let inCall = false;

callBtn.addEventListener("click", () => (inCall ? stopCall() : startCall()));

function setStatus(text, live) {
  statusText.textContent = text;
  dot.classList.toggle("live", !!live);
}

function addMessage(role, text) {
  if (emptyHint) emptyHint.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// --------------------------------------------------------------------------- //
// Start / stop
// --------------------------------------------------------------------------- //
async function startCall() {
  callBtn.disabled = true;
  setStatus("Connecting…");

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    setStatus("Microphone permission denied");
    callBtn.disabled = false;
    return;
  }

  playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: OUTPUT_RATE });
  nextPlayTime = 0;

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    inCall = true;
    callBtn.disabled = false;
    callBtn.textContent = "⏹ End call";
    callBtn.classList.add("stop");
    setStatus("Connected — listening", true);
    startMicCapture();
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleEvent(JSON.parse(event.data));
    } else {
      playPcm16(new Int16Array(event.data));
    }
  };

  ws.onclose = () => stopCall();
  ws.onerror = () => setStatus("Connection error");
}

function stopCall() {
  inCall = false;
  callBtn.textContent = "📞 Start call";
  callBtn.classList.remove("stop");
  callBtn.disabled = false;
  setStatus("Idle");
  flushPlayback();

  if (processor) { processor.disconnect(); processor = null; }
  if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
  if (captureCtx) { captureCtx.close(); captureCtx = null; }
  if (playbackCtx) { playbackCtx.close(); playbackCtx = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  ws = null;
}

// --------------------------------------------------------------------------- //
// Backend events
// --------------------------------------------------------------------------- //
function handleEvent(evt) {
  if (evt.type && evt.type.startsWith("graph_") || evt.type === "agent_step") {
    console.debug("[agent-panel]", evt.type, evt.kind || "", evt.name || "");
  }
  switch (evt.type) {
    case "transcript":
      if (evt.content) addMessage(evt.role === "user" ? "user" : "assistant", evt.content);
      break;
    case "function_call":
      addMessage("tool", `🧠 LangGraph brain · ${evt.arguments?.question ?? evt.name}`);
      break;
    case "graph_start":
      agentReset();
      break;
    case "agent_step":
      agentStep(evt);
      break;
    case "graph_done":
      agentDone();
      break;
    case "user_started_speaking":
      flushPlayback(); // barge-in
      break;
    case "status":
      if (evt.event === "Error") setStatus("Deepgram error — see console");
      console.log("[status]", evt.event, evt.detail);
      break;
    default:
      break;
  }
}

// --------------------------------------------------------------------------- //
// Live multi-step agent panel
// --------------------------------------------------------------------------- //
// The backend streams the agent's reasoning loop: `agent_step` events with
// kind "tool_call" (the agent chose a tool) and "tool_result" (a tool returned).
// We light up the agent hub, the tool it calls, and append each step to a trace.
const AGENT_TOOL_IDS = ["search", "inventory", "calculator", "tradein", "booking"];
let agentStepCount = 0;
let agentCallCounts = {};

function gEl(id) { return document.getElementById(id); }
function gToggle(id, cls, on) { const el = gEl(id); if (el) el.classList.toggle(cls, on); }

function agentReset() {
  ["agent", "start", "end", ...AGENT_TOOL_IDS].forEach((n) => {
    gToggle(`gn-${n}`, "done", false);
    gToggle(`gn-${n}`, "current", false);
  });
  document.querySelectorAll(".gedge").forEach((e) => e.classList.remove("done", "active"));
  AGENT_TOOL_IDS.forEach((t) => {
    const b = gEl(`badge-${t}`);
    if (b) { b.classList.remove("show"); b.querySelector("text").textContent = ""; }
  });
  agentStepCount = 0;
  agentCallCounts = {};
  gToggle("gn-start", "done", true);
  gToggle("edge-start-agent", "done", true);
  gToggle("gn-agent", "done", true);
  gToggle("gn-agent", "current", true); // agent "thinking"
  if (graphPathEl) graphPathEl.innerHTML = "<b>▶</b> agent reasoning…";
  if (agentTraceEl) agentTraceEl.innerHTML = "";
}

function agentAddTrace(html, cls) {
  if (!agentTraceEl) return;
  const div = document.createElement("div");
  div.className = "step" + (cls ? " " + cls : "");
  div.innerHTML = html;
  agentTraceEl.appendChild(div);
  agentTraceEl.scrollTop = agentTraceEl.scrollHeight;
}

function agentStep(evt) {
  if (evt.kind === "tool_call") {
    agentStepCount += 1;
    const ui = evt.ui; // tool UI id (search/inventory/…)
    // pulse the edge + light the tool node
    gToggle(`edge-agent-${ui}`, "active", true);
    setTimeout(() => { gToggle(`edge-agent-${ui}`, "active", false); gToggle(`edge-agent-${ui}`, "done", true); }, 700);
    gToggle(`gn-${ui}`, "done", true);
    gToggle(`gn-${ui}`, "current", true);
    setTimeout(() => gToggle(`gn-${ui}`, "current", false), 900);
    // badge = order of first call (keep first number, count repeats)
    const b = gEl(`badge-${ui}`);
    if (b) {
      agentCallCounts[ui] = (agentCallCounts[ui] || 0) + 1;
      const t = b.querySelector("text");
      t.textContent = agentCallCounts[ui] > 1 ? `×${agentCallCounts[ui]}` : String(agentStepCount);
      b.classList.add("show");
    }
    agentAddTrace(
      `<span class="tool">${agentStepCount}. 🔧 ${evt.name}</span> ` +
      `<span class="args">${escapeHtml(evt.args || "")}</span>`
    );
    if (graphPathEl) graphPathEl.innerHTML = `<b>step ${agentStepCount}</b> · ${evt.name}`;
  } else if (evt.kind === "tool_result") {
    // attach the result to the most recent step row
    const rows = agentTraceEl ? agentTraceEl.querySelectorAll(".step") : [];
    if (rows.length) {
      const res = document.createElement("span");
      res.className = "res";
      res.textContent = "↳ " + (evt.result || "");
      rows[rows.length - 1].appendChild(res);
      agentTraceEl.scrollTop = agentTraceEl.scrollHeight;
    }
  }
}

function agentDone() {
  gToggle("gn-agent", "current", false);
  gToggle("edge-agent-end", "done", true);
  gToggle("gn-end", "done", true);
  const n = agentStepCount;
  if (graphPathEl) graphPathEl.innerHTML = `✓ <b>${n} tool ${n === 1 ? "call" : "calls"}</b> → answer spoken`;
  agentAddTrace("💬 <b>final answer composed &amp; spoken</b>", "final");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// --------------------------------------------------------------------------- //
// Mic capture -> 16 kHz PCM16 -> backend
// --------------------------------------------------------------------------- //
function startMicCapture() {
  captureCtx = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = captureCtx.createMediaStreamSource(micStream);

  // ScriptProcessorNode is deprecated but is the simplest cross-browser way to
  // grab raw PCM frames for a demo. (AudioWorklet is the modern replacement.)
  processor = captureCtx.createScriptProcessor(4096, 1, 1);
  const inputRate = captureCtx.sampleRate;

  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const float32 = e.inputBuffer.getChannelData(0);
    const pcm16 = downsampleToPcm16(float32, inputRate, TARGET_INPUT_RATE);
    ws.send(pcm16.buffer);
  };

  // Route through a muted gain node so the processor runs without echoing the mic.
  const mute = captureCtx.createGain();
  mute.gain.value = 0;
  sourceNode.connect(processor);
  processor.connect(mute);
  mute.connect(captureCtx.destination);
}

function downsampleToPcm16(float32, inRate, outRate) {
  const ratio = inRate / outRate;
  const outLength = Math.round(float32.length / ratio);
  const out = new Int16Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const idx = i * ratio;
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, float32.length - 1);
    const sample = float32[i0] + (float32[i1] - float32[i0]) * (idx - i0); // linear interp
    out[i] = Math.max(-1, Math.min(1, sample)) * 0x7fff;
  }
  return out;
}

// --------------------------------------------------------------------------- //
// Playback: 24 kHz PCM16 -> Web Audio, scheduled back-to-back
// --------------------------------------------------------------------------- //
function playPcm16(int16) {
  if (!playbackCtx) return;
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 0x8000;

  const buffer = playbackCtx.createBuffer(1, float32.length, OUTPUT_RATE);
  buffer.copyToChannel(float32, 0);

  const src = playbackCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(playbackCtx.destination);

  const now = playbackCtx.currentTime;
  if (nextPlayTime < now) nextPlayTime = now + 0.04;
  src.start(nextPlayTime);
  nextPlayTime += buffer.duration;

  activeSources.push(src);
  src.onended = () => (activeSources = activeSources.filter((s) => s !== src));
}

function flushPlayback() {
  activeSources.forEach((s) => { try { s.stop(); } catch (_) {} });
  activeSources = [];
  if (playbackCtx) nextPlayTime = playbackCtx.currentTime;
}
