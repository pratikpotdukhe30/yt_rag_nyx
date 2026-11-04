import { ApiError, ask, ingest, waitForJob } from "./api.js";

const VIDEO_ID_RE = /^[A-Za-z0-9_-]{11}$/;   // same rule as the backend schema
const TIMESTAMP_RE = /\[(?:(\d+):)?(\d{1,2}):(\d{2})\]/g;

const els = {
  status: document.getElementById("status"),
  messages: document.getElementById("messages"),
  form: document.getElementById("form"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  lang: document.getElementById("lang"),
};
const state = { tabId: null, videoId: null };

// One place mapping backend error codes → user-friendly text
const FRIENDLY = {
  guardrail_blocked: "This question can't be processed. Try rephrasing it.",
  rate_limited: "Too many questions. Wait a moment and try again.",
  transcript_not_found: "This video has no English or Italian captions.",
  video_not_indexed: "This video isn't ready yet.",
  upstream_error: "The AI service is having trouble. Try again shortly.",
  service_unavailable: "Service temporarily unavailable.",
  network_error: "Can't reach the server. Is it running?",
  timeout: "Indexing is taking too long. Try again later.",
};

function friendly(err) {
  if (!(err instanceof ApiError)) return "Unexpected error.";
  const msg = FRIENDLY[err.code] ?? err.message;
  return err.requestId ? `${msg} (ref ${err.requestId.slice(0, 8)})` : msg;
}

function extractVideoId(url) {
  try {
    const u = new URL(url);
    if (u.hostname.endsWith("youtube.com")) {
      if (u.pathname === "/watch") return u.searchParams.get("v");
      const m = u.pathname.match(/^\/(shorts|live)\/([^/?]+)/);
      if (m) return m[2];
    }
    if (u.hostname === "youtu.be") return u.pathname.slice(1);
  } catch { /* not a URL */ }
  return null;
}

// SECURITY: LLM/transcript text is untrusted → never innerHTML. Text nodes + real <a> elements only.
function renderAnswer(container, text) {
  container.replaceChildren();
  let last = 0;
  for (const m of text.matchAll(TIMESTAMP_RE)) {
    container.append(document.createTextNode(text.slice(last, m.index)));
    const seconds = Number(m[1] ?? 0) * 3600 + Number(m[2]) * 60 + Number(m[3]);
    const link = document.createElement("a");
    link.className = "ts";
    link.href = "#";
    link.textContent = m[0];
    link.addEventListener("click", (e) => { e.preventDefault(); seek(seconds); });
    container.append(link);
    last = m.index + m[0].length;
  }
  container.append(document.createTextNode(text.slice(last)));
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  renderAnswer(div, text);
  els.messages.append(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

// Jump the YouTube player to a timestamp without reloading the page
async function seek(seconds) {
  if (state.tabId == null) return;
  await chrome.scripting.executeScript({
    target: { tabId: state.tabId },
    func: (s) => {
      const v = document.querySelector("video");
      if (v) { v.currentTime = s; v.play(); }
    },
    args: [seconds],
  });
}

function setStatus(text) { els.status.textContent = text; }
function setReady(ready) { els.input.disabled = els.send.disabled = !ready; }

async function loadActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const id = tab?.url ? extractVideoId(tab.url) : null;

  if (!id || !VIDEO_ID_RE.test(id)) {
    state.videoId = null;
    setReady(false);
    setStatus("Open a YouTube video to start.");
    return;
  }
  if (id === state.videoId) return;            // same video: nothing to do

  state.tabId = tab.id;
  state.videoId = id;
  els.messages.replaceChildren();
  setReady(false);
  setStatus("Preparing video…");

  try {
    // Idempotent backend: already-indexed videos complete almost instantly
    const job = await ingest(id, els.lang.value);
    const done = job.status === "completed"
      ? job
      : await waitForJob(job.job_id, (s) => setStatus(`Indexing video… (${s})`));

    if (state.videoId !== id) return;          // user switched video meanwhile → drop stale result
    setStatus(done.result?.already_indexed ? "Ready." : `Ready (${done.result.chunks} sections indexed).`);
    setReady(true);
    els.input.focus();
  } catch (e) {
    if (state.videoId === id) setStatus(friendly(e));
  }
}

els.form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const question = els.input.value.trim();
  if (!question || !state.videoId) return;

  addMessage("user", question);
  els.input.value = "";
  setReady(false);
  const pending = addMessage("bot", "…");

  try {
    const r = await ask(state.videoId, question, els.lang.value);
    renderAnswer(pending, r.answer);
    if (!r.covered) pending.classList.add("muted");
  } catch (e) {
    pending.textContent = friendly(e);
    pending.classList.add("error");
  } finally {
    setReady(true);
    els.input.focus();
  }
});

// Enter sends, Shift+Enter makes a new line
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); els.form.requestSubmit(); }
});

// Default answer language from the browser
els.lang.value = navigator.language.startsWith("it") ? "it" : "en";

// YouTube is a single-page app: video changes = URL changes without a page load
chrome.tabs.onActivated.addListener(loadActiveTab);
chrome.tabs.onUpdated.addListener((_tabId, info) => { if (info.url) loadActiveTab(); });

loadActiveTab();