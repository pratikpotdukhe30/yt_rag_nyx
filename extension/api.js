// The ONLY file that knows HTTP. Change API_BASE to your https domain when you deploy.
export const API_BASE = "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(status, code, message, requestId = null, retryAfter = null) {
    super(message);
    this.status = status;          // HTTP status (0 = network failure)
    this.code = code;              // stable machine code from our backend contract
    this.requestId = requestId;    // for support: links to server logs + LangSmith trace
    this.retryAfter = retryAfter;  // seconds, from the Retry-After header on 429
  }
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError(0, "network_error", "Can't reach the server.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const e = body.error ?? {};
    throw new ApiError(
      res.status,
      e.code ?? "unknown_error",
      e.message ?? res.statusText,
      e.request_id ?? null,
      Number(res.headers.get("Retry-After")) || null,
    );
  }
  return body;
}

export const ingest = (videoId, language) =>
  request("/v1/ingest", { method: "POST", body: JSON.stringify({ video_id: videoId, language }) });

export const getJob = (jobId) => request(`/v1/ingest/${jobId}`);

export const ask = (videoId, question, language) =>
  request("/v1/chat", { method: "POST", body: JSON.stringify({ video_id: videoId, question, language }) });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Polls a job with exponential backoff (1s → 1.5s → … max 5s). Respects 429 Retry-After.
export async function waitForJob(jobId, onStatus, { maxWaitMs = 180_000 } = {}) {
  let delay = 1000;
  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    let job;
    try {
      job = await getJob(jobId);
    } catch (e) {
      if (e.status === 429) { await sleep((e.retryAfter ?? 5) * 1000); continue; }
      throw e;
    }
    onStatus(job.status);
    if (job.status === "completed") return job;
    if (job.status === "failed") throw new ApiError(200, job.error.code, job.error.message);
    await sleep(delay);
    delay = Math.min(delay * 1.5, 5000);   // backoff: don't hammer the server while it works
  }
  throw new ApiError(0, "timeout", "Indexing is taking too long. Try again later.");
}