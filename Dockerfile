# syntax=docker/dockerfile:1

# ---------- Stage 1: builder (has uv, installs deps) ----------
FROM python:3.12-slim AS builder

# Pin this tag to your local uv version (`uv --version`) for reproducible builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependency layer FIRST → cached until pyproject/uv.lock change
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------- Stage 2: runtime (no uv, no build tools) ----------
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app ./app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface

RUN mkdir -p /app/chroma_db /home/appuser/.cache/huggingface \
 && chown -R appuser:appuser /app/chroma_db /home/appuser

USER appuser
EXPOSE 8000

# 0.0.0.0 = reachable from outside the container. One worker: in-memory jobs + embedded Chroma.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]