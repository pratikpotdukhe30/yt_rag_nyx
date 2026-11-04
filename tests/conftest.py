import os

# Set BEFORE any app module reads settings. Real env vars beat .env,
# so tests can never pick up your real keys or spend money.
os.environ.update({
    "OPENAI_API_KEY": "test-openai-key",
    "GROQ_API_KEY": "test-groq-key",
    "LANGSMITH_TRACING": "false",
    "ENV": "dev",
    "LOG_LEVEL": "WARNING",
})

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()   # drop any settings cached before the env above was set