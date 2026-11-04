from functools import lru_cache
from typing import Literal, List

from pydantic import SecretStr, Field, model_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


    #App
    app_name: str = "youtube_chatbot_nyx"
    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    #LLM
    # ---------- Resilience ----------
    openai_base_url: str | None = None          # None = official API; lets us simulate outages in tests
    breaker_failure_threshold: int = Field(default=3, gt=0)
    breaker_recovery_seconds: float = Field(default=30.0, gt=0)
    openai_api_key: SecretStr
    openai_model: str = "gpt-4o-mini"
    #hf_api_token: SecretStr
    #hf_fallback_model : str = "openai/gpt-oss-20b"

    groq_api_key: SecretStr  # fallback provider (independent of OpenAI)
    groq_fallback_model: str = "openai/gpt-oss-20b"


    llm_timeout_seconds : float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, gt=0)

    #Embeddings
    #embedding_model : str = "intfloat/multilingual-e5-base"
    #embedding_device: Literal["cpu", "cuda"] = "cpu"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    embedding_query_prefix: str = ""  # e5 needed "query: ", bge-m3 needs none
    embedding_doc_prefix: str = ""  # e5 needed "passage:

    #Vector store
    chunk_size: int = Field(default=1024, gt=0)
    chunk_overlap: int = Field(default=200, gt=0)
    top_k: int = Field(default=4, gt=0)
    similarity_threshold: float = Field(default=0.43, ge=0.0, le=1.0)

    # ---------- Caching ----------
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=3600, gt=0)

    # ---------- Workers ----------
    ingest_max_workers: int = Field(default=2, gt=0)  # backpressure: max concurrent embeddings

    chroma_persist_dir: str = "./chroma_db"
    #languages
    supported_languages: List[str] = ["en", "it"]

    # ---------- Guardrails ----------
    prompt_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    prompt_guard_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    guard_fail_open: bool = True     # guard down → allow + log (availability over strictness)

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap > self.chunk_size:
            raise ValidationError("Chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()



