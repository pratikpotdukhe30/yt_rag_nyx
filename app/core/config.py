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
    openai_api_key: SecretStr
    openai_model: str = "gpt-4o-mini"
    hf_api_token: SecretStr
    hf_fallback_model : str = "Qwen/Qwen2.5-7B-Instruct"


    llm_timeout_seconds : float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, gt=0)

    #Embeddings
    embedding_model : str = "intfloat/multilingual-e5-base"
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    #Vector store
    chunk_size: int = Field(default=1024, gt=0)
    chunk_overlap: int = Field(default=200, gt=0)
    top_k: int = Field(default=4, gt=0)
    similarity_threshold: float = Field(default=0.80, ge=0.0, lt=1.0)

    #languages
    supported_languages: List[str] = ["en", "it"]

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap > self.chunk_size:
            raise ValidationError("Chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()



