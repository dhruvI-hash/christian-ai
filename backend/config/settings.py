"""
Configuration settings for ChristianAI backend.
Loads all environment variables using pydantic-settings.
"""

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = ""

    # --- Google Gemini ---
    gemini_api_key: str = ""

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"

    # --- OpenRouter ---
    openrouter_api_key: str = ""

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # --- ChromaDB ---
    chroma_persist_dir: str = "./chroma_data"

    # --- Redis ---
    redis_url: Optional[str] = None

    # --- Defaults ---
    default_llm_provider: Literal["openai", "gemini", "ollama", "openrouter"] = "openai"
    default_llm_model: str = "gpt-4o"
    default_vector_db: Literal["qdrant", "chroma"] = "qdrant"
    default_collection: str = "bible"

    # --- Server ---
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"

    # --- Safety ---
    enable_guardrail: bool = True
    enable_post_safety: bool = True

    # --- Observability ---
    log_level: str = "INFO"

    # --- RAG ---
    chunk_size: int = 512
    chunk_overlap: int = 128
    hybrid_search_strategy: Literal["rrf", "alpha"] = "rrf"
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


# Singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
