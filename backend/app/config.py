from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration sourced from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "animal-face-cbir"
    app_version: str = "0.1.0"

    database_url: str = Field(
        default="postgresql+asyncpg://app:app@postgres:5432/cbir",
        description="Async SQLAlchemy DSN.",
    )
    storage_root: str = Field(
        default="/app/storage",
        description="Filesystem root for original images and cached plots.",
    )
    cors_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed CORS origins.",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached accessor; the cache also lets tests override the constructor."""
    return Settings()
