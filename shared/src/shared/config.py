"""Typed settings, read from environment variables and .env files.

Precedence (highest first): real environment variables, then the project's own `.env`,
then the repo-root `.env`, then the defaults below. Names are case-insensitive, so
`DATABASE_URL` in the environment sets `database_url` here.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Later files win: a project's .env overrides the repo-root ../.env.
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    # 127.0.0.1, not "localhost": on Windows, localhost tries IPv6 first and these services
    # only listen on IPv4, which added seconds to every connection (see 00-foundations M6).
    ollama_base_url: str = "http://127.0.0.1:11434"
    # Port 5433: the Docker Postgres from 00-foundations/docker-compose.yml.
    database_url: str = "postgresql://postgres:postgres@127.0.0.1:5433/ai_systems"


@lru_cache
def get_settings() -> Settings:
    """Settings are read once and reused. Tests can call get_settings.cache_clear()."""
    return Settings()
