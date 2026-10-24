"""Application settings, Neon Lakebase Postgres URLs, and external API configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_ROOT_DIR = _BACKEND_DIR.parent


def _to_psycopg_url(url: str) -> str:
    """Convert standard postgres:// or postgresql:// URL to SQLAlchemy psycopg v3 URL."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


class Settings(BaseSettings):
    app_name: str = "Westminster Weather API"
    environment: str = "development"
    debug: bool = True

    # Neon Lakebase Postgres environment variables (pulled by `neon deploy` into .env.local)
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/westminster_weather"
    )
    database_url_unpooled: str | None = None
    neon_branch: str = "production"

    # Take-up model database (Neon branch `takeup-dev`); never falls back to DATABASE_URL
    takeup_database_url: str | None = None
    takeup_database_url_unpooled: str | None = None

    # UK Public Legislative Data Sources
    parliament_bills_api_base_url: str = "https://bills-api.parliament.uk/api/v1"
    legislation_gov_uk_base_url: str = "https://www.legislation.gov.uk"
    gov_uk_content_api_base_url: str = "https://www.gov.uk/api/content"

    # Government Bill type ID in UK Parliament Bills API (1 = Government Bill)
    government_bill_type_id: int = 1

    model_config = SettingsConfigDict(
        env_file=(
            str(_ROOT_DIR / ".env.local"),
            str(_ROOT_DIR / ".env"),
            str(_BACKEND_DIR / ".env.local"),
            str(_BACKEND_DIR / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def sqlalchemy_pooled_url(self) -> str:
        """Pooled connection URL (-pooler) for application query traffic."""
        return _to_psycopg_url(self.database_url)

    @property
    def sqlalchemy_direct_url(self) -> str:
        """Direct/unpooled connection URL for schema migrations and DDL."""
        return _to_psycopg_url(self.database_url_unpooled or self.database_url)

    @property
    def sqlalchemy_takeup_pooled_url(self) -> str | None:
        """Pooled take-up database URL (TAKEUP_DATABASE_URL), or None if not set."""
        return _to_psycopg_url(self.takeup_database_url) if self.takeup_database_url else None

    @property
    def sqlalchemy_takeup_direct_url(self) -> str | None:
        """Direct take-up database URL for DDL; falls back to the pooled take-up URL only."""
        url = self.takeup_database_url_unpooled or self.takeup_database_url
        return _to_psycopg_url(url) if url else None


settings = Settings()
