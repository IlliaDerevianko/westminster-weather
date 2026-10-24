"""Application settings and external API configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Westminster Weather API"
    environment: str = "development"
    debug: bool = True

    # Database configuration
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/westminster_weather"
    )

    # UK Public Legislative Data Sources
    parliament_bills_api_base_url: str = "https://bills-api.parliament.uk/api/v1"
    legislation_gov_uk_base_url: str = "https://www.legislation.gov.uk"
    gov_uk_content_api_base_url: str = "https://www.gov.uk/api/content"

    # Government Bill type ID in UK Parliament Bills API (1 = Government Bill)
    government_bill_type_id: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
