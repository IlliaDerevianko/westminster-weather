"""Settings must never expose database URLs (which carry passwords) in repr()."""

from westminster_weather.config import Settings


def test_database_urls_are_hidden_from_repr() -> None:
    secret = "postgresql://user:s3cret-pw@example.invalid/db"
    settings = Settings(
        database_url=secret,
        database_url_unpooled=secret,
        database_url_takeup=secret,
        database_url_takeup_unpooled=secret,
    )
    assert "s3cret-pw" not in repr(settings)
    assert "s3cret-pw" not in str(settings)
    assert settings.database_url_takeup == secret
