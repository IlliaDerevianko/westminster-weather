"""Async client for the official GOV.UK Content API and Search API."""

from typing import Any

import httpx

from westminster_weather.config import settings


class GovUkContentClient:
    """Fetches real GOV.UK consultation, command paper, and policy paper metadata."""

    def __init__(self, base_url: str | None = None, timeout: float = 20.0) -> None:
        self.base_url = (base_url or settings.gov_uk_content_api_base_url).rstrip("/")
        self.timeout = timeout

    async def get_content(self, content_path: str) -> dict[str, Any]:
        """Fetch structured JSON for a GOV.UK path via https://www.gov.uk/api/content/<path>."""
        clean_path = content_path.lstrip("/")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/{clean_path}")
            response.raise_for_status()
            return response.json()
