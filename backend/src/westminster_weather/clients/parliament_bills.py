"""Async client for the official UK Parliament Bills API (Government Bills only)."""

from typing import Any

import httpx

from westminster_weather.config import settings


class ParliamentBillsClient:
    """Fetches UK Government Bills and stage history from bills-api.parliament.uk."""

    def __init__(self, base_url: str | None = None, timeout: float = 20.0) -> None:
        self.base_url = (base_url or settings.parliament_bills_api_base_url).rstrip("/")
        self.timeout = timeout

    async def list_government_bills(
        self, skip: int = 0, take: int = 20, current_house: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "BillType": settings.government_bill_type_id,
            "Skip": skip,
            "Take": take,
        }
        if current_house:
            params["CurrentHouse"] = current_house

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.get("/Bills", params=params)
            response.raise_for_status()
            return response.json()

    async def get_bill(self, bill_id: int) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.get(f"/Bills/{bill_id}")
            response.raise_for_status()
            return response.json()

    async def get_bill_stages(self, bill_id: int) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.get(f"/Bills/{bill_id}/Stages")
            response.raise_for_status()
            return response.json()
