"""Async client for the official UK Parliament Bills API (Government Bills only)."""

import asyncio
from types import TracebackType
from typing import Any

import httpx

from westminster_weather.config import settings

RETRY_STATUSES = {429, 500, 502, 503, 504}


class ParliamentBillsClient:
    """Fetches UK Government Bills and stage history from bills-api.parliament.uk.

    Use as ``async with ParliamentBillsClient() as client:`` to share one connection
    across many requests; outside a ``with`` block each call opens its own connection.
    """

    def __init__(
        self, base_url: str | None = None, timeout: float = 15.0, max_retries: int = 5
    ) -> None:
        self.base_url = (base_url or settings.parliament_bills_api_base_url).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ParliamentBillsClient":
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._http is not None:
            return await self._get_with_retries(self._http, path, params)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as http:
            return await self._get_with_retries(http, path, params)

    async def _get_with_retries(
        self, http: httpx.AsyncClient, path: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = await http.get(path, params=params)
                if response.status_code not in RETRY_STATUSES:
                    response.raise_for_status()
                    return response.json()
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            if attempt < self.max_retries:
                await asyncio.sleep(2**attempt)
        response.raise_for_status()
        return response.json()

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
        return await self._get("/Bills", params)

    async def list_all_government_bills(self, page_size: int = 100) -> list[dict[str, Any]]:
        bills: list[dict[str, Any]] = []
        while True:
            page = await self.list_government_bills(skip=len(bills), take=page_size)
            items = page.get("items", [])
            bills.extend(items)
            if not items or len(bills) >= page.get("totalResults", 0):
                return bills

    async def get_bill(self, bill_id: int) -> dict[str, Any]:
        return await self._get(f"/Bills/{bill_id}")

    async def get_bill_stages(self, bill_id: int, skip: int = 0, take: int = 20) -> dict[str, Any]:
        return await self._get(f"/Bills/{bill_id}/Stages", {"Skip": skip, "Take": take})

    async def get_all_bill_stages(self, bill_id: int, page_size: int = 100) -> list[dict[str, Any]]:
        """All stages of a bill; the API pages them (20 by default), so keep fetching."""
        stages: list[dict[str, Any]] = []
        while True:
            page = await self.get_bill_stages(bill_id, skip=len(stages), take=page_size)
            items = page.get("items", [])
            stages.extend(items)
            if not items or len(stages) >= page.get("totalResults", 0):
                return stages

    async def list_stage_types(self) -> list[dict[str, Any]]:
        """Every stage type the API knows (1st reading, Committee stage, Royal Assent, ...)."""
        return (await self._get("/Stages", {"Take": 100})).get("items", [])
