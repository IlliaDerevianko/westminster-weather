"""Bill text helpers shared by Tier A and Tier B linking: long titles and first readings.

The ``bills`` table has no long title, so long titles come from the Bills API
(``GET /Bills/{id}`` → ``longTitle``), cached as one JSON file per bill under
``backend/data/cache/takeup/bills/``. Nothing is written to the database here.
"""

import json
import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from westminster_weather.takeup.tier_b import CACHE_DIR, RETRY_STATUSES, USER_AGENT

log = logging.getLogger(__name__)

BILLS_CACHE_DIR = CACHE_DIR / "bills"
BILL_URL = "https://bills-api.parliament.uk/api/v1/Bills/{bill_id}"
# Bills API stage IDs for first reading: 6 = Commons 1st reading, 1 = Lords 1st reading.
FIRST_READING_STAGE_IDS = (6, 1)
MAX_WORKERS = 5


def get_json(client: httpx.Client, url: str, max_retries: int = 5) -> Any:
    """GET with retries on 429 / 5xx and transport errors (exponential back-off)."""
    for attempt in range(max_retries + 1):
        try:
            response = client.get(url)
        except httpx.TransportError:
            if attempt == max_retries:
                raise
        else:
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response.json()
            if attempt == max_retries:
                response.raise_for_status()
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"unreachable: {url}")


def _cache_path(bill_id: int, cache_dir: Path) -> Path:
    return cache_dir / f"{bill_id}.json"


def _read_cached(bill_id: int, cache_dir: Path) -> dict[str, Any] | None:
    path = _cache_path(bill_id, cache_dir)
    return json.loads(path.read_text()) if path.exists() else None


def bill_long_titles(
    bill_ids: Iterable[int], cache_dir: Path = BILLS_CACHE_DIR, fetch: bool = True
) -> dict[int, str]:
    """Long title of each bill ('A Bill to ...'), from the cache or the Bills API.

    Each bill's API record is cached whole, so other fields can be reused later. With
    ``fetch=False`` only cached bills are returned (offline use). Bills that cannot be fetched
    are logged and left out.
    """
    ids = sorted({int(i) for i in bill_ids})
    missing = [i for i in ids if not _cache_path(i, cache_dir).exists()]
    if missing and fetch:
        log.info("bill long titles: %d cached, %d to fetch", len(ids) - len(missing), len(missing))
        cache_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT}) as client:

            def fetch_one(bill_id: int) -> None:
                try:
                    data = get_json(client, BILL_URL.format(bill_id=bill_id))
                except httpx.HTTPError as exc:
                    log.warning("bill %d: long title not fetched (%s)", bill_id, exc)
                    return
                path = _cache_path(bill_id, cache_dir)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data))
                tmp.replace(path)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                list(pool.map(fetch_one, missing))

    titles: dict[int, str] = {}
    for bill_id in ids:
        data = _read_cached(bill_id, cache_dir)
        if data is not None:
            titles[bill_id] = (data.get("longTitle") or "").strip()
    return titles


def first_readings(stage_sittings: pd.DataFrame) -> pd.Series:
    """First reading date per bill: earliest sitting of Bills API stage 6 or 1.

    Returns a Series indexed by ``bill_id`` (dates as ``datetime.date``), sorted by bill_id.
    """
    rows = stage_sittings[stage_sittings["api_stage_id"].isin(FIRST_READING_STAGE_IDS)]
    dates = pd.to_datetime(rows["event_date"]).groupby(rows["bill_id"].astype(int)).min()
    out = dates.dt.date
    out.index.name = "bill_id"
    out.name = "first_reading"
    return out.sort_index()
