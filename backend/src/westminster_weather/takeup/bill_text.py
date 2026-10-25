"""Bill text helpers shared by Tier A and Tier B linking: long titles and first readings.

The ``bills`` table has no long title, so long titles come from the Bills API
(``GET /Bills/{id}`` → ``longTitle``), cached in one JSON file,
``backend/data/cache/takeup/bills.json`` (bill id → API record). Nothing is written to the
database here.
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

BILLS_CACHE_FILE = CACHE_DIR / "bills.json"
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


def _read_cache(cache_file: Path) -> dict[int, dict[str, Any]]:
    if not cache_file.exists():
        return {}
    return {int(k): v for k, v in json.loads(cache_file.read_text()).items()}


def _write_cache(cache_file: Path, records: dict[int, dict[str, Any]]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps({str(k): records[k] for k in sorted(records)}))
    tmp.replace(cache_file)


def bill_long_titles(
    bill_ids: Iterable[int], cache_file: Path = BILLS_CACHE_FILE, fetch: bool = True
) -> dict[int, str]:
    """Long title of each bill ('A Bill to ...'), from the cache or the Bills API.

    Each bill's API record is cached whole, so other fields can be reused later. With
    ``fetch=False`` only cached bills are returned (offline use). Bills that cannot be fetched
    are logged and left out.
    """
    ids = sorted({int(i) for i in bill_ids})
    records = _read_cache(cache_file)
    missing = [i for i in ids if i not in records]
    if missing and fetch:
        log.info("bill long titles: %d cached, %d to fetch", len(ids) - len(missing), len(missing))
        with httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT}) as client:

            def fetch_one(bill_id: int) -> tuple[int, dict[str, Any] | None]:
                try:
                    return bill_id, get_json(client, BILL_URL.format(bill_id=bill_id))
                except httpx.HTTPError as exc:
                    log.warning("bill %d: long title not fetched (%s)", bill_id, exc)
                    return bill_id, None

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                fetched = {b: d for b, d in pool.map(fetch_one, missing) if d is not None}
        if fetched:
            records.update(fetched)
            _write_cache(cache_file, records)

    return {
        bill_id: (records[bill_id].get("longTitle") or "").strip()
        for bill_id in ids
        if bill_id in records
    }


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
