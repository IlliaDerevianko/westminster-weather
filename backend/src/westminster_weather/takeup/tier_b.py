"""Tier B: ministers' commitments to legislate, fetched from Parliament's APIs.

Fetching only fills the local cache (``backend/data/cache/takeup/``); nothing is written to
the database here. Every response is cached, so re-runs skip finished work and an
interrupted run resumes where it stopped.

Sources:
  - Written ministerial statements (Nov 2014 onwards). The list endpoint truncates the text
    to ~250 characters, so each statement's full text is fetched separately.
  - Hansard spoken contributions containing a trigger phrase (server-side phrase search).
  - Hansard written statements containing a trigger phrase, for years before the written
    statements API (2006–2014).
  - Members API government posts, to check a Hansard speaker was a minister on the date.
"""

import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

from westminster_weather.takeup.triggers import SEARCH_PHRASES

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "takeup"
WMS_URL = "https://questions-statements-api.parliament.uk/api/writtenstatements/statements"
HANSARD_URL = "https://hansard-api.parliament.uk/search/contributions/{kind}.json"
MEMBER_BIOGRAPHY_URL = "https://members-api.parliament.uk/api/Members/{member_id}/Biography"
USER_AGENT = "WestminsterWeather/0.1 (OxHack 2026 student project)"
RETRY_STATUSES = {429, 500, 502, 503, 504}
PAGE_SIZE = 100
# Seconds between requests per host. Hansard allows about 1,000 requests per 5 minutes; the
# written statements API publishes no limit, and every 429 is logged so a too-fast pace shows.
HOST_INTERVALS = {
    "hansard-api.parliament.uk": 0.35,
    "questions-statements-api.parliament.uk": 0.05,
}
DEFAULT_INTERVAL = 0.1


class PoliteClient:
    """httpx client with retries and a minimum gap between requests to the same host."""

    def __init__(self, max_retries: int = 6) -> None:
        self.http = httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT})
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._next_slot: dict[str, float] = {}

    def _wait_turn(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(host, now))
            self._next_slot[host] = slot + HOST_INTERVALS.get(host, DEFAULT_INTERVAL)
        time.sleep(max(0.0, slot - now))

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        host = httpx.URL(url).host
        for attempt in range(self.max_retries + 1):
            self._wait_turn(host)
            try:
                response = self.http.get(url, params=params)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            else:
                if response.status_code not in RETRY_STATUSES:
                    response.raise_for_status()
                    return response.json()
                log.warning("HTTP %s from %s (attempt %d)", response.status_code, host, attempt + 1)
                if attempt == self.max_retries:
                    response.raise_for_status()
            time.sleep(min(2**attempt, 60))
        raise RuntimeError(f"unreachable: {url}")

    def close(self) -> None:
        self.http.close()


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _slug(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-")


def _months(start: date, end: date) -> Iterator[tuple[date, date]]:
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        yield max(date(y, m, 1), start), min(date(ny, nm, 1) - timedelta(days=1), end)
        y, m = ny, nm


def _is_finished(period_end: date, today: date) -> bool:
    """Periods that ended over a week ago will not change, so their cache is final."""
    return (today - period_end).days > 7


# ------------------------------------------------------------- written statements ----
def fetch_written_statements(
    client: PoliteClient, start: date, end: date, list_workers: int = 4, workers: int = 16
) -> int:
    """Cache every written ministerial statement in [start, end], with full text."""
    today = date.today()

    def list_month(month: tuple[date, date]) -> list[dict[str, Any]]:
        month_start, month_end = month
        path = CACHE_DIR / "wms" / "list" / f"{month_start:%Y-%m-%d}_{month_end:%Y-%m-%d}.json"
        if path.exists() and _is_finished(month_end, today):
            return _read(path)
        items, skip = [], 0
        while True:
            page = client.get_json(
                WMS_URL,
                {
                    "madeWhenFrom": month_start.isoformat(),
                    "madeWhenTo": month_end.isoformat(),
                    "skip": skip,
                    "take": PAGE_SIZE,
                },
            )
            batch = [r.get("value", r) for r in page.get("results", [])]
            items += batch
            skip += len(batch)
            if not batch or skip >= page.get("totalResults", 0):
                break
        _write(path, items)
        return items

    ids: list[int] = []
    months = list(_months(start, end))
    with ThreadPoolExecutor(max_workers=list_workers) as pool:
        for month, items in zip(months, pool.map(list_month, months), strict=True):
            ids += [int(v["id"]) for v in items]
            log.info("wms list %s: %d statements (%d so far)", month[0], len(items), len(ids))

    missing = [
        i for i in dict.fromkeys(ids) if not (CACHE_DIR / "wms" / "full" / f"{i}.json").exists()
    ]
    log.info("wms full text: %d cached, %d to fetch", len(set(ids)) - len(missing), len(missing))

    def fetch_one(statement_id: int) -> None:
        data = client.get_json(f"{WMS_URL}/{statement_id}")
        _write(CACHE_DIR / "wms" / "full" / f"{statement_id}.json", data.get("value", data))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch_one, missing):
            done += 1
            if done % 500 == 0 or done == len(missing):
                log.info("wms full text %d/%d", done, len(missing))
    return len(set(ids))


# ---------------------------------------------------------------------- Hansard ----
def fetch_hansard(client: PoliteClient, kind: str, start: date, end: date, workers: int = 2) -> int:
    """Cache Hansard contributions (kind 'Spoken' or 'Written') containing a trigger phrase,
    one file per phrase per year. Hansard responds slowly (~20 s) and returns HTTP 500 under
    heavier parallel load, so only a couple of searches run at once. A search that still fails
    after retries is logged and skipped (not cached), so the next run retries it."""
    today = date.today()
    failed: list[str] = []

    def fetch_one(phrase: str, y_start: date, y_end: date) -> int:
        path = CACHE_DIR / "hansard" / kind.lower() / _slug(phrase) / f"{y_start}_{y_end}.json"
        if path.exists() and _is_finished(y_end, today):
            return len(_read(path))
        items, skip = [], 0
        try:
            while True:
                page = client.get_json(
                    HANSARD_URL.format(kind=kind),
                    {
                        "queryParameters.searchTerm": f'"{phrase}"',
                        "queryParameters.startDate": y_start.isoformat(),
                        "queryParameters.endDate": y_end.isoformat(),
                        "queryParameters.take": PAGE_SIZE,
                        "queryParameters.skip": skip,
                    },
                )
                batch = page.get("Results", [])
                items += batch
                skip += len(batch)
                if not batch or skip >= page.get("TotalResultCount", 0):
                    break
        except (httpx.HTTPError, RuntimeError) as exc:
            failed.append(f"{phrase} {y_start}..{y_end}")
            log.error(
                "hansard %s '%s' %s..%s FAILED, will retry next run: %s",
                kind,
                phrase,
                y_start,
                y_end,
                exc,
            )
            return 0
        _write(path, items)
        return len(items)

    tasks = [
        (phrase, max(date(year, 1, 1), start), min(date(year, 12, 31), end))
        for phrase in SEARCH_PHRASES
        for year in range(start.year, end.year + 1)
    ]
    total = done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n in pool.map(lambda task: fetch_one(*task), tasks):
            total += n
            done += 1
            if done % 10 == 0 or done == len(tasks):
                log.info(
                    "hansard %s: %d/%d searches, %d contributions", kind, done, len(tasks), total
                )
    if failed:
        log.error("hansard %s: %d searches FAILED (rerun to retry): %s", kind, len(failed), failed)
    return total


def cached_hansard_member_ids() -> set[int]:
    ids = set()
    for path in (CACHE_DIR / "hansard").glob("*/*/*.json"):
        ids |= {int(r["MemberId"]) for r in _read(path) if r.get("MemberId")}
    return ids


# ----------------------------------------------------------------------- Members ----
def fetch_government_posts(client: PoliteClient, member_ids: set[int], workers: int = 3) -> int:
    """Cache each member's government posts (name, start, end) for the minister check."""
    missing = sorted(i for i in member_ids if not (CACHE_DIR / "members" / f"{i}.json").exists())
    log.info("members: %d cached, %d to fetch", len(member_ids) - len(missing), len(missing))

    def fetch_one(member_id: int) -> None:
        bio = client.get_json(MEMBER_BIOGRAPHY_URL.format(member_id=member_id))
        posts = (bio.get("value", bio) or {}).get("governmentPosts") or []
        _write(
            CACHE_DIR / "members" / f"{member_id}.json",
            [
                {"name": p.get("name"), "start": p.get("startDate"), "end": p.get("endDate")}
                for p in posts
            ],
        )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch_one, missing):
            done += 1
            if done % 200 == 0 or done == len(missing):
                log.info("members %d/%d", done, len(missing))
    return len(member_ids)


def fetch_all(start: date, end: date, sources: list[str]) -> None:
    """Run the requested fetches in order: wms, spoken, written, members."""
    client = PoliteClient()
    try:
        if "wms" in sources:
            n = fetch_written_statements(client, max(start, date(2014, 11, 1)), end)
            log.info("DONE wms: %d statements", n)
        if "spoken" in sources:
            log.info("DONE hansard spoken: %d", fetch_hansard(client, "Spoken", start, end))
        if "written" in sources:
            log.info("DONE hansard written: %d", fetch_hansard(client, "Written", start, end))
        if "members" in sources:
            log.info(
                "DONE members: %d", fetch_government_posts(client, cached_hansard_member_ids())
            )
    finally:
        client.close()
