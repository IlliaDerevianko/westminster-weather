"""Download, cache and flatten UK Parliament Bills API data (BUILD_PLAN task 1).

Raw JSON is cached per bill in ``data/cache/bills/{bill_id}.json``. A bill is re-downloaded
only when its ``lastUpdate`` in the list endpoint differs from the cached copy, so re-running
``fetch`` is cheap. The flattened tables follow DATA_CONTRACT.md §3.1 (``bills``) and §3.2
(``stage_sittings``).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from westminster_weather.config import settings
from westminster_weather.data.paths import CACHE_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

MAX_PARALLEL = 5
PAGE_SIZE = 250
MAX_RETRIES = 5

# General elections in the data period (MODEL_SPEC §4.2). Used to tell a session that ended
# by dissolution from one that ended by prorogation.
ELECTION_DATES: tuple[date, ...] = (
    date(2010, 5, 6),
    date(2015, 5, 7),
    date(2017, 6, 8),
    date(2019, 12, 12),
    date(2024, 7, 4),
)

STAGE_ROYAL_ASSENT = 11
STAGE_COMMONS_FIRST_READING = 6
STAGE_LORDS_FIRST_READING = 1
STAGE_COMMONS_PUBLIC_BILL_COMMITTEE = 8
STAGE_COMMITTEE_OF_WHOLE_HOUSE = 49
CARRY_OVER_STAGES = (18, 19)


# --------------------------------------------------------------------------- HTTP


class BillsApiDownloader:
    """Synchronous, polite batch downloader for bills-api.parliament.uk."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.client = httpx.Client(
            base_url=(base_url or settings.parliament_bills_api_base_url).rstrip("/"),
            timeout=timeout,
            headers={"User-Agent": "westminster-weather/0.1 (research)"},
            limits=httpx.Limits(max_connections=MAX_PARALLEL),
        )

    def close(self) -> None:
        self.client.close()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET with retries and exponential backoff on network errors, 429 and 5xx."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.get(path, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if not retryable or attempt == MAX_RETRIES - 1:
                    raise
                wait = 2**attempt
                logger.warning("GET %s failed (%s); retrying in %ss", path, exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def list_bills(self, bill_type: int | None) -> list[dict[str, Any]]:
        """All bills of one type (or of every type when ``bill_type`` is None)."""
        items: list[dict[str, Any]] = []
        while True:
            params: dict[str, Any] = {"Take": PAGE_SIZE, "Skip": len(items)}
            if bill_type is not None:
                params["BillType"] = bill_type
            page = self.get("/Bills", params)
            items.extend(page["items"])
            if not page["items"] or len(items) >= page["totalResults"]:
                return items

    def bill_detail(self, bill_id: int) -> dict[str, Any]:
        return self.get(f"/Bills/{bill_id}")

    def bill_stages(self, bill_id: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        while True:
            page = self.get(f"/Bills/{bill_id}/Stages", {"Take": 100, "Skip": len(items)})
            items.extend(page["items"])
            if not page["items"] or len(items) >= page["totalResults"]:
                return items


# --------------------------------------------------------------------------- cache


def _cache_path(cache_dir: Path, bill_id: int) -> Path:
    return cache_dir / "bills" / f"{bill_id}.json"


def _read_cache(cache_dir: Path, bill_id: int) -> dict[str, Any] | None:
    path = _cache_path(cache_dir, bill_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _needs_download(cached: dict[str, Any] | None, list_item: dict[str, Any], detail: bool) -> bool:
    if cached is None:
        return True
    if cached["list_item"].get("lastUpdate") != list_item.get("lastUpdate"):
        return True
    return detail and cached.get("detail") is None


def fetch_all(
    cache_dir: Path = CACHE_DIR,
    include_all_types: bool = True,
    downloader: BillsApiDownloader | None = None,
) -> dict[str, int]:
    """Download government bills (list + detail + stages) and, optionally, the stages of
    every other bill type (used only for the sitting-day calendar).

    Returns counts of bills listed and bills (re-)downloaded.
    """
    own = downloader is None
    dl = downloader or BillsApiDownloader()
    try:
        gov_type = settings.government_bill_type_id
        logger.info("Listing bills")
        listed = dl.list_bills(None if include_all_types else gov_type)
        _write_json(cache_dir / "reference" / "stages.json", dl.get("/Stages", {"take": 100}))
        _write_json(cache_dir / "reference" / "bill_types.json", dl.get("/BillTypes"))
        _write_json(cache_dir / "lists" / "bills.json", listed)

        def work(item: dict[str, Any]) -> bool:
            bill_id = item["billId"]
            is_gov = item["billTypeId"] == gov_type
            cached = _read_cache(cache_dir, bill_id)
            if not _needs_download(cached, item, detail=is_gov):
                return False
            payload = {
                "fetched_at": datetime.now(UTC).isoformat(),
                "list_item": item,
                "detail": dl.bill_detail(bill_id) if is_gov else None,
                "stages": dl.bill_stages(bill_id),
            }
            _write_json(_cache_path(cache_dir, bill_id), payload)
            return True

        downloaded = 0
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            for i, did_download in enumerate(pool.map(work, listed), start=1):
                downloaded += did_download
                if i % 250 == 0:
                    logger.info("  %d / %d checked, %d downloaded", i, len(listed), downloaded)
        n_gov = sum(item["billTypeId"] == gov_type for item in listed)
        return {"listed": len(listed), "government": n_gov, "downloaded": downloaded}
    finally:
        if own:
            dl.close()


def load_cached_bills(cache_dir: Path = CACHE_DIR) -> list[dict[str, Any]]:
    """All cached bill payloads that appear in the most recent list."""
    listed = json.loads((cache_dir / "lists" / "bills.json").read_text())
    payloads = []
    for item in listed:
        cached = _read_cache(cache_dir, item["billId"])
        if cached is not None:
            payloads.append(cached)
    return payloads


# --------------------------------------------------------------------------- flatten


def _to_date(value: str | None) -> date | None:
    return datetime.fromisoformat(value).date() if value else None


def flatten_stage_sittings(payloads: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """One row per stage sitting (DATA_CONTRACT §3.2), for every bill in ``payloads``."""
    rows = []
    for payload in payloads:
        item = payload["list_item"]
        for stage in payload["stages"]:
            for sitting in stage["stageSittings"]:
                rows.append(
                    {
                        "sitting_id": sitting["id"],
                        "bill_id": item["billId"],
                        "bill_type_id": item["billTypeId"],
                        "api_stage_id": stage["stageId"],
                        "bill_stage_id": stage["id"],
                        "house": stage["house"],
                        "session_id": stage["sessionId"],
                        "sort_order": stage["sortOrder"],
                        "event_date": _to_date(sitting["date"]),
                        "ingested_at": pd.Timestamp(payload["fetched_at"]),
                    }
                )
    columns = [
        "sitting_id", "bill_id", "bill_type_id", "api_stage_id", "bill_stage_id", "house",
        "session_id", "sort_order", "event_date", "ingested_at",
    ]  # fmt: skip
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(["bill_id", "event_date", "sort_order"]).reset_index(drop=True)


def _best_cut(earlier: list[date], later: list[date]) -> date:
    """Boundary date between two consecutive sessions that misplaces the fewest sittings.

    A handful of sittings in the API carry dates outside their session (e.g. a 2016 date
    filed under the 2014-15 session), so plain min/max would make sessions overlap."""
    candidates = sorted(set(earlier) | set(later))
    best, best_errors = candidates[0], None
    for cut in candidates:
        errors = sum(d > cut for d in earlier) + sum(d <= cut for d in later)
        if best_errors is None or errors < best_errors:
            best, best_errors = cut, errors
    return best


def session_bounds(all_sittings: pd.DataFrame) -> pd.DataFrame:
    """Start and end date per session and how it ended (``'prorogation'`` or
    ``'dissolution'``), estimated from sitting dates; corrections live in
    ``data/calendar.py``. The latest session is current (``end_date`` and ``end_type`` NULL).

    A session ended by dissolution if a general election falls between its last sitting and
    the next session's first sitting.
    """
    dates = all_sittings.dropna(subset=["event_date"]).groupby("session_id")["event_date"]
    by_session = {sid: sorted(g.unique()) for sid, g in dates}
    ids = sorted(by_session)
    starts = {ids[0]: by_session[ids[0]][0]}
    ends: dict[int, date | None] = {ids[-1]: None}
    for prev, nxt in zip(ids, ids[1:], strict=False):
        cut = _best_cut(by_session[prev], by_session[nxt])
        ends[prev] = max((d for d in by_session[prev] if d <= cut), default=cut)
        starts[nxt] = min((d for d in by_session[nxt] if d > cut), default=cut)

    rows = []
    for sid in ids:
        end = ends[sid]
        if end is None:
            end_type = None
        else:
            next_start = min(starts[s] for s in ids if s > sid)
            dissolved = any(end <= e < next_start for e in ELECTION_DATES)
            end_type = "dissolution" if dissolved else "prorogation"
        rows.append(
            {"session_id": sid, "start_date": starts[sid], "end_date": end, "end_type": end_type}
        )
    return pd.DataFrame(rows)


def _department(detail: dict[str, Any]) -> str | None:
    sponsors = sorted(detail.get("sponsors") or [], key=lambda s: s.get("sortOrder") or 0)
    for sponsor in sponsors:
        org = sponsor.get("organisation")
        if org and org.get("name"):
            return org["name"]
    return None


def flatten_bills(
    payloads: Iterable[dict[str, Any]],
    sittings: pd.DataFrame,
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    """One row per government bill (DATA_CONTRACT §3.1) plus a few raw API flags
    (``is_act``, ``is_defeated``, ``withdrawn_date``, ``included_session_ids``) kept for
    labelling in later tasks."""
    by_bill = {bid: g for bid, g in sittings.groupby("bill_id")}
    session_end = dict(zip(sessions["session_id"], sessions["end_date"], strict=True))
    session_end_type = dict(zip(sessions["session_id"], sessions["end_type"], strict=True))
    current_session = sessions["session_id"].max()

    rows = []
    for payload in payloads:
        item, detail = payload["list_item"], payload["detail"] or {}
        bill_id = item["billId"]
        bill_sittings = by_bill.get(bill_id, sittings.iloc[0:0])
        first_dates = bill_sittings.groupby("api_stage_id")["event_date"].min().to_dict()

        first_reading_stage = (
            STAGE_LORDS_FIRST_READING
            if item["originatingHouse"] == "Lords"
            else STAGE_COMMONS_FIRST_READING
        )
        introduced = first_dates.get(first_reading_stage) or (
            bill_sittings["event_date"].min() if len(bill_sittings) else None
        )

        if STAGE_COMMITTEE_OF_WHOLE_HOUSE in first_dates:
            committee_type = "whole_house"
        elif STAGE_COMMONS_PUBLIC_BILL_COMMITTEE in first_dates:
            committee_type = "public_bill_committee"
        else:
            committee_type = None

        carry_dates = [first_dates[s] for s in CARRY_OVER_STAGES if s in first_dates]
        last_session = max(item["includedSessionIds"] or [item["introducedSessionId"]])
        withdrawn = _to_date(item["billWithdrawn"])

        if item["isAct"]:
            outcome, outcome_date = "royal_assent", first_dates.get(STAGE_ROYAL_ASSENT)
        elif withdrawn is not None:
            outcome, outcome_date = "withdrawn", withdrawn
        elif item["isDefeated"]:
            last = bill_sittings["event_date"].max() if len(bill_sittings) else None
            outcome, outcome_date = "defeated", last
        elif last_session == current_session:
            outcome, outcome_date = "in_progress", None
        elif session_end_type.get(last_session) == "dissolution":
            outcome, outcome_date = "fell_dissolution", session_end.get(last_session)
        else:
            outcome, outcome_date = "fell_session_end", session_end.get(last_session)

        rows.append(
            {
                "bill_id": bill_id,
                "short_title": item["shortTitle"],
                "originating_house": item["originatingHouse"],
                "introduced_session_id": item["introducedSessionId"],
                "introduced_date": introduced,
                "department": _department(detail),
                "kings_speech_flag": None,  # Tier 2, not available from the Bills API
                "size_clauses": None,  # Tier 2
                "size_pages": None,  # Tier 2
                "commons_committee_type": committee_type,
                "carry_over_date": min(carry_dates) if carry_dates else None,
                "outcome": outcome,
                "outcome_date": outcome_date,
                "last_api_update": pd.Timestamp(item["lastUpdate"]),
                "ingested_at": pd.Timestamp(payload["fetched_at"]),
                "is_act": item["isAct"],
                "is_defeated": item["isDefeated"],
                "withdrawn_date": withdrawn,
                "included_session_ids": list(item["includedSessionIds"] or []),
            }
        )
    df = pd.DataFrame(rows)
    for col in ("kings_speech_flag", "size_clauses", "size_pages"):
        df[col] = df[col].astype("object")
    return df.sort_values("bill_id").reset_index(drop=True)


def build_tables(
    cache_dir: Path = CACHE_DIR, processed_dir: Path = PROCESSED_DIR
) -> dict[str, pd.DataFrame]:
    """Flatten the cache into ``bills``, ``stage_sittings`` (government bills only) and
    ``all_stage_sittings`` (every bill type, for the calendar) parquet files."""
    gov_type = settings.government_bill_type_id
    payloads = load_cached_bills(cache_dir)
    all_sittings = flatten_stage_sittings(payloads)
    gov_payloads = [p for p in payloads if p["list_item"]["billTypeId"] == gov_type]
    gov_sittings = all_sittings[all_sittings["bill_type_id"] == gov_type].reset_index(drop=True)
    sessions = session_bounds(all_sittings)
    bills = flatten_bills(gov_payloads, gov_sittings, sessions)

    tables = {
        "bills": bills,
        "stage_sittings": gov_sittings,
        "all_stage_sittings": all_sittings,
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(processed_dir / f"{name}.parquet", index=False)
    return tables
