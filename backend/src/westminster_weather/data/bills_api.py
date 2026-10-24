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
from westminster_weather.data import calendar
from westminster_weather.data.paths import CACHE_DIR, PROCESSED_DIR
from westminster_weather.data.stage_map import CARRY_OVER_STAGE_IDS
from westminster_weather.domain.stages import StopReason

logger = logging.getLogger(__name__)

MAX_PARALLEL = 5
PAGE_SIZE = 250
MAX_RETRIES = 5

STAGE_ROYAL_ASSENT = 11
STAGE_COMMONS_FIRST_READING = 6
STAGE_LORDS_FIRST_READING = 1
STAGE_COMMONS_PUBLIC_BILL_COMMITTEE = 8
STAGE_COMMITTEE_OF_WHOLE_HOUSE = 49

OUTCOME_STOP_REASON: dict[str, StopReason | None] = {
    "royal_assent": None,
    "in_progress": None,
    "fell_session_end": StopReason.SESSION_RUNS_OUT,
    "fell_dissolution": StopReason.ELECTION_CALLED,
    "withdrawn": StopReason.PAUSED_OR_WITHDRAWN,
    "defeated": StopReason.DEFEATED,
}


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


def split_scheduled(sittings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sittings into those that have happened and those scheduled for after the day
    they were downloaded.

    Scheduled sittings have no record of when they were announced, so they may only be used
    for live forecasts, never for training or backtests.
    """
    future = sittings["event_date"] > sittings["ingested_at"].dt.date
    return (
        sittings[~future].reset_index(drop=True),
        sittings[future].reset_index(drop=True),
    )


def _stop_reason_value(outcome: str) -> str | None:
    reason = OUTCOME_STOP_REASON[outcome]
    return reason.value if reason is not None else None


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
    """One row per government bill (DATA_CONTRACT §3.1) plus ``stop_reason`` (the
    ``StopReason`` matching ``outcome``) and a few raw API flags (``is_act``,
    ``is_defeated``, ``withdrawn_date``, ``included_session_ids``).

    A bill that is not an Act, withdrawn or defeated is ``in_progress`` only if it is in the
    current session, or had a carry-over motion in the session just before it and that
    session was prorogued (DATA_CONTRACT §8; a dissolution kills carried-over bills too).
    Otherwise it fell when its last session ended: ``fell_dissolution`` if that session
    ended by dissolution, else ``fell_session_end``.
    """
    by_bill = {bid: g for bid, g in sittings.groupby("bill_id")}
    session_end = dict(zip(sessions["session_id"], sessions["end_date"], strict=True))
    session_end_type = dict(zip(sessions["session_id"], sessions["end_type"], strict=True))
    current_session = sessions["session_id"].max()
    previous_session = sessions.loc[sessions["session_id"] < current_session, "session_id"].max()

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

        carry = bill_sittings[bill_sittings["api_stage_id"].isin(CARRY_OVER_STAGE_IDS)]
        carry_dates = list(carry["event_date"])
        last_session = max(item["includedSessionIds"] or [item["introducedSessionId"]])
        carried_into_current = (
            last_session == previous_session
            and session_end_type.get(last_session) == "prorogation"
            and (carry["session_id"] == last_session).any()
        )
        withdrawn = _to_date(item["billWithdrawn"])

        if item["isAct"]:
            outcome, outcome_date = "royal_assent", first_dates.get(STAGE_ROYAL_ASSENT)
        elif withdrawn is not None:
            outcome, outcome_date = "withdrawn", withdrawn
        elif item["isDefeated"]:
            last = bill_sittings["event_date"].max() if len(bill_sittings) else None
            outcome, outcome_date = "defeated", last
        elif last_session == current_session or carried_into_current:
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
                "stop_reason": _stop_reason_value(outcome),
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
    """Flatten the cache into parquet tables:

    - ``bills``: government bills (DATA_CONTRACT §3.1)
    - ``stage_sittings``: government-bill sittings that have happened (§3.2)
    - ``all_stage_sittings``: past sittings of every bill type (input to the calendar)
    - ``scheduled_sittings``: future-dated sittings of every bill type (live forecasts only)
    - ``sessions``, ``parliaments``, ``sitting_days``: calendar tables (§2.1, §2.2, §2.4)
    """
    gov_type = settings.government_bill_type_id
    payloads = load_cached_bills(cache_dir)
    all_sittings, scheduled = split_scheduled(flatten_stage_sittings(payloads))
    gov_payloads = [p for p in payloads if p["list_item"]["billTypeId"] == gov_type]
    gov_sittings = all_sittings[all_sittings["bill_type_id"] == gov_type].reset_index(drop=True)
    sessions = calendar.sessions(all_sittings)
    bills = flatten_bills(gov_payloads, gov_sittings, sessions)

    tables = {
        "bills": bills,
        "stage_sittings": gov_sittings,
        "all_stage_sittings": all_sittings,
        "scheduled_sittings": scheduled,
        "sessions": sessions,
        "parliaments": calendar.parliaments(),
        "sitting_days": calendar.sitting_days(all_sittings),
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(processed_dir / f"{name}.parquet", index=False)
    return tables
