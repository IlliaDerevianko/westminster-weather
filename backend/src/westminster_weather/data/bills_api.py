"""Download, flatten, and persist UK Parliament Bills API data to Neon Postgres (BUILD_PLAN task 1).

Fetches UK Government Bills (and stage sittings of all bill types for the sitting-day calendar)

from ``bills-api.parliament.uk`` and persists the flattened tables directly into Neon Lakebase
Postgres following ``DATA_CONTRACT.md`` (§2.1 ``sessions``, §2.4 ``sitting_days``, §2.5
``stage_map``, §3.1 ``bills``, §3.2 ``stage_sittings``, §3.2a ``scheduled_sittings``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pandas as pd

from westminster_weather.config import settings
from westminster_weather.data import calendar
from westminster_weather.data.stage_map import CARRY_OVER_STAGE_IDS, STAGE_ID_TO_STATE
from westminster_weather.db import save_dataset_tables_to_db
from westminster_weather.domain.stages import StopReason

logger = logging.getLogger(__name__)

MAX_PARALLEL = 20
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

    def list_stage_types(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        while True:
            page = self.get("/Stages", {"Take": 100, "Skip": len(items)})
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


def _needs_download(cached: dict[str, Any] | None, list_item: dict[str, Any], detail: bool) -> bool:
    if cached is None:
        return True
    if cached["list_item"].get("lastUpdate") != list_item.get("lastUpdate"):
        return True
    return detail and cached.get("detail") is None


def fetch_all_payloads(
    include_all_types: bool = True,
    downloader: BillsApiDownloader | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Download government bills (list + detail + stages) and, optionally, the stages of
    every other bill type (used for the sitting-day calendar).
    """
    own = downloader is None
    dl = downloader or BillsApiDownloader()
    try:
        gov_type = settings.government_bill_type_id
        logger.info("Listing bills from UK Parliament Bills API")
        stage_types = dl.list_stage_types()
        listed = dl.list_bills(None if include_all_types else gov_type)

        def work(item: dict[str, Any]) -> dict[str, Any]:
            bill_id = item["billId"]
            is_gov = item["billTypeId"] == gov_type
            return {
                "fetched_at": datetime.now(UTC).isoformat(),
                "list_item": item,
                "detail": dl.bill_detail(bill_id) if is_gov else None,
                "stages": dl.bill_stages(bill_id),
            }

        payloads: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            for i, payload in enumerate(pool.map(work, listed), start=1):
                payloads.append(payload)
                if i % 250 == 0 or i == len(listed):
                    logger.info("  %d / %d bills fetched", i, len(listed))
        n_gov = sum(item["billTypeId"] == gov_type for item in listed)
        counts = {"listed": len(listed), "government": n_gov, "downloaded": len(payloads)}
        return payloads, stage_types, counts
    finally:
        if own:
            dl.close()


# --------------------------------------------------------------------------- flatten


def _to_date(value: str | None) -> date | None:
    return datetime.fromisoformat(value).date() if value else None


def flatten_stage_map(
    stage_types: Iterable[dict[str, Any]],
    payloads: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    """One row per API stage (DATA_CONTRACT §2.5)."""
    names: dict[int, tuple[str, str | None]] = {
        t["id"]: (t["name"], t.get("house")) for t in stage_types
    }
    for payload in payloads:
        for stage in payload["stages"]:
            names.setdefault(
                stage["stageId"], (stage.get("description", ""), stage.get("house"))
            )
    rows = [
        {
            "api_stage_id": stage_id,
            "api_stage_name": name,
            "house": house if house in ("Commons", "Lords") else None,
            "model_state": STAGE_ID_TO_STATE[stage_id].value
            if stage_id in STAGE_ID_TO_STATE
            else None,
        }
        for stage_id, (name, house) in sorted(names.items())
    ]
    return pd.DataFrame(
        rows, columns=["api_stage_id", "api_stage_name", "house", "model_state"]
    )


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
        "sitting_id",
        "bill_id",
        "bill_type_id",
        "api_stage_id",
        "bill_stage_id",
        "house",
        "session_id",
        "sort_order",
        "event_date",
        "ingested_at",
    ]
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(["bill_id", "event_date", "sort_order"]).reset_index(drop=True)


def split_scheduled(sittings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sittings into those that have happened and those scheduled for after the day
    they were downloaded.
    """
    if sittings.empty:
        return sittings.copy(), sittings.copy()
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
    """One row per government bill (DATA_CONTRACT §3.1) plus ``stop_reason`` and API flags."""
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
                "kings_speech_flag": None,
                "size_clauses": None,
                "size_pages": None,
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


def build_tables_from_payloads(
    payloads: list[dict[str, Any]],
    stage_types: list[dict[str, Any]] | None = None,
    *,
    persist_to_db: bool = False,
) -> dict[str, pd.DataFrame]:
    """Flatten bill payloads into DATA_CONTRACT tables and optionally persist to Neon Postgres."""
    gov_type = settings.government_bill_type_id
    all_sittings, scheduled = split_scheduled(flatten_stage_sittings(payloads))
    gov_payloads = [p for p in payloads if p["list_item"]["billTypeId"] == gov_type]
    gov_sittings = all_sittings[all_sittings["bill_type_id"] == gov_type].reset_index(drop=True)
    sessions = calendar.sessions(all_sittings)
    bills = flatten_bills(gov_payloads, gov_sittings, sessions)
    stage_map = flatten_stage_map(stage_types or [], payloads)

    tables = {
        "stage_map": stage_map,
        "bills": bills,
        "stage_sittings": gov_sittings,
        "all_stage_sittings": all_sittings,
        "scheduled_sittings": scheduled,
        "sessions": sessions,
        "parliaments": calendar.parliaments(),
        "sitting_days": calendar.sitting_days(all_sittings),
    }
    if persist_to_db:
        save_dataset_tables_to_db(tables)
    return tables


def fetch_and_persist_all(
    include_all_types: bool = True,
    downloader: BillsApiDownloader | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Fetch all UK Parliament Bills and persist flattened tables into Neon Lakebase Postgres."""
    payloads, stage_types, counts = fetch_all_payloads(

        include_all_types=include_all_types, downloader=downloader
    )
    tables = build_tables_from_payloads(payloads, stage_types, persist_to_db=True)
    return tables, counts
