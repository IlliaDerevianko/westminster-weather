"""Tests for turning Bills API responses into the stage 1 dataset tables (offline)."""

from datetime import date, datetime
from itertools import count

from sqlalchemy import create_engine, func, select

from westminster_weather.dataset.build import (
    build_dataset,
    build_sitting_rows,
    infer_session_ends,
)
from westminster_weather.dataset.tables import bills, metadata, stage_map, stage_sittings, upsert

AS_OF = date(2026, 10, 24)
INGESTED_AT = datetime(2026, 10, 24, 12, 0)
STAGE_TYPES = [
    {"id": 6, "name": "1st reading", "house": "Commons"},
    {"id": 7, "name": "2nd reading", "house": "Commons"},
    {"id": 8, "name": "Committee stage", "house": "Commons"},
    {"id": 11, "name": "Royal Assent", "house": "Unassigned"},
    {"id": 18, "name": "Carry-over motion", "house": "Commons"},
    {"id": 49, "name": "Committee of the whole House", "house": "Commons"},
]
_sitting_ids = count(1)


def stage(stage_id: int, session_id: int, *dates: str, house: str = "Commons") -> dict:
    return {
        "stageId": stage_id,
        "sessionId": session_id,
        "house": house,
        "description": f"stage {stage_id}",
        "stageSittings": [{"id": next(_sitting_ids), "date": f"{d}T00:00:00"} for d in dates],
    }


def bill(bill_id: int, session_ids: list[int], **fields) -> dict:
    return {
        "billId": bill_id,
        "shortTitle": f"Bill {bill_id}",
        "originatingHouse": "Commons",
        "introducedSessionId": session_ids[0],
        "includedSessionIds": session_ids,
        "lastUpdate": "2026-10-23T18:34:16.022428",
        "isAct": False,
        "isDefeated": False,
        "billWithdrawn": None,
        **fields,
    }


def sample_raw_bills() -> list:
    """Session 1 ends at the 2024 dissolution, session 2 by prorogation, session 3 is current."""
    detail = {"sponsors": [{"sortOrder": 1, "organisation": {"name": "Department for Transport"}}]}
    return [
        (bill(1, [1], isAct=True), [stage(6, 1, "2024-01-10"), stage(11, 1, "2024-03-01")], {}),
        (bill(2, [1]), [stage(6, 1, "2024-02-01")], {}),
        (bill(3, [2]), [stage(6, 2, "2024-07-20")], {}),
        (bill(4, [2], billWithdrawn="2024-09-01T00:00:00"), [stage(6, 2, "2024-08-01")], {}),
        (bill(5, [2], isDefeated=True), [stage(6, 2, "2024-08-02"), stage(7, 2, "2024-09-03")], {}),
        (
            bill(6, [2, 3]),
            [
                stage(6, 2, "2025-01-05"),
                stage(49, 2, "2025-02-01"),
                stage(8, 2, "2025-02-10"),
                stage(18, 2, "2025-03-01"),
                stage(7, 3, "2026-06-01"),
            ],
            detail,
        ),
        (bill(7, [3]), [stage(6, 3, "2026-12-01")], {}),  # only a scheduled sitting
    ]


def test_future_sittings_are_dropped() -> None:
    rows = build_sitting_rows(1, [stage(6, 3, "2026-10-24", "2026-10-25")], AS_OF, INGESTED_AT)
    assert [r["event_date"] for r in rows] == [AS_OF]


def test_session_end_inference() -> None:
    sittings = [
        {"session_id": 1, "event_date": date(2024, 5, 24)},
        {"session_id": 2, "event_date": date(2024, 7, 18)},
        {"session_id": 2, "event_date": date(2025, 4, 1)},
        {"session_id": 3, "event_date": date(2026, 5, 14)},
    ]
    assert infer_session_ends(sittings) == {1: date(2024, 5, 30), 2: None}


def test_bill_outcomes() -> None:
    ds = build_dataset(STAGE_TYPES, sample_raw_bills(), AS_OF, INGESTED_AT)
    outcomes = {b["bill_id"]: (b["outcome"], b["outcome_date"]) for b in ds.bills}
    assert outcomes == {
        1: ("royal_assent", date(2024, 3, 1)),
        2: ("fell_dissolution", date(2024, 5, 30)),
        3: ("fell_session_end", None),
        4: ("withdrawn", date(2024, 9, 1)),
        5: ("defeated", date(2024, 9, 3)),
        6: ("in_progress", None),
    }
    assert ds.skipped_not_introduced == 1
    assert ds.problems == []


def test_bill_row_fields() -> None:
    ds = build_dataset(STAGE_TYPES, sample_raw_bills(), AS_OF, INGESTED_AT)
    row = next(b for b in ds.bills if b["bill_id"] == 6)
    assert row["introduced_date"] == date(2025, 1, 5)
    assert row["commons_committee_type"] == "whole_house"  # whole House sat first
    assert row["carry_over_date"] == date(2025, 3, 1)
    assert row["department"] == "Department for Transport"
    assert row["kings_speech_flag"] is None


def test_stage_map_includes_stages_missing_from_stage_types() -> None:
    raw = [(bill(1, [1]), [stage(45, 1, "2024-01-10")], {})]
    ds = build_dataset(STAGE_TYPES, raw, AS_OF, INGESTED_AT)
    by_id = {r["api_stage_id"]: r for r in ds.stage_map}
    assert by_id[45]["model_state"] is None
    assert by_id[6]["model_state"] == "C_1R"
    assert by_id[11]["house"] is None  # 'Unassigned' is stored as NULL


def test_upsert_is_idempotent() -> None:
    ds = build_dataset(STAGE_TYPES, sample_raw_bills(), AS_OF, INGESTED_AT)
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    for _ in range(2):
        with engine.begin() as conn:
            upsert(conn, stage_map, ds.stage_map)
            upsert(conn, bills, ds.bills)
            upsert(conn, stage_sittings, ds.stage_sittings)
    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(bills)) == len(ds.bills)
        assert conn.scalar(select(func.count()).select_from(stage_sittings)) == len(
            ds.stage_sittings
        )
