"""Turn raw Bills API responses into rows for the dataset tables. No network access here."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

# DATA_CONTRACT.md 2.5: Bills API stage ID -> model state. Every other stage is still
# stored in stage_map, with model_state NULL (ignored by the model).
MODEL_STATE_BY_API_STAGE = {
    6: "C_1R",
    7: "C_2R",
    14: "C_MOTIONS",
    15: "C_MOTIONS",
    8: "C_COMMITTEE",
    49: "C_COMMITTEE",
    9: "C_REPORT",
    10: "C_3R",
    1: "L_1R_2R",
    2: "L_1R_2R",
    3: "L_COMMITTEE",
    4: "L_REPORT_3R",
    5: "L_REPORT_3R",
    12: "PING_PONG",
    13: "PING_PONG",
    11: "ROYAL_ASSENT",
}
ROYAL_ASSENT_STAGE = 11
FIRST_READING_STAGE = {"Commons": 6, "Lords": 1}
PUBLIC_BILL_COMMITTEE_STAGE = 8
WHOLE_HOUSE_COMMITTEE_STAGE = 49
CARRY_OVER_MOTION_STAGES = (18, 19)
HOUSES = {"Commons", "Lords"}

# Dissolutions within Bills API coverage. A session ending in one of these means its
# unfinished bills fell at dissolution rather than at prorogation.
DISSOLUTIONS = [
    date(2010, 4, 12),
    date(2015, 3, 30),
    date(2017, 5, 3),
    date(2019, 11, 6),
    date(2024, 5, 30),
]


@dataclass
class Dataset:
    stage_map: list[dict[str, Any]] = field(default_factory=list)
    bills: list[dict[str, Any]] = field(default_factory=list)
    stage_sittings: list[dict[str, Any]] = field(default_factory=list)
    skipped_not_introduced: int = 0
    problems: list[str] = field(default_factory=list)


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value[:10]) if value else None


def _timestamp(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def _house(value: str | None) -> str | None:
    return value if value in HOUSES else None


def build_stage_map_rows(
    stage_types: list[dict[str, Any]], bill_stages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per API stage. Stages seen on bills but missing from /Stages are added too,
    so every sitting has a stage_map row."""
    names = {t["id"]: (t["name"], t.get("house")) for t in stage_types}
    for stage in bill_stages:
        names.setdefault(stage["stageId"], (stage.get("description", ""), stage.get("house")))
    return [
        {
            "api_stage_id": stage_id,
            "api_stage_name": name,
            "house": _house(house),
            "model_state": MODEL_STATE_BY_API_STAGE.get(stage_id),
        }
        for stage_id, (name, house) in sorted(names.items())
    ]


def build_sitting_rows(
    bill_id: int, stages: list[dict[str, Any]], as_of: date, ingested_at: datetime
) -> list[dict[str, Any]]:
    """One row per stage sitting that has happened. The API also lists scheduled future
    sittings; those are dropped so no event_date is ever after ingestion."""
    rows = []
    for stage in stages:
        for sitting in stage.get("stageSittings") or []:
            event_date = _date(sitting.get("date"))
            if event_date is None or event_date > as_of:
                continue
            rows.append(
                {
                    "sitting_id": sitting["id"],
                    "bill_id": bill_id,
                    "api_stage_id": stage["stageId"],
                    "house": _house(stage.get("house")),
                    "session_id": stage["sessionId"],
                    "event_date": event_date,
                    "ingested_at": ingested_at,
                }
            )
    return rows


def infer_session_ends(sittings: list[dict[str, Any]]) -> dict[int, date | None]:
    """For each finished session: the dissolution date if it ended in one, else None
    (prorogation). The latest session is still running and is left out.

    The Bills API has no sessions endpoint, so session spans come from sitting dates.
    """
    first: dict[int, date] = {}
    last: dict[int, date] = {}
    for s in sittings:
        sid, d = s["session_id"], s["event_date"]
        first[sid] = min(first.get(sid, d), d)
        last[sid] = max(last.get(sid, d), d)

    ends: dict[int, date | None] = {}
    for sid in last:
        later_starts = [d for other, d in first.items() if other > sid]
        if not later_starts:
            continue
        next_start = min(later_starts)
        ends[sid] = next((d for d in DISSOLUTIONS if last[sid] <= d < next_start), None)
    return ends


def bill_outcome(
    bill: dict[str, Any], sittings: list[dict[str, Any]], session_ends: dict[int, date | None]
) -> tuple[str, date | None]:
    royal_assent = [s["event_date"] for s in sittings if s["api_stage_id"] == ROYAL_ASSENT_STAGE]
    if bill.get("isAct") or royal_assent:
        return "royal_assent", min(royal_assent, default=None)
    if bill.get("billWithdrawn"):
        return "withdrawn", _date(bill["billWithdrawn"])
    if bill.get("isDefeated"):
        return "defeated", max(s["event_date"] for s in sittings)
    last_session = max(bill.get("includedSessionIds") or [bill["introducedSessionId"]])
    if last_session in session_ends:
        dissolution = session_ends[last_session]
        # Prorogation dates are not in the Bills API, so fell_session_end has no date yet.
        return ("fell_dissolution", dissolution) if dissolution else ("fell_session_end", None)
    return "in_progress", None


def sponsoring_department(detail: dict[str, Any]) -> str | None:
    sponsors = sorted(detail.get("sponsors") or [], key=lambda s: s.get("sortOrder") or 0)
    for sponsor in sponsors:
        name = (sponsor.get("organisation") or {}).get("name")
        if name:
            return name
    return None


def build_bill_row(
    bill: dict[str, Any],
    detail: dict[str, Any],
    sittings: list[dict[str, Any]],
    session_ends: dict[int, date | None],
    ingested_at: datetime,
) -> dict[str, Any] | None:
    """The bills row, or None if the bill has no sittings yet (not introduced)."""
    if not sittings:
        return None
    dates_by_stage: dict[int, list[date]] = defaultdict(list)
    for s in sittings:
        dates_by_stage[s["api_stage_id"]].append(s["event_date"])

    def first_sitting(*stage_ids: int) -> date | None:
        return min((d for sid in stage_ids for d in dates_by_stage.get(sid, [])), default=None)

    first_reading = FIRST_READING_STAGE.get(bill.get("originatingHouse", ""))
    introduced = (first_reading and first_sitting(first_reading)) or min(
        s["event_date"] for s in sittings
    )

    public_bill_committee = first_sitting(PUBLIC_BILL_COMMITTEE_STAGE)
    whole_house = first_sitting(WHOLE_HOUSE_COMMITTEE_STAGE)
    if public_bill_committee and (not whole_house or public_bill_committee <= whole_house):
        committee_type = "public_bill_committee"
    elif whole_house:
        committee_type = "whole_house"
    else:
        committee_type = None

    outcome, outcome_date = bill_outcome(bill, sittings, session_ends)
    return {
        "bill_id": bill["billId"],
        "short_title": bill["shortTitle"],
        "originating_house": bill["originatingHouse"],
        "introduced_session_id": bill["introducedSessionId"],
        "introduced_date": introduced,
        "department": sponsoring_department(detail),
        "kings_speech_flag": None,  # Tier 2: not in the Bills API
        "size_clauses": None,  # needs the first published version of the bill text
        "size_pages": None,
        "commons_committee_type": committee_type,
        "carry_over_date": first_sitting(*CARRY_OVER_MOTION_STAGES),
        "outcome": outcome,
        "outcome_date": outcome_date,
        "last_api_update": _timestamp(bill.get("lastUpdate")),
        "ingested_at": ingested_at,
    }


def build_dataset(
    stage_types: list[dict[str, Any]],
    raw_bills: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
    as_of: date,
    ingested_at: datetime,
) -> Dataset:
    """raw_bills: (bill from /Bills, its stages, its detail from /Bills/{id}) per bill."""
    sittings_by_bill = {
        bill["billId"]: build_sitting_rows(bill["billId"], stages, as_of, ingested_at)
        for bill, stages, _ in raw_bills
    }
    session_ends = infer_session_ends([s for rows in sittings_by_bill.values() for s in rows])

    ds = Dataset(
        stage_map=build_stage_map_rows(
            stage_types, [stage for _, stages, _ in raw_bills for stage in stages]
        )
    )
    for bill, _, detail in raw_bills:
        sittings = sittings_by_bill[bill["billId"]]
        row = build_bill_row(bill, detail, sittings, session_ends, ingested_at)
        if row is None:
            ds.skipped_not_introduced += 1
            continue
        ds.bills.append(row)
        ds.stage_sittings.extend(sittings)
    ds.problems = validate(ds, session_ends)
    return ds


def validate(ds: Dataset, session_ends: dict[int, date | None]) -> list[str]:
    """DATA_CONTRACT.md section 8 checks that apply to these tables."""
    problems = []
    assented = {s["bill_id"] for s in ds.stage_sittings if s["api_stage_id"] == ROYAL_ASSENT_STAGE}
    for bill in ds.bills:
        bid = bill["bill_id"]
        if (bill["outcome"] == "royal_assent") != (bid in assented):
            problems.append(
                f"bill {bid}: outcome {bill['outcome']} but Royal Assent sitting "
                f"{'found' if bid in assented else 'missing'}"
            )
        if (
            bill["outcome"] == "in_progress"
            and bill["introduced_session_id"] in session_ends
            and bill["carry_over_date"] is None
        ):
            problems.append(f"bill {bid}: in_progress from a finished session with no carry-over")
    return problems
