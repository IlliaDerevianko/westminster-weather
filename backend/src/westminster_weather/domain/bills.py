"""Domain model and Neon Lakebase Postgres persistence for tracking UK Government Bills
across Phase 1 (Before Parliament) using real historical data from UK public APIs.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

from sqlalchemy import delete, select

from westminster_weather.clients.govuk_content import GovUkContentClient
from westminster_weather.clients.parliament_bills import ParliamentBillsClient
from westminster_weather.db import SessionLocal, init_db
from westminster_weather.domain.stages import (
    GOVERNMENT_BILL_PIPELINE,
    BillStage,
    StageNode,
    StopReason,
)
from westminster_weather.models import (
    ApiEvidenceEventRow,
    StageRecordRow,
    TrackedBillRow,
)


class StageStatus(StrEnum):
    COMPLETED = "completed"
    CURRENT = "current"
    UPCOMING = "upcoming"
    STOPPED = "stopped"


@dataclass(frozen=True)
class ApiEvidenceEvent:
    """A single timestamped event fetched from an official UK API and stored in Neon."""

    event_date: date
    title: str
    note: str
    source_api_url: str
    web_url: str
    document_type: str


@dataclass(frozen=True)
class StageHistoricalRecord:
    """Real historical milestone data for a stage, stored in Neon Postgres."""

    stage: BillStage
    entered_date: date
    headline: str
    source_api_url: str
    web_url: str
    events: tuple[ApiEvidenceEvent, ...] = ()


@dataclass
class TrackedStageNode:
    """Evaluation of a single Phase 1 stage as of a specific `as_of_date`."""

    node: StageNode
    status: StageStatus
    entered_date: date | None = None
    completed_date: date | None = None
    summary: str | None = None
    source_reference: str | None = None
    source_api_url: str | None = None
    web_url: str | None = None
    visible_events: list[ApiEvidenceEvent] = field(default_factory=list)
    active_stop_reason: StopReason | None = None


@dataclass
class GovernmentBill:
    """Represents a UK Government Bill and evaluates its Phase 1 state at any `as_of_date`."""

    id: str
    parliament_bill_id: int
    short_title: str
    long_title: str
    sponsoring_department: str
    earliest_signal_date: date
    first_reading_date: date
    stage_records: dict[BillStage, StageHistoricalRecord]

    def evaluate_at_date(
        self, as_of_date: date
    ) -> tuple[BillStage | None, int, list[TrackedStageNode]]:
        """Evaluate every Phase 1 stage dynamically as of `as_of_date`."""
        reached_orders: list[int] = [
            node.order
            for node in GOVERNMENT_BILL_PIPELINE
            if (rec := self.stage_records.get(node.stage)) is not None
            and rec.entered_date <= as_of_date
        ]
        current_order = max(reached_orders) if reached_orders else 0
        current_stage: BillStage | None = None

        tracked: list[TrackedStageNode] = []
        for idx, node in enumerate(GOVERNMENT_BILL_PIPELINE):
            rec = self.stage_records.get(node.stage)
            next_node = (
                GOVERNMENT_BILL_PIPELINE[idx + 1]
                if idx + 1 < len(GOVERNMENT_BILL_PIPELINE)
                else None
            )
            next_rec = self.stage_records.get(next_node.stage) if next_node else None

            if current_order == 0 or node.order > current_order or rec is None:
                tracked.append(
                    TrackedStageNode(
                        node=node,
                        status=StageStatus.UPCOMING,
                    )
                )
                continue

            if node.order < current_order:
                status = StageStatus.COMPLETED
                completed_date = next_rec.entered_date if next_rec else rec.entered_date
            elif node.stage == BillStage.READY_FOR_FIRST_READING:
                status = StageStatus.COMPLETED
                current_stage = node.stage
                completed_date = rec.entered_date
            else:
                status = StageStatus.CURRENT
                current_stage = node.stage
                completed_date = None

            visible_events = [e for e in rec.events if e.event_date <= as_of_date]

            tracked.append(
                TrackedStageNode(
                    node=node,
                    status=status,
                    entered_date=rec.entered_date,
                    completed_date=completed_date,
                    summary=rec.headline,
                    source_reference=rec.web_url,
                    source_api_url=rec.source_api_url,
                    web_url=rec.web_url,
                    visible_events=visible_events,
                )
            )

        return current_stage, current_order, tracked


def _parse_iso_date(iso_str: str) -> date:
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).date()


async def fetch_online_safety_bill_from_uk_apis() -> GovernmentBill:
    """Fetch the Online Safety Bill's Phase 1 history directly from GOV.UK & Parliament APIs."""
    govuk = GovUkContentClient()

    green_paper_path = "/government/consultations/internet-safety-strategy-green-paper"
    white_paper_path = "/government/consultations/online-harms-white-paper"
    draft_bill_path = "/government/publications/draft-online-safety-bill"
    queens_speech_path = (
        "/government/publications/queens-speech-2021-background-briefing-notes"
    )
    joint_comm_resp_path = (
        "/government/publications/"
        "joint-committee-report-on-the-draft-online-safety-bill-government-response"
    )

    async with ParliamentBillsClient() as parliament:
        (
            green_paper,
            white_paper,
            draft_bill,
            queens_speech,
            joint_comm_resp,
            parl_bill,
            parl_stages,
        ) = await asyncio.gather(
            govuk.get_content(green_paper_path),
            govuk.get_content(white_paper_path),
            govuk.get_content(draft_bill_path),
            govuk.get_content(queens_speech_path),
            govuk.get_content(joint_comm_resp_path),
            parliament.get_bill(3137),
            parliament.get_bill_stages(3137),
        )

    # Stage 1: Signal appears
    s1_date = _parse_iso_date(green_paper["first_published_at"])
    s1_api_url = f"https://www.gov.uk/api/content{green_paper_path}"
    s1_web_url = f"https://www.gov.uk{green_paper['base_path']}"
    s1_events = [
        ApiEvidenceEvent(
            event_date=_parse_iso_date(ch["public_timestamp"]),
            title=green_paper["title"],
            note=ch["note"],
            source_api_url=s1_api_url,
            web_url=s1_web_url,
            document_type=green_paper["document_type"],
        )
        for ch in sorted(
            green_paper.get("details", {}).get("change_history", []),
            key=lambda x: x["public_timestamp"],
        )
        if "First published" in ch["note"]
    ]

    # Stage 2: Government commits
    s2_history = [
        ch
        for ch in sorted(
            green_paper.get("details", {}).get("change_history", []),
            key=lambda x: x["public_timestamp"],
        )
        if "First published" not in ch["note"]
    ]
    s2_date = _parse_iso_date(s2_history[0]["public_timestamp"])
    s2_events = [
        ApiEvidenceEvent(
            event_date=_parse_iso_date(ch["public_timestamp"]),
            title=green_paper["title"],
            note=ch["note"],
            source_api_url=s1_api_url,
            web_url=s1_web_url,
            document_type=green_paper["document_type"],
        )
        for ch in s2_history
    ]

    # Stage 3: Consultation
    s3_date = _parse_iso_date(white_paper["first_published_at"])
    s3_api_url = f"https://www.gov.uk/api/content{white_paper_path}"
    s3_web_url = f"https://www.gov.uk{white_paper['base_path']}"
    s3_events = [
        ApiEvidenceEvent(
            event_date=_parse_iso_date(ch["public_timestamp"]),
            title=white_paper["title"],
            note=ch["note"],
            source_api_url=s3_api_url,
            web_url=s3_web_url,
            document_type=white_paper["document_type"],
        )
        for ch in sorted(
            white_paper.get("details", {}).get("change_history", []),
            key=lambda x: x["public_timestamp"],
        )
    ]

    # Stage 4: Drafting
    s4_date = _parse_iso_date(white_paper["public_updated_at"])
    s4_api_url = f"https://www.gov.uk/api/content{draft_bill_path}"
    s4_web_url = f"https://www.gov.uk{draft_bill['base_path']}"
    draft_pub_date = _parse_iso_date(draft_bill["first_published_at"])
    s4_events = [
        ApiEvidenceEvent(
            event_date=s4_date,
            title=white_paper["title"],
            note=(
                "Full government response to Online Harms consultation published; "
                "Parliamentary Counsel begins drafting the Online Safety Bill."
            ),
            source_api_url=s3_api_url,
            web_url=s3_web_url,
            document_type=white_paper["document_type"],
        ),
        ApiEvidenceEvent(
            event_date=draft_pub_date,
            title=draft_bill["title"],
            note=draft_bill["description"],
            source_api_url=s4_api_url,
            web_url=s4_web_url,
            document_type=draft_bill["document_type"],
        ),
    ]

    # Stage 5: Slot in the programme
    s5_date = _parse_iso_date(queens_speech["first_published_at"])
    s5_api_url = f"https://www.gov.uk/api/content{queens_speech_path}"
    s5_web_url = f"https://www.gov.uk{queens_speech['base_path']}"
    s5_events = [
        ApiEvidenceEvent(
            event_date=s5_date,
            title=queens_speech["title"],
            note=queens_speech["description"],
            source_api_url=s5_api_url,
            web_url=s5_web_url,
            document_type=queens_speech["document_type"],
        ),
        ApiEvidenceEvent(
            event_date=draft_pub_date,
            title=draft_bill["title"],
            note=(
                "Draft Online Safety Bill published for Joint Committee "
                "pre-legislative scrutiny."
            ),
            source_api_url=s4_api_url,
            web_url=s4_web_url,
            document_type=draft_bill["document_type"],
        ),
    ]

    # Stage 6: Ready for first reading
    first_reading_stage = next(
        st
        for st in parl_stages.get("items", [])
        if st.get("description") == "1st reading" and st.get("house") == "Commons"
    )
    s6_date = _parse_iso_date(first_reading_stage["stageSittings"][0]["date"])
    s6_api_url = "https://bills-api.parliament.uk/api/v1/Bills/3137/Stages"
    s6_web_url = f"https://bills.parliament.uk/bills/{parl_bill['billId']}"
    jc_api_url = f"https://www.gov.uk/api/content{joint_comm_resp_path}"
    jc_web_url = f"https://www.gov.uk{joint_comm_resp['base_path']}"
    s6_events = [
        ApiEvidenceEvent(
            event_date=_parse_iso_date(joint_comm_resp["first_published_at"]),
            title=joint_comm_resp["title"],
            note=joint_comm_resp["description"].strip(),
            source_api_url=jc_api_url,
            web_url=jc_web_url,
            document_type=joint_comm_resp["document_type"],
        ),
        ApiEvidenceEvent(
            event_date=s6_date,
            title=f"UK Parliament Bill #{parl_bill['billId']}: {parl_bill['shortTitle']}",
            note=(
                f"Formally introduced for Commons 1st Reading "
                f"(stageSittingId={first_reading_stage['stageSittings'][0]['id']})."
            ),
            source_api_url=s6_api_url,
            web_url=s6_web_url,
            document_type="parliament_bill_stage",
        ),
    ]

    records: dict[BillStage, StageHistoricalRecord] = {
        BillStage.SIGNAL_APPEARS: StageHistoricalRecord(
            stage=BillStage.SIGNAL_APPEARS,
            entered_date=s1_date,
            headline=f"{green_paper['title']}: {green_paper['description']}",
            source_api_url=s1_api_url,
            web_url=s1_web_url,
            events=tuple(s1_events),
        ),
        BillStage.GOVERNMENT_COMMITS: StageHistoricalRecord(
            stage=BillStage.GOVERNMENT_COMMITS,
            entered_date=s2_date,
            headline=(
                "Government Response to the Internet Safety Strategy Green Paper commits "
                "to primary legislation for online safety."
            ),
            source_api_url=s1_api_url,
            web_url=s1_web_url,
            events=tuple(s2_events),
        ),
        BillStage.CONSULTATION: StageHistoricalRecord(
            stage=BillStage.CONSULTATION,
            entered_date=s3_date,
            headline=f"{white_paper['title']}: {white_paper['description']}",
            source_api_url=s3_api_url,
            web_url=s3_web_url,
            events=tuple(s3_events),
        ),
        BillStage.DRAFTING: StageHistoricalRecord(
            stage=BillStage.DRAFTING,
            entered_date=s4_date,
            headline=f"{draft_bill['title']}: {draft_bill['description']}",
            source_api_url=s4_api_url,
            web_url=s4_web_url,
            events=tuple(s4_events),
        ),
        BillStage.SLOT_IN_PROGRAMME: StageHistoricalRecord(
            stage=BillStage.SLOT_IN_PROGRAMME,
            entered_date=s5_date,
            headline=f"{queens_speech['title']}: {queens_speech['description']}",
            source_api_url=s5_api_url,
            web_url=s5_web_url,
            events=tuple(s5_events),
        ),
        BillStage.READY_FOR_FIRST_READING: StageHistoricalRecord(
            stage=BillStage.READY_FOR_FIRST_READING,
            entered_date=s6_date,
            headline=(
                f"{joint_comm_resp['title']} published and Bill #{parl_bill['billId']} "
                f"introduced for Commons 1st Reading."
            ),
            source_api_url=s6_api_url,
            web_url=s6_web_url,
            events=tuple(s6_events),
        ),
    }

    return GovernmentBill(
        id="online-safety-bill",
        parliament_bill_id=parl_bill["billId"],
        short_title="Online Safety Bill",
        long_title=parl_bill.get("longTitle") or draft_bill["description"],
        sponsoring_department=(
            "Department for Digital, Culture, Media & Sport (DCMS) & Home Office"
        ),
        earliest_signal_date=s1_date,
        first_reading_date=s6_date,
        stage_records=records,
    )


async def sync_online_safety_bill_to_db() -> tuple[int, int]:
    """Fetch the Online Safety Bill from UK public APIs and persist it into Neon Postgres."""
    await init_db()
    bill = await fetch_online_safety_bill_from_uk_apis()

    stages_count = 0
    events_count = 0

    async with SessionLocal() as session:
        async with session.begin():
            await session.execute(
                delete(TrackedBillRow).where(TrackedBillRow.id == bill.id)
            )

            bill_row = TrackedBillRow(
                id=bill.id,
                parliament_bill_id=bill.parliament_bill_id,
                short_title=bill.short_title,
                long_title=bill.long_title,
                sponsoring_department=bill.sponsoring_department,
                earliest_signal_date=bill.earliest_signal_date,
                first_reading_date=bill.first_reading_date,
            )
            session.add(bill_row)

            for stage_enum, rec in bill.stage_records.items():
                stages_count += 1
                session.add(
                    StageRecordRow(
                        bill_id=bill.id,
                        stage=stage_enum.value,
                        entered_date=rec.entered_date,
                        headline=rec.headline,
                        source_api_url=rec.source_api_url,
                        web_url=rec.web_url,
                    )
                )
                for ev in rec.events:
                    events_count += 1
                    session.add(
                        ApiEvidenceEventRow(
                            bill_id=bill.id,
                            stage=stage_enum.value,
                            event_date=ev.event_date,
                            title=ev.title,
                            note=ev.note,
                            source_api_url=ev.source_api_url,
                            web_url=ev.web_url,
                            document_type=ev.document_type,
                        )
                    )

    return stages_count, events_count


def _row_to_domain(row: TrackedBillRow) -> GovernmentBill:
    events_by_stage: dict[str, list[ApiEvidenceEvent]] = {}
    for ev in sorted(row.evidence_events, key=lambda e: (e.event_date, e.id)):
        events_by_stage.setdefault(ev.stage, []).append(
            ApiEvidenceEvent(
                event_date=ev.event_date,
                title=ev.title,
                note=ev.note,
                source_api_url=ev.source_api_url,
                web_url=ev.web_url,
                document_type=ev.document_type,
            )
        )

    records: dict[BillStage, StageHistoricalRecord] = {}
    for sr in row.stage_records:
        stage_enum = BillStage(sr.stage)
        records[stage_enum] = StageHistoricalRecord(
            stage=stage_enum,
            entered_date=sr.entered_date,
            headline=sr.headline,
            source_api_url=sr.source_api_url,
            web_url=sr.web_url,
            events=tuple(events_by_stage.get(sr.stage, [])),
        )

    return GovernmentBill(
        id=row.id,
        parliament_bill_id=row.parliament_bill_id,
        short_title=row.short_title,
        long_title=row.long_title,
        sponsoring_department=row.sponsoring_department,
        earliest_signal_date=row.earliest_signal_date,
        first_reading_date=row.first_reading_date,
        stage_records=records,
    )


_DB_INIT_LOCK = asyncio.Lock()


async def get_online_safety_bill_from_live_apis() -> GovernmentBill:
    """Load the Online Safety Bill from Neon Lakebase Postgres, syncing from UK APIs if empty."""
    await init_db()
    async with SessionLocal() as session:
        result = await session.execute(
            select(TrackedBillRow).where(TrackedBillRow.id == "online-safety-bill")
        )
        row = result.scalar_one_or_none()
        if row is not None and len(row.stage_records) == len(GOVERNMENT_BILL_PIPELINE):
            return _row_to_domain(row)

    async with _DB_INIT_LOCK:
        async with SessionLocal() as session:
            result = await session.execute(
                select(TrackedBillRow).where(TrackedBillRow.id == "online-safety-bill")
            )
            row = result.scalar_one_or_none()
            if row is not None and len(row.stage_records) == len(GOVERNMENT_BILL_PIPELINE):
                return _row_to_domain(row)

        await sync_online_safety_bill_to_db()

        async with SessionLocal() as session:
            result = await session.execute(
                select(TrackedBillRow).where(TrackedBillRow.id == "online-safety-bill")
            )
            row = result.scalar_one()
            return _row_to_domain(row)
