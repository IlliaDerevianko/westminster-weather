"""Domain model and Neon Lakebase Postgres persistence for tracking UK Government Bills
across Phase 1 (Before Parliament) using real historical data from UK public APIs.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from sqlalchemy import or_, select

from westminster_weather.data.stage_map import OPTIONAL_STATES, STAGE_ID_TO_STATE, chain
from westminster_weather.db import SessionLocal, init_db
from westminster_weather.domain.stages import (
    GOVERNMENT_BILL_PIPELINE,
    PARLIAMENTARY_STAGE_INFO,
    POST_ROYAL_ASSENT_STAGE_INFO,
    BillStage,
    LifecyclePhase,
    StageNode,
    StopReason,
)
from westminster_weather.models import (
    ActCommencementRow,
    ActInForceStatusRow,
    BillRow,
    PostRoyalAssentEventRow,
    StageMapRow,
    StageSittingRow,
    TrackedBillRow,
)


class StageStatus(StrEnum):
    COMPLETED = "completed"
    FAST_TRACKED = "fast_tracked"
    SKIPPED = "skipped"
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
    sitting_count: int = 0


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


def fully_in_force_date(
    act_comm: ActCommencementRow | None,
    in_force_status: ActInForceStatusRow | None,
    ra_date: date | None,
) -> date | None:
    """Date an Act became fully in force, or ``None`` if provisions are still prospective."""
    in_force_dt = (
        act_comm.law_fully_in_force_date
        if act_comm is not None and act_comm.law_fully_in_force_date is not None
        else ra_date
    )
    # legislation.gov.uk's current version (`act_in_force_status`) overrides the estimate above:
    # fully in force only once no provision is still prospective, on the latest start date.
    if in_force_status is not None:
        in_force_dt = (
            (in_force_status.latest_in_force_start_date or in_force_dt)
            if in_force_status.fully_in_force
            else None
        )
    return in_force_dt


POST_ROYAL_ASSENT_LABELS: dict[BillStage, str] = {
    BillStage.ROYAL_ASSENT: "Royal Assent",
    **{st: label for st, label, _, _ in POST_ROYAL_ASSENT_STAGE_INFO},
}


def latest_confirmed_post_ra_stage(
    in_force_dt: date | None, event_stages: set[str], as_of: date
) -> BillStage:
    """Furthest stage an Act has completed by ``as_of``, with the same rules as the bill
    page: fully in force, else the latest SI stage with any instrument, else "some sections
    start" once anything after it has happened, else Royal Assent. ``event_stages`` are the
    `post_royal_assent_events` stages with an event on or before ``as_of``."""
    if in_force_dt is not None and in_force_dt <= as_of:
        return BillStage.LAW_FULLY_IN_FORCE
    for st in (
        BillStage.REGULATIONS_LAID,
        BillStage.REGULATIONS_DRAFTED,
        BillStage.COMMENCEMENT_ORDERS,
    ):
        if st.value in event_stages:
            return st
    early = {BillStage.ROYAL_ASSENT.value, BillStage.SOME_SECTIONS_START.value}
    if event_stages - early:
        return BillStage.SOME_SECTIONS_START
    return BillStage.ROYAL_ASSENT


async def list_all_bills_from_db(
    search: str | None = None,
    outcome: str | None = None,
    house: str | None = None,
) -> list[dict[str, Any]]:
    """Return all UK Government Bills from the `bills` table in Neon Lakebase Postgres."""
    await init_db()
    async with SessionLocal() as session:
        tracked_res = await session.execute(
            select(TrackedBillRow.parliament_bill_id, TrackedBillRow.earliest_signal_date)
        )
        pre_parl_dates = {pid: sig_dt for pid, sig_dt in tracked_res.all() if pid is not None}

        stmt = select(BillRow)
        if search and search.strip():
            q = f"%{search.strip()}%"
            conditions = [
                BillRow.short_title.ilike(q),
                BillRow.department.ilike(q),
            ]
            if search.strip().isdigit():
                conditions.append(BillRow.bill_id == int(search.strip()))
            stmt = stmt.where(or_(*conditions))
        if outcome and outcome.strip():
            stmt = stmt.where(BillRow.outcome == outcome.strip())
        if house and house.strip():
            stmt = stmt.where(BillRow.originating_house == house.strip())

        stmt = stmt.order_by(BillRow.introduced_date.desc().nulls_last(), BillRow.bill_id.desc())
        rows = (await session.execute(stmt)).scalars().all()

        # Latest confirmed stage for Acts (as of today), from the same tables as the bill page
        today = date.today()
        act_comms = {
            r.bill_id: r for r in (await session.execute(select(ActCommencementRow))).scalars()
        }
        in_force = {
            r.bill_id: r for r in (await session.execute(select(ActInForceStatusRow))).scalars()
        }
        event_stages: dict[int, set[str]] = {}
        for bid, st in (
            await session.execute(
                select(PostRoyalAssentEventRow.bill_id, PostRoyalAssentEventRow.stage)
                .where(PostRoyalAssentEventRow.event_date <= today)
                .distinct()
            )
        ).all():
            event_stages.setdefault(bid, set()).add(st)

        def latest_stage(r: BillRow) -> BillStage | None:
            if r.outcome != "royal_assent":
                return None
            ac = act_comms.get(r.bill_id)
            ra_date = ac.royal_assent_date if ac is not None else r.outcome_date
            in_force_dt = fully_in_force_date(ac, in_force.get(r.bill_id), ra_date)
            return latest_confirmed_post_ra_stage(
                in_force_dt, event_stages.get(r.bill_id, set()), today
            )

        latest = {r.bill_id: latest_stage(r) for r in rows}
        return [
            {
                "bill_id": r.bill_id,
                "short_title": r.short_title,
                "originating_house": r.originating_house,
                "introduced_session_id": r.introduced_session_id,
                "earliest_signal_date": pre_parl_dates.get(r.bill_id) or r.introduced_date,
                "introduced_date": r.introduced_date,
                "department": r.department,
                "outcome": r.outcome,
                "outcome_date": r.outcome_date,
                "stop_reason": r.stop_reason,
                "is_act": r.is_act,
                "is_defeated": r.is_defeated,
                "withdrawn_date": r.withdrawn_date,
                "has_pre_parliament_data": r.bill_id in pre_parl_dates,
                "latest_stage": latest[r.bill_id],
                "latest_stage_label": (
                    POST_ROYAL_ASSENT_LABELS[latest[r.bill_id]] if latest[r.bill_id] else None
                ),
            }
            for r in rows
        ]


async def evaluate_parliament_bill_from_db(
    bill_id: int, as_of_date: date | None = None
) -> dict[str, Any] | None:
    """Evaluate any UK Government Bill across all 3 phases from Neon Lakebase Postgres."""
    await init_db()
    async with SessionLocal() as session:
        bill_row = (
            await session.execute(select(BillRow).where(BillRow.bill_id == bill_id))
        ).scalar_one_or_none()
        if bill_row is None:
            return None

        tracked_pre_row = (
            await session.execute(
                select(TrackedBillRow).where(TrackedBillRow.parliament_bill_id == bill_id)
            )
        ).scalar_one_or_none()

        sittings = (
            (
                await session.execute(
                    select(StageSittingRow)
                    .where(StageSittingRow.bill_id == bill_id)
                    .order_by(StageSittingRow.event_date, StageSittingRow.sort_order)
                )
            )
            .scalars()
            .all()
        )
        stage_map_rows = (await session.execute(select(StageMapRow))).scalars().all()
        stage_names = {r.api_stage_id: r.api_stage_name for r in stage_map_rows}

        act_comm = (
            await session.execute(
                select(ActCommencementRow).where(ActCommencementRow.bill_id == bill_id)
            )
        ).scalar_one_or_none()
        in_force_status = (
            await session.execute(
                select(ActInForceStatusRow).where(ActInForceStatusRow.bill_id == bill_id)
            )
        ).scalar_one_or_none()
        post_event_rows = (
            (
                await session.execute(
                    select(PostRoyalAssentEventRow)
                    .where(PostRoyalAssentEventRow.bill_id == bill_id)
                    .order_by(PostRoyalAssentEventRow.event_date, PostRoyalAssentEventRow.event_id)
                )
            )
            .scalars()
            .all()
        )

    pre_domain = _row_to_domain(tracked_pre_row) if tracked_pre_row is not None else None
    all_dates = [s.event_date for s in sittings]
    post_dates = [e.event_date for e in post_event_rows]
    first_reading_dt = bill_row.introduced_date or (min(all_dates) if all_dates else date.today())
    earliest_date = pre_domain.earliest_signal_date if pre_domain is not None else first_reading_dt
    latest_sitting_date = max(all_dates) if all_dates else first_reading_dt
    latest_post_date = max(post_dates) if post_dates else None
    default_as_of = max(
        d for d in (bill_row.outcome_date, latest_sitting_date, latest_post_date) if d is not None
    )
    eval_date = as_of_date or default_as_of

    # Group sittings by BillStage
    sittings_by_state: dict[BillStage, list[StageSittingRow]] = {}
    first_sitting_by_state: dict[BillStage, date] = {}
    for s in sittings:
        mapped = STAGE_ID_TO_STATE.get(s.api_stage_id)
        if mapped is not None:
            sittings_by_state.setdefault(mapped, []).append(s)
            if (
                mapped not in first_sitting_by_state
                or s.event_date < first_sitting_by_state[mapped]
            ):
                first_sitting_by_state[mapped] = s.event_date

    ordered_parl_states = chain(bill_row.originating_house)

    # Compute full entry dates along the Parliamentary chain (handling skipped non-optional states)
    full_entry_dates: dict[BillStage, date] = {}
    prev_entry: date | None = None
    for i, st in enumerate(ordered_parl_states):
        entry = first_sitting_by_state.get(st)
        if entry is None and st not in OPTIONAL_STATES:
            later = [
                first_sitting_by_state[later_st]
                for later_st in ordered_parl_states[i + 1 :]
                if later_st in first_sitting_by_state
            ]
            if later:
                entry = min(later)
        if entry is not None:
            if prev_entry is not None and entry < prev_entry:
                entry = prev_entry
            full_entry_dates[st] = entry
            prev_entry = entry

    entered_parl_as_of = [
        st
        for st in ordered_parl_states
        if st in full_entry_dates and full_entry_dates[st] <= eval_date
    ]

    is_stopped_as_of = (
        bill_row.outcome in ("fell_session_end", "fell_dissolution", "withdrawn", "defeated")
        and bill_row.outcome_date is not None
        and bill_row.outcome_date <= eval_date
    )
    active_stop = (
        StopReason(bill_row.stop_reason) if (is_stopped_as_of and bill_row.stop_reason) else None
    )

    api_url = f"https://bills-api.parliament.uk/api/v1/Bills/{bill_row.bill_id}/Stages"
    web_url = f"https://bills.parliament.uk/bills/{bill_row.bill_id}"

    tracked_nodes: list[TrackedStageNode] = []
    current_stage: BillStage | None = None
    current_order = 0

    # ------------------------------------------------------------------
    # PHASE 1: BEFORE PARLIAMENT (Orders 1..6)
    # ------------------------------------------------------------------
    has_entered_parliament_as_of = len(entered_parl_as_of) > 0 or (first_reading_dt <= eval_date)
    if pre_domain is not None:
        pre_stage, pre_order, pre_nodes = pre_domain.evaluate_at_date(eval_date)
        if has_entered_parliament_as_of:
            for pn in pre_nodes:
                pn.status = StageStatus.COMPLETED
                if pn.completed_date is None:
                    pn.completed_date = first_reading_dt
        elif pre_stage is not None:
            current_stage = pre_stage
            current_order = pre_order
        tracked_nodes.extend(pre_nodes)
    else:
        for node in GOVERNMENT_BILL_PIPELINE:
            if has_entered_parliament_as_of:
                is_ready_stage = node.stage == BillStage.READY_FOR_FIRST_READING
                tracked_nodes.append(
                    TrackedStageNode(
                        node=node,
                        status=(
                            StageStatus.COMPLETED if is_ready_stage else StageStatus.FAST_TRACKED
                        ),
                        entered_date=first_reading_dt if is_ready_stage else None,
                        completed_date=first_reading_dt,
                        summary=(
                            f"Pre-legislative preparation completed prior to First Reading "
                            f"on {first_reading_dt.isoformat()}."
                            if is_ready_stage
                            else (
                                f"Fast-tracked / bypassed — No separate formal "
                                f"{node.label.lower()} stage recorded prior to First Reading "
                                f"on {first_reading_dt.isoformat()}."
                            )
                        ),
                        source_reference=web_url,
                        source_api_url=api_url,
                        web_url=web_url,
                        visible_events=(
                            [
                                ApiEvidenceEvent(
                                    event_date=first_reading_dt,
                                    title=f"Introduced in {bill_row.originating_house}",
                                    note=(
                                        f"{bill_row.short_title} cleared pre-legislative "
                                        f"preparation and entered {bill_row.originating_house}."
                                    ),
                                    source_api_url=api_url,
                                    web_url=web_url,
                                    document_type="parliament_introduction",
                                )
                            ]
                            if is_ready_stage
                            else []
                        ),
                        sitting_count=1 if is_ready_stage else 0,
                    )
                )
            else:
                tracked_nodes.append(TrackedStageNode(node=node, status=StageStatus.UPCOMING))

    # The very first signal is the date of the first Phase 1 stage (`signal_appears`).
    first_pre_node = tracked_nodes[0] if tracked_nodes else None
    if first_pre_node is not None:
        earliest_date = (
            first_pre_node.entered_date or first_pre_node.completed_date or earliest_date
        )

    # ------------------------------------------------------------------
    # PHASE 2: INSIDE PARLIAMENT (Orders 7..15) & ROYAL ASSENT (Order 16)
    # ------------------------------------------------------------------
    base_order = len(GOVERNMENT_BILL_PIPELINE)
    for idx, st in enumerate(ordered_parl_states, start=1):
        order = base_order + idx
        phase, label, description, stop_risks = PARLIAMENTARY_STAGE_INFO[st]
        node = StageNode(
            stage=st,
            phase=phase,
            order=order,
            label=label,
            description=description,
            stop_risks=stop_risks,
        )
        entry_dt = full_entry_dates.get(st)
        if entry_dt is None or entry_dt > eval_date:
            if (
                st == BillStage.PING_PONG
                and entry_dt is None
                and BillStage.ROYAL_ASSENT in entered_parl_as_of
            ):
                ra_dt = full_entry_dates.get(BillStage.ROYAL_ASSENT)
                tracked_nodes.append(
                    TrackedStageNode(
                        node=node,
                        status=StageStatus.SKIPPED,
                        entered_date=ra_dt,
                        completed_date=ra_dt,
                        summary=(
                            "0 ping-pong rounds — Second House agreed to the Bill "
                            "without exchanging amendments."
                        ),
                        source_reference=web_url,
                        source_api_url=api_url,
                        web_url=web_url,
                        visible_events=[],
                        sitting_count=0,
                    )
                )
            else:
                tracked_nodes.append(TrackedStageNode(node=node, status=StageStatus.UPCOMING))
            continue

        st_sittings = [s for s in sittings_by_state.get(st, []) if s.event_date <= eval_date]
        idx_in_entered = entered_parl_as_of.index(st)
        if len(st_sittings) == 0:
            # Stage had 0 formal sittings and was fast-tracked/dispensed with on entry_dt
            status = StageStatus.FAST_TRACKED
            completed_dt = entry_dt
            node_stop = None
        elif idx_in_entered + 1 < len(entered_parl_as_of):
            next_st = entered_parl_as_of[idx_in_entered + 1]
            status = StageStatus.COMPLETED
            completed_dt = full_entry_dates[next_st]
            node_stop = None
        elif st == BillStage.ROYAL_ASSENT:
            status = StageStatus.COMPLETED
            completed_dt = entry_dt
            current_stage = st
            current_order = order
            node_stop = None
        elif is_stopped_as_of:
            status = StageStatus.STOPPED
            completed_dt = max(bill_row.outcome_date or entry_dt, entry_dt)
            current_stage = st
            current_order = order
            node_stop = active_stop
        else:
            status = StageStatus.CURRENT
            completed_dt = None
            current_stage = st
            current_order = order
            node_stop = None

        visible_events = [
            ApiEvidenceEvent(
                event_date=s.event_date,
                title=(
                    f"{stage_names.get(s.api_stage_id, label)} "
                    f"({s.house or bill_row.originating_house})"
                ),
                note=(
                    f"Parliamentary stage sitting #{s.sitting_id} in Session {s.session_id} "
                    f"for {bill_row.short_title}."
                ),
                source_api_url=api_url,
                web_url=web_url,
                document_type="parliament_stage_sitting",
            )
            for s in st_sittings
        ]
        plural = "s" if len(st_sittings) != 1 else ""
        if status == StageStatus.FAST_TRACKED:
            summary_text = (
                f"Fast-tracked / bypassed on {entry_dt.isoformat()} "
                f"(0 sittings — {label} did not formally occur as the Bill "
                "advanced directly to the next stage)."
            )
        elif st == BillStage.PING_PONG:
            summary_text = (
                f"{len(st_sittings)} ping-pong round{plural} recorded "
                f"(entered {entry_dt.isoformat()}, ≤ {eval_date.isoformat()})."
            )
        else:
            summary_text = (
                f"Entered {label} on {entry_dt.isoformat()} "
                f"({len(st_sittings)} sitting{plural} recorded ≤ {eval_date.isoformat()})."
            )
        tracked_nodes.append(
            TrackedStageNode(
                node=node,
                status=status,
                entered_date=entry_dt,
                completed_date=completed_dt,
                summary=summary_text,
                source_reference=web_url,
                source_api_url=api_url,
                web_url=web_url,
                visible_events=visible_events,
                active_stop_reason=node_stop,
                sitting_count=len(st_sittings),
            )
        )

    # ------------------------------------------------------------------
    # PHASE 3: AFTER ROYAL ASSENT (Orders 17..22)
    # ------------------------------------------------------------------
    ra_date = full_entry_dates.get(BillStage.ROYAL_ASSENT)
    has_royal_assent_as_of = ra_date is not None and ra_date <= eval_date
    post_events_by_stage: dict[str, list[PostRoyalAssentEventRow]] = {}
    for pev in post_event_rows:
        post_events_by_stage.setdefault(pev.stage, []).append(pev)

    # Also enrich the Royal Assent node (Order 16) with legislation.gov.uk Act citation if present
    if has_royal_assent_as_of and act_comm is not None and tracked_nodes:
        ra_node = tracked_nodes[-1]
        if ra_node.node.stage == BillStage.ROYAL_ASSENT:
            act_web = f"https://www.legislation.gov.uk/ukpga/{act_comm.ukpga_year}/{act_comm.ukpga_number}/enacted"
            act_xml = f"{act_web}/data.xml"
            ra_node.source_reference = act_web
            ra_node.source_api_url = act_xml
            ra_node.web_url = act_web
            ra_node.summary = (
                f"Enacted as {act_comm.act_title} ({act_comm.ukpga_year} c. "
                f"{act_comm.ukpga_number}) on {act_comm.royal_assent_date.isoformat()}."
            )

    in_force_dt = fully_in_force_date(act_comm, in_force_status, ra_date)
    is_in_force_as_of = (
        has_royal_assent_as_of and in_force_dt is not None and in_force_dt <= eval_date
    )

    post_base_order = base_order + len(ordered_parl_states)
    for post_idx, (post_st, post_label, post_desc, post_stops) in enumerate(
        POST_ROYAL_ASSENT_STAGE_INFO, start=1
    ):
        order = post_base_order + post_idx
        node = StageNode(
            stage=post_st,
            phase=LifecyclePhase.AFTER_ROYAL_ASSENT,
            order=order,
            label=post_label,
            description=post_desc,
            stop_risks=post_stops,
        )
        if not has_royal_assent_as_of or ra_date is None:
            tracked_nodes.append(TrackedStageNode(node=node, status=StageStatus.UPCOMING))
            continue

        all_stage_events = post_events_by_stage.get(post_st.value, [])
        vis_stage_events = [e for e in all_stage_events if e.event_date <= eval_date]
        leg_base_url = (
            f"https://www.legislation.gov.uk/ukpga/{act_comm.ukpga_year}/{act_comm.ukpga_number}"
            if act_comm is not None
            else f"https://www.legislation.gov.uk/ukpga/{ra_date.year}"
        )
        leg_xml_url = f"{leg_base_url}/data.xml"

        if post_st == BillStage.SOME_SECTIONS_START:
            ev_models = [
                ApiEvidenceEvent(
                    event_date=e.event_date,
                    title=e.title,
                    note=e.note,
                    source_api_url=e.source_api_url,
                    web_url=e.web_url,
                    document_type=e.document_type,
                )
                for e in vis_stage_events
            ]
            # Completed once Commencement Orders, Regulations, or Law Fully in Force is entered
            has_later_post = is_in_force_as_of or any(
                e.event_date <= eval_date
                for k, evs in post_events_by_stage.items()
                if k not in (BillStage.ROYAL_ASSENT.value, BillStage.SOME_SECTIONS_START.value)
                for e in evs
            )
            st_status = StageStatus.COMPLETED if has_later_post else StageStatus.CURRENT
            tracked_nodes.append(
                TrackedStageNode(
                    node=node,
                    status=st_status,
                    entered_date=ra_date,
                    completed_date=ra_date if st_status == StageStatus.COMPLETED else None,
                    summary=(
                        "Initial citation, extent, and immediate commencement "
                        f"provisions took effect on {ra_date.isoformat()}."
                    ),
                    source_reference=leg_base_url,
                    source_api_url=leg_xml_url,
                    web_url=leg_base_url,
                    visible_events=ev_models,
                    sitting_count=len(ev_models) or 1,
                )
            )
            current_stage = post_st
            current_order = order

        elif post_st in (
            BillStage.COMMENCEMENT_ORDERS,
            BillStage.REGULATIONS_DRAFTED,
            BillStage.REGULATIONS_LAID,
        ):
            total_for_stage = len(all_stage_events)
            if total_for_stage == 0:
                if is_in_force_as_of:
                    zero_msg = {
                        BillStage.COMMENCEMENT_ORDERS: (
                            "0 commencement orders — Act commenced directly via its enactment "
                            "commencement clause without separate Commencement Order SIs."
                        ),
                        BillStage.REGULATIONS_DRAFTED: (
                            "0 draft affirmative regulations — No separate Draft Statutory "
                            "Instruments (UKDSI) published under this Act."
                        ),
                        BillStage.REGULATIONS_LAID: (
                            "0 secondary regulations — No separate Statutory Instruments (UKSI) "
                            "required under this Act."
                        ),
                    }[post_st]
                    tracked_nodes.append(
                        TrackedStageNode(
                            node=node,
                            status=StageStatus.SKIPPED,
                            entered_date=ra_date,
                            completed_date=ra_date,
                            summary=zero_msg,
                            source_reference=leg_base_url,
                            source_api_url=leg_xml_url,
                            web_url=leg_base_url,
                            visible_events=[],
                            sitting_count=0,
                        )
                    )
                else:
                    # Still upcoming (provisions remain to be commenced), but say that none
                    # exist yet so the front end can tell this apart from a bill before Royal
                    # Assent.
                    none_yet_msg = {
                        BillStage.COMMENCEMENT_ORDERS: (
                            "None made yet — No Commencement Order SIs under this Act on "
                            "legislation.gov.uk, and the Act is not yet fully in force."
                        ),
                        BillStage.REGULATIONS_DRAFTED: (
                            "None drafted yet — No Draft Statutory Instruments (UKDSI) under "
                            "this Act on legislation.gov.uk, and the Act is not yet fully in force."
                        ),
                        BillStage.REGULATIONS_LAID: (
                            "None laid yet — No Statutory Instruments (UKSI) under this Act on "
                            "legislation.gov.uk, and the Act is not yet fully in force."
                        ),
                    }[post_st]
                    tracked_nodes.append(
                        TrackedStageNode(
                            node=node,
                            status=StageStatus.UPCOMING,
                            summary=none_yet_msg,
                            source_reference=leg_base_url,
                            source_api_url=leg_xml_url,
                            web_url=leg_base_url,
                            visible_events=[],
                            sitting_count=0,
                        )
                    )
            elif not vis_stage_events:
                tracked_nodes.append(TrackedStageNode(node=node, status=StageStatus.UPCOMING))
            else:
                first_ev_dt = min(e.event_date for e in vis_stage_events)
                last_ev_dt = max(e.event_date for e in vis_stage_events)
                all_done_as_of = len(vis_stage_events) == total_for_stage
                st_status = StageStatus.COMPLETED if all_done_as_of else StageStatus.CURRENT
                ev_models = [
                    ApiEvidenceEvent(
                        event_date=e.event_date,
                        title=e.title,
                        note=e.note,
                        source_api_url=e.source_api_url,
                        web_url=e.web_url,
                        document_type=e.document_type,
                    )
                    for e in vis_stage_events
                ]
                noun = {
                    BillStage.COMMENCEMENT_ORDERS: "Commencement Order SI",
                    BillStage.REGULATIONS_DRAFTED: "Draft Statutory Instrument",
                    BillStage.REGULATIONS_LAID: "Statutory Instrument",
                }[post_st]
                plural = "s" if len(vis_stage_events) != 1 else ""
                tracked_nodes.append(
                    TrackedStageNode(
                        node=node,
                        status=st_status,
                        entered_date=first_ev_dt,
                        completed_date=last_ev_dt if all_done_as_of else None,
                        summary=(
                            f"{len(vis_stage_events)} of {total_for_stage} {noun}{plural} "
                            f"recorded on legislation.gov.uk ≤ {eval_date.isoformat()}."
                        ),
                        source_reference=vis_stage_events[-1].web_url,
                        source_api_url=vis_stage_events[-1].source_api_url,
                        web_url=vis_stage_events[-1].web_url,
                        visible_events=ev_models,
                        sitting_count=len(vis_stage_events),
                    )
                )
                current_stage = post_st
                current_order = order

        elif post_st == BillStage.LAW_FULLY_IN_FORCE:
            if is_in_force_as_of and in_force_dt is not None:
                ev_models = [
                    ApiEvidenceEvent(
                        event_date=e.event_date,
                        title=e.title,
                        note=e.note,
                        source_api_url=e.source_api_url,
                        web_url=e.web_url,
                        document_type=e.document_type,
                    )
                    for e in vis_stage_events
                ]
                comm_cnt = act_comm.commencement_orders_count if act_comm else 0
                summary_txt = (
                    f"All {comm_cnt} Commencement Order(s) in force as of "
                    f"{in_force_dt.isoformat()}."
                    if comm_cnt > 0
                    else (
                        f"Act commenced directly on {in_force_dt.isoformat()} "
                        "without separate Commencement Orders."
                    )
                )
                tracked_nodes.append(
                    TrackedStageNode(
                        node=node,
                        status=StageStatus.COMPLETED,
                        entered_date=in_force_dt,
                        completed_date=in_force_dt,
                        summary=summary_txt,
                        source_reference=leg_base_url,
                        source_api_url=leg_xml_url,
                        web_url=leg_base_url,
                        visible_events=ev_models,
                        sitting_count=len(ev_models) or 1,
                    )
                )
                current_stage = post_st
                current_order = order
            elif (
                has_royal_assent_as_of
                and in_force_status is not None
                and not in_force_status.fully_in_force
            ):
                pending = in_force_status.n_section_groups_prospective
                total = in_force_status.n_section_groups
                tracked_nodes.append(
                    TrackedStageNode(
                        node=node,
                        status=StageStatus.CURRENT,
                        summary=(
                            f"Not fully in force: {pending} of {total} sections still "
                            "prospective (not yet in force) on legislation.gov.uk, checked "
                            f"{in_force_status.checked_at.date().isoformat()}."
                        ),
                        source_reference=in_force_status.source_url,
                        source_api_url=f"{in_force_status.source_url}/data.xml",
                        web_url=in_force_status.source_url,
                        visible_events=[],
                        sitting_count=0,
                    )
                )
            else:
                tracked_nodes.append(TrackedStageNode(node=node, status=StageStatus.UPCOMING))

    # Build dynamic database-driven milestone dates across all phases.
    # Milestones on the same date are merged into one entry (one button per date).
    milestone_parts: dict[date, list[tuple[str, str]]] = {}

    def add_milestone(dt: date | None, label: str, note: str) -> None:
        if dt is not None:
            milestone_parts.setdefault(dt, []).append((label, note))

    if pre_domain is not None:
        for node in GOVERNMENT_BILL_PIPELINE:
            rec = pre_domain.stage_records.get(node.stage)
            if rec is not None:
                add_milestone(rec.entered_date, f"1. {node.label}", rec.headline)
    else:
        signal_node = GOVERNMENT_BILL_PIPELINE[0]
        add_milestone(
            earliest_date,
            f"1. {signal_node.label}",
            "First signal: no earlier pre-parliamentary record, so the first signal "
            "is the bill's introduction to Parliament.",
        )
    for st in ordered_parl_states:
        if st in full_entry_dates:
            phase, label, _, _ = PARLIAMENTARY_STAGE_INFO[st]
            phase_num = "3" if phase == LifecyclePhase.AFTER_ROYAL_ASSENT else "2"
            add_milestone(full_entry_dates[st], f"{phase_num}. {label}", label)
    add_milestone(
        bill_row.outcome_date,
        f"Outcome: {bill_row.outcome}",
        f"Final outcome: {bill_row.outcome}",
    )
    if act_comm is not None:
        for post_dt, post_lbl in (
            (act_comm.first_commencement_order_date, "3. 1st Commencement Order"),
            (act_comm.first_draft_regulation_date, "3. 1st Draft Regulation"),
            (act_comm.first_laid_regulation_date, "3. 1st Laid Regulation"),
            (act_comm.law_fully_in_force_date, "3. Law in Force"),
        ):
            add_milestone(post_dt, post_lbl, f"{act_comm.act_title}: {post_lbl}")

    milestones: list[dict[str, Any]] = []
    for dt in sorted(milestone_parts):
        parts = list(dict.fromkeys(milestone_parts[dt]))
        milestones.append(
            {
                "label": f"{dt.isoformat()} ({' · '.join(lbl for lbl, _ in parts)})",
                "date": dt,
                "note": "; ".join(note for _, note in parts),
            }
        )

    if has_royal_assent_as_of:
        current_phase = LifecyclePhase.AFTER_ROYAL_ASSENT
    elif has_entered_parliament_as_of:
        current_phase = LifecyclePhase.INSIDE_PARLIAMENT
    else:
        current_phase = LifecyclePhase.BEFORE_PARLIAMENT

    total_stages = len(tracked_nodes)

    # Full text: the Act on legislation.gov.uk once it has Royal Assent (as of the date),
    # otherwise the bill's published versions on bills.parliament.uk
    if has_royal_assent_as_of and act_comm is not None:
        full_text_url = (
            f"https://www.legislation.gov.uk/ukpga/{act_comm.ukpga_year}/{act_comm.ukpga_number}"
        )
        full_text_label = "Read the full Act on legislation.gov.uk"
    else:
        full_text_url = f"https://bills.parliament.uk/bills/{bill_row.bill_id}/publications"
        full_text_label = "Read the full bill text on bills.parliament.uk"

    return {
        "id": str(bill_row.bill_id),
        "parliament_bill_id": bill_row.bill_id,
        "short_title": bill_row.short_title,
        "long_title": (
            pre_domain.long_title
            if pre_domain is not None
            else (
                f"UK Government Bill #{bill_row.bill_id} originating in the House of "
                f"{bill_row.originating_house} (Session {bill_row.introduced_session_id})."
            )
        ),
        "sponsoring_department": (
            bill_row.department
            or (pre_domain.sponsoring_department if pre_domain else "HM Government")
        ),
        "originating_house": bill_row.originating_house,
        "introduced_session_id": bill_row.introduced_session_id,
        "outcome": bill_row.outcome,
        "outcome_date": bill_row.outcome_date,
        "stop_reason": bill_row.stop_reason,
        "as_of_date": eval_date,
        "earliest_signal_date": earliest_date,
        "first_reading_date": first_reading_dt,
        "current_phase": current_phase,
        "current_stage": current_stage,
        "current_stage_order": current_order,
        "total_stages": total_stages,
        "tracked_nodes": tracked_nodes,
        "milestone_dates": milestones,
        "full_text_url": full_text_url,
        "full_text_label": full_text_label,
    }
