"""Domain model for a tracked UK Government Bill and its evolution across all 24 stages."""

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from westminster_weather.domain.stages import (
    GOVERNMENT_BILL_PIPELINE,
    BillStage,
    LifecyclePhase,
    StageNode,
    StopReason,
)


class StageStatus(StrEnum):
    COMPLETED = "completed"
    CURRENT = "current"
    UPCOMING = "upcoming"
    STOPPED = "stopped"


@dataclass
class StageHistoryEntry:
    stage: BillStage
    entered_date: date
    completed_date: date | None = None
    summary: str = ""
    source_reference: str | None = None


@dataclass
class TrackedStageNode:
    node: StageNode
    status: StageStatus
    entered_date: date | None = None
    completed_date: date | None = None
    summary: str | None = None
    source_reference: str | None = None
    active_stop_reason: StopReason | None = None


@dataclass
class GovernmentBill:
    """Represents a UK Government Bill and its current position in the 24-stage pipeline."""

    id: str
    short_title: str
    long_title: str
    sponsoring_department: str
    simulated_as_of_date: date
    current_stage: BillStage
    stopped_reason: StopReason | None = None
    passage_probability: float = 0.0
    estimated_months_to_commencement: int | None = None
    history: dict[BillStage, StageHistoryEntry] = field(default_factory=dict)

    @property
    def current_node(self) -> StageNode:
        return next(n for n in GOVERNMENT_BILL_PIPELINE if n.stage == self.current_stage)

    @property
    def current_phase(self) -> LifecyclePhase:
        return self.current_node.phase

    def build_tracked_pipeline(self) -> list[TrackedStageNode]:
        """Dynamically map all 24 pipeline stages with this bill's progression state."""
        current_order = self.current_node.order
        tracked: list[TrackedStageNode] = []

        for node in GOVERNMENT_BILL_PIPELINE:
            entry = self.history.get(node.stage)
            if node.order < current_order:
                status = StageStatus.COMPLETED
            elif node.order == current_order:
                status = StageStatus.STOPPED if self.stopped_reason else StageStatus.CURRENT
            else:
                status = StageStatus.UPCOMING

            tracked.append(
                TrackedStageNode(
                    node=node,
                    status=status,
                    entered_date=entry.entered_date if entry else None,
                    completed_date=entry.completed_date if entry else None,
                    summary=entry.summary if entry else None,
                    source_reference=entry.source_reference if entry else None,
                    active_stop_reason=(
                        self.stopped_reason if node.order == current_order else None
                    ),
                )
            )
        return tracked


# Real historical UK Government Bill simulated in its Pre-Parliament ("Before Parliament") phase:
# The Online Safety Bill (which later became the Online Safety Act 2023), viewed as of May 2021
# while in the "Drafting" stage (Order 4 of 24).
ONLINE_SAFETY_BILL_PRE_PARLIAMENT = GovernmentBill(
    id="online-safety-bill",
    short_title="Online Safety Bill",
    long_title=(
        "A Bill to make provision for and in connection with the regulation by OFCOM of "
        "certain internet services through a statutory duty of care."
    ),
    sponsoring_department=(
        "Department for Digital, Culture, Media & Sport (DCMS) & Home Office"
    ),
    simulated_as_of_date=date(2021, 5, 12),
    current_stage=BillStage.DRAFTING,
    stopped_reason=None,
    passage_probability=0.78,
    estimated_months_to_commencement=34,
    history={
        BillStage.SIGNAL_APPEARS: StageHistoryEntry(
            stage=BillStage.SIGNAL_APPEARS,
            entered_date=date(2017, 10, 11),
            completed_date=date(2018, 5, 19),
            summary=(
                "UK Government publishes the Internet Safety Strategy Green Paper following "
                "growing public, inquiry, and campaign pressure over online harms."
            ),
            source_reference="GOV.UK: Internet Safety Strategy Green Paper (Oct 2017)",
        ),
        BillStage.GOVERNMENT_COMMITS: StageHistoryEntry(
            stage=BillStage.GOVERNMENT_COMMITS,
            entered_date=date(2018, 5, 20),
            completed_date=date(2019, 4, 7),
            summary=(
                "Ministers at DCMS and the Home Office formally commit to bringing forward "
                "primary legislation establishing a statutory duty of care on platforms."
            ),
            source_reference="Government Response to Internet Safety Strategy (May 2018)",
        ),
        BillStage.CONSULTATION: StageHistoryEntry(
            stage=BillStage.CONSULTATION,
            entered_date=date(2019, 4, 8),
            completed_date=date(2020, 12, 15),
            summary=(
                "Online Harms White Paper (CP 57) published for public consultation; "
                "Full Government Response (CP 354) confirms Ofcom as regulator."
            ),
            source_reference="Online Harms White Paper (CP 57) & Full Response (CP 354)",
        ),
        BillStage.DRAFTING: StageHistoryEntry(
            stage=BillStage.DRAFTING,
            entered_date=date(2020, 12, 16),
            completed_date=None,
            summary=(
                "Office of the Parliamentary Counsel and DCMS legal team drafting the "
                "Draft Online Safety Bill (CP 405). Navigating complex definitions around "
                "'legal but harmful' content and end-to-end encryption."
            ),
            source_reference="Draft Online Safety Bill (CP 405, May 2021)",
        ),
    },
)

BILLS_STORE: dict[str, GovernmentBill] = {
    ONLINE_SAFETY_BILL_PRE_PARLIAMENT.id: ONLINE_SAFETY_BILL_PRE_PARLIAMENT,
}
