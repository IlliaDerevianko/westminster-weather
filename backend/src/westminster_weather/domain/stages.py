"""Domain model for Phase 1 (Before Parliament) of a UK Government Bill lifecycle."""

from dataclasses import dataclass
from enum import StrEnum


class LifecyclePhase(StrEnum):
    BEFORE_PARLIAMENT = "before_parliament"


class BillStage(StrEnum):
    SIGNAL_APPEARS = "signal_appears"
    GOVERNMENT_COMMITS = "government_commits"
    CONSULTATION = "consultation"
    DRAFTING = "drafting"
    SLOT_IN_PROGRAMME = "slot_in_programme"
    READY_FOR_FIRST_READING = "ready_for_first_reading"


class StopReason(StrEnum):
    NEVER_ADOPTED = "never_adopted"
    SHELVED = "shelved"
    STUCK_IN_DRAFTING = "stuck_in_drafting"
    NO_SLOT_THIS_SESSION = "no_slot_this_session"


@dataclass(frozen=True)
class StageNode:
    stage: BillStage
    phase: LifecyclePhase
    order: int
    label: str
    description: str
    stop_risks: tuple[StopReason, ...]


GOVERNMENT_BILL_PIPELINE: tuple[StageNode, ...] = (
    StageNode(
        stage=BillStage.SIGNAL_APPEARS,
        phase=LifecyclePhase.BEFORE_PARLIAMENT,
        order=1,
        label="Signal appears",
        description="Manifesto, crisis, petition, report",
        stop_risks=(),
    ),
    StageNode(
        stage=BillStage.GOVERNMENT_COMMITS,
        phase=LifecyclePhase.BEFORE_PARLIAMENT,
        order=2,
        label="Government commits",
        description='Minister says "we will legislate"',
        stop_risks=(StopReason.NEVER_ADOPTED,),
    ),
    StageNode(
        stage=BillStage.CONSULTATION,
        phase=LifecyclePhase.BEFORE_PARLIAMENT,
        order=3,
        label="Consultation",
        description="Green or white paper, then response",
        stop_risks=(StopReason.SHELVED,),
    ),
    StageNode(
        stage=BillStage.DRAFTING,
        phase=LifecyclePhase.BEFORE_PARLIAMENT,
        order=4,
        label="Drafting",
        description="Government lawyers write the bill",
        stop_risks=(StopReason.STUCK_IN_DRAFTING,),
    ),
    StageNode(
        stage=BillStage.SLOT_IN_PROGRAMME,
        phase=LifecyclePhase.BEFORE_PARLIAMENT,
        order=5,
        label="Slot in the programme",
        description="Cabinet agrees, King's Speech",
        stop_risks=(StopReason.NO_SLOT_THIS_SESSION,),
    ),
    StageNode(
        stage=BillStage.READY_FOR_FIRST_READING,
        phase=LifecyclePhase.BEFORE_PARLIAMENT,
        order=6,
        label="Ready for first reading",
        description="Bill ready for formal introduction in Parliament",
        stop_risks=(),
    ),
)
