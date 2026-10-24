"""Domain model for Phase 1 (Before Parliament) of a UK Government Bill lifecycle."""

from dataclasses import dataclass
from enum import StrEnum


class LifecyclePhase(StrEnum):
    BEFORE_PARLIAMENT = "before_parliament"
    INSIDE_PARLIAMENT = "inside_parliament"
    AFTER_ROYAL_ASSENT = "after_royal_assent"


class BillStage(StrEnum):
    # 1. Before Parliament
    SIGNAL_APPEARS = "signal_appears"
    GOVERNMENT_COMMITS = "government_commits"
    CONSULTATION = "consultation"
    DRAFTING = "drafting"
    SLOT_IN_PROGRAMME = "slot_in_programme"
    READY_FOR_FIRST_READING = "ready_for_first_reading"

    # 2. Inside Parliament (MODEL_SPEC §2.1, DATA_CONTRACT §2.5)
    COMMONS_FIRST_READING = "commons_1st_reading"
    COMMONS_SECOND_READING = "commons_2nd_reading"
    PROGRAMME_AND_MONEY_MOTIONS = "programme_and_money_motions"
    COMMONS_COMMITTEE = "commons_committee"
    COMMONS_REPORT_STAGE = "commons_report_stage"
    COMMONS_THIRD_READING = "commons_3rd_reading"
    LORDS_FIRST_AND_SECOND_READING = "lords_1st_and_2nd_reading"
    LORDS_COMMITTEE = "lords_committee"
    LORDS_REPORT_AND_THIRD_READING = "lords_report_and_3rd_reading"
    PING_PONG = "ping_pong"
    BOTH_HOUSES_AGREE = "both_houses_agree"

    # 3. After Royal Assent
    ROYAL_ASSENT = "royal_assent"


class StopReason(StrEnum):
    # 1. Before Parliament stops
    NEVER_ADOPTED = "never_adopted"
    SHELVED = "shelved"
    STUCK_IN_DRAFTING = "stuck_in_drafting"
    NO_SLOT_THIS_SESSION = "no_slot_this_session"

    # 2. Inside Parliament stops (DATA_CONTRACT §3.1)
    DEFEATED = "defeated"
    SESSION_RUNS_OUT = "session_runs_out"
    PAUSED_OR_WITHDRAWN = "paused_or_withdrawn"
    ELECTION_CALLED = "election_called"
    DEADLOCK = "deadlock"



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
