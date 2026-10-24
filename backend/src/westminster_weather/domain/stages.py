"""Domain model for the 3-phase UK Government Bill lifecycle and stop states."""

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

    # 2. Inside Parliament
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
    SOME_SECTIONS_START = "some_sections_start"
    COMMENCEMENT_ORDERS = "commencement_orders"
    REGULATIONS_DRAFTED = "regulations_drafted"
    REGULATIONS_LAID = "regulations_laid"
    LAW_FULLY_IN_FORCE = "law_fully_in_force"
    AMENDED_OR_REPEALED = "amended_or_repealed"


class StopReason(StrEnum):
    # 1. Before Parliament stops
    NEVER_ADOPTED = "never_adopted"
    SHELVED = "shelved"
    STUCK_IN_DRAFTING = "stuck_in_drafting"
    NO_SLOT_THIS_SESSION = "no_slot_this_session"

    # 2. Inside Parliament stops
    DEFEATED = "defeated"
    SESSION_RUNS_OUT = "session_runs_out"
    PAUSED_OR_WITHDRAWN = "paused_or_withdrawn"
    ELECTION_CALLED = "election_called"
    DEADLOCK = "deadlock"

    # 3. After Royal Assent stops
    NEVER_COMMENCED = "never_commenced"
    ANNULLED = "annulled"


@dataclass(frozen=True)
class StageNode:
    stage: BillStage
    phase: LifecyclePhase
    order: int
    label: str
    description: str
    stop_risks: tuple[StopReason, ...]


GOVERNMENT_BILL_PIPELINE: tuple[StageNode, ...] = (
    # Phase 1: Before Parliament
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
        description="Continues in Inside Parliament",
        stop_risks=(),
    ),
    # Phase 2: Inside Parliament
    StageNode(
        stage=BillStage.COMMONS_FIRST_READING,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=7,
        label="Commons 1st reading",
        description="Title read out, bill published",
        stop_risks=(StopReason.SESSION_RUNS_OUT, StopReason.ELECTION_CALLED),
    ),
    StageNode(
        stage=BillStage.COMMONS_SECOND_READING,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=8,
        label="Commons 2nd reading",
        description="Debate on principle, then vote",
        stop_risks=(
            StopReason.DEFEATED,
            StopReason.SESSION_RUNS_OUT,
            StopReason.ELECTION_CALLED,
        ),
    ),
    StageNode(
        stage=BillStage.PROGRAMME_AND_MONEY_MOTIONS,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=9,
        label="Programme and money motions",
        description="Timetable and spending agreed",
        stop_risks=(StopReason.SESSION_RUNS_OUT, StopReason.ELECTION_CALLED),
    ),
    StageNode(
        stage=BillStage.COMMONS_COMMITTEE,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=10,
        label="Commons committee",
        description="Evidence, line-by-line amendments",
        stop_risks=(StopReason.SESSION_RUNS_OUT, StopReason.ELECTION_CALLED),
    ),
    StageNode(
        stage=BillStage.COMMONS_REPORT_STAGE,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=11,
        label="Commons report stage",
        description="Whole House, rebel amendments",
        stop_risks=(
            StopReason.PAUSED_OR_WITHDRAWN,
            StopReason.SESSION_RUNS_OUT,
            StopReason.ELECTION_CALLED,
        ),
    ),
    StageNode(
        stage=BillStage.COMMONS_THIRD_READING,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=12,
        label="Commons 3rd reading",
        description="Final vote, sent to the Lords",
        stop_risks=(StopReason.SESSION_RUNS_OUT, StopReason.ELECTION_CALLED),
    ),
    StageNode(
        stage=BillStage.LORDS_FIRST_AND_SECOND_READING,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=13,
        label="Lords 1st and 2nd reading",
        description="Debate on principle",
        stop_risks=(StopReason.SESSION_RUNS_OUT, StopReason.ELECTION_CALLED),
    ),
    StageNode(
        stage=BillStage.LORDS_COMMITTEE,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=14,
        label="Lords committee",
        description="Whole House, detailed amendments",
        stop_risks=(StopReason.SESSION_RUNS_OUT, StopReason.ELECTION_CALLED),
    ),
    StageNode(
        stage=BillStage.LORDS_REPORT_AND_THIRD_READING,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=15,
        label="Lords report and 3rd reading",
        description="Where government defeats happen",
        stop_risks=(StopReason.SESSION_RUNS_OUT, StopReason.ELECTION_CALLED),
    ),
    StageNode(
        stage=BillStage.PING_PONG,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=16,
        label="Ping-pong",
        description="Houses trade amendments back and forth",
        stop_risks=(
            StopReason.DEADLOCK,
            StopReason.SESSION_RUNS_OUT,
            StopReason.ELECTION_CALLED,
        ),
    ),
    StageNode(
        stage=BillStage.BOTH_HOUSES_AGREE,
        phase=LifecyclePhase.INSIDE_PARLIAMENT,
        order=17,
        label="Both Houses agree",
        description="Continues to Royal Assent",
        stop_risks=(StopReason.SESSION_RUNS_OUT, StopReason.ELECTION_CALLED),
    ),
    # Phase 3: After Royal Assent
    StageNode(
        stage=BillStage.ROYAL_ASSENT,
        phase=LifecyclePhase.AFTER_ROYAL_ASSENT,
        order=18,
        label="Royal Assent",
        description="Bill becomes an Act",
        stop_risks=(),
    ),
    StageNode(
        stage=BillStage.SOME_SECTIONS_START,
        phase=LifecyclePhase.AFTER_ROYAL_ASSENT,
        order=19,
        label="Some sections start",
        description="On Assent or on a set date",
        stop_risks=(),
    ),
    StageNode(
        stage=BillStage.COMMENCEMENT_ORDERS,
        phase=LifecyclePhase.AFTER_ROYAL_ASSENT,
        order=20,
        label="Commencement orders",
        description="Minister switches on the rest",
        stop_risks=(StopReason.NEVER_COMMENCED,),
    ),
    StageNode(
        stage=BillStage.REGULATIONS_DRAFTED,
        phase=LifecyclePhase.AFTER_ROYAL_ASSENT,
        order=21,
        label="Regulations drafted",
        description="Secondary legislation fills detail",
        stop_risks=(),
    ),
    StageNode(
        stage=BillStage.REGULATIONS_LAID,
        phase=LifecyclePhase.AFTER_ROYAL_ASSENT,
        order=22,
        label="Regulations laid",
        description="Parliament has about 40 days",
        stop_risks=(StopReason.ANNULLED,),
    ),
    StageNode(
        stage=BillStage.LAW_FULLY_IN_FORCE,
        phase=LifecyclePhase.AFTER_ROYAL_ASSENT,
        order=23,
        label="Law fully in force",
        description="Applies to everyone",
        stop_risks=(),
    ),
    StageNode(
        stage=BillStage.AMENDED_OR_REPEALED,
        phase=LifecyclePhase.AFTER_ROYAL_ASSENT,
        order=24,
        label="Amended or repealed",
        description="Changed by later laws",
        stop_risks=(),
    ),
)
