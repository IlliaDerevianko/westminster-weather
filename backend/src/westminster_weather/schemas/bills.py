"""Pydantic schemas for Phase 1 (Before Parliament) Government Bill tracking."""

from datetime import date

from pydantic import BaseModel, Field

from westminster_weather.domain.bills import StageStatus
from westminster_weather.domain.stages import BillStage, LifecyclePhase, StopReason


class ApiEvidenceEventSchema(BaseModel):
    event_date: date
    title: str
    note: str
    source_api_url: str
    web_url: str
    document_type: str


class StageNodeSchema(BaseModel):
    stage: BillStage
    phase: LifecyclePhase
    order: int
    label: str
    description: str
    stop_risks: list[StopReason]


class TrackedStageNodeSchema(StageNodeSchema):
    status: StageStatus
    entered_date: date | None = None
    completed_date: date | None = None
    summary: str | None = None
    source_reference: str | None = None
    source_api_url: str | None = None
    web_url: str | None = None
    visible_events: list[ApiEvidenceEventSchema] = []
    active_stop_reason: StopReason | None = None
    sitting_count: int = 0


class MilestoneDateSchema(BaseModel):
    label: str
    date: date
    note: str


class BillSummarySchema(BaseModel):
    bill_id: int
    short_title: str
    originating_house: str
    introduced_session_id: int
    earliest_signal_date: date | None = None
    introduced_date: date | None = None
    department: str | None = None
    outcome: str
    outcome_date: date | None = None
    stop_reason: str | None = None
    is_act: bool
    is_defeated: bool
    withdrawn_date: date | None = None
    has_pre_parliament_data: bool = False
    latest_stage: BillStage | None = Field(
        None, description="Acts only: furthest stage completed after Royal Assent, as of today"
    )
    latest_stage_label: str | None = None


class GovernmentBillTrackingResponse(BaseModel):
    id: str
    parliament_bill_id: int
    short_title: str
    long_title: str
    sponsoring_department: str
    originating_house: str = "Commons"
    introduced_session_id: int | None = None
    outcome: str = "in_progress"
    outcome_date: date | None = None
    stop_reason: str | None = None
    as_of_date: date
    earliest_signal_date: date
    first_reading_date: date
    current_phase: LifecyclePhase
    current_stage: BillStage | None = None
    current_stage_order: int
    total_stages: int
    stages: list[TrackedStageNodeSchema]
    milestone_dates: list[MilestoneDateSchema] = []
    full_text_url: str | None = Field(
        None, description="Act on legislation.gov.uk after Royal Assent, else bill texts on "
        "bills.parliament.uk"
    )
    full_text_label: str | None = None


class TimelineQuantiles(BaseModel):
    p10_days: int = Field(..., description="10th percentile (optimistic) days from as_of")
    p50_days: int = Field(..., description="50th percentile (median) days from as_of")
    p90_days: int = Field(..., description="90th percentile (conservative) days from as_of")
    estimated_median_date: date | None = None
    p10_date: date | None = None
    p90_date: date | None = None


class StageForecastSchema(BaseModel):
    """Per remaining state (DATA_CONTRACT §6.3)."""

    model_state: BillStage
    p_reach: float = Field(..., ge=0.0, le=1.0, description="Probability the bill reaches it")
    p_fail_defeated: float = Field(..., ge=0.0, le=1.0)
    p_fail_withdrawn: float = Field(..., ge=0.0, le=1.0)
    p_fail_session_end: float = Field(..., ge=0.0, le=1.0)
    p_fail_dissolution: float = Field(..., ge=0.0, le=1.0)
    entry_date_p50: date | None = Field(
        None, description="Median date of reaching the state, among paths that reach it"
    )


class BillForecastResponse(BaseModel):
    """Passage forecast for one bill on ``as_of`` (MODEL_SPEC §5.1)."""

    bill_id: int
    short_title: str
    as_of: date
    current_phase: LifecyclePhase
    current_stage: BillStage | None = None
    passage_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability that the Government Bill reaches Royal Assent"
    )
    p_ra_before_session_end: float = Field(..., ge=0.0, le=1.0)
    commencement_probability: float | None = Field(
        None,
        description="Stage 3: probability that the Act's law applies on the day of Royal Assent "
        "(no commencement order needed); 1 or 0 once known; null before Royal Assent",
    )
    time_to_royal_assent: TimelineQuantiles | None = None
    time_to_commencement: TimelineQuantiles | None = Field(
        None,
        description="Stage 3: days from as_of until the law applies (first commencement order, "
        "or the day its sections started); negative once applied; null before Royal Assent",
    )
    primary_stop_risks: dict[StopReason, float] = Field(
        default_factory=dict,
        description="Probability distribution over potential stop states",
    )
    stages: list[StageForecastSchema] | None = None
    model_variant: str = Field(..., description="'train' (backtest) or 'live' model set")
    n_paths: int
