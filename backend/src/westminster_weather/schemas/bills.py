"""Pydantic schemas for Government Bill lifecycle, passage probability, and timelines."""

from datetime import date

from pydantic import BaseModel, Field

from westminster_weather.domain.bills import StageStatus
from westminster_weather.domain.stages import BillStage, LifecyclePhase, StopReason


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
    active_stop_reason: StopReason | None = None


class GovernmentBillTrackingResponse(BaseModel):
    id: str
    short_title: str
    long_title: str
    sponsoring_department: str
    simulated_as_of_date: date
    current_phase: LifecyclePhase
    current_stage: BillStage
    current_stage_order: int
    total_stages: int
    stopped_reason: StopReason | None = None
    passage_probability: float = Field(..., ge=0.0, le=1.0)
    estimated_months_to_commencement: int | None = None
    stages: list[TrackedStageNodeSchema]


class UpdateBillStageRequest(BaseModel):
    current_stage: BillStage
    stopped_reason: StopReason | None = None


class TimelineQuantiles(BaseModel):
    p10_days: int = Field(..., description="10th percentile (optimistic) days from now")
    p50_days: int = Field(..., description="50th percentile (median) days from now")
    p90_days: int = Field(..., description="90th percentile (conservative) days from now")
    estimated_median_date: date | None = None


class BillForecastResponse(BaseModel):
    bill_id: int
    short_title: str
    current_phase: LifecyclePhase
    current_stage: BillStage
    passage_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability that the Government Bill reaches Royal Assent"
    )
    commencement_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the Bill reaches Royal Assent and is commenced into force",
    )
    time_to_royal_assent: TimelineQuantiles | None = None
    time_to_commencement: TimelineQuantiles | None = None
    primary_stop_risks: dict[StopReason, float] = Field(
        default_factory=dict,
        description="Probability distribution over potential stop states",
    )
