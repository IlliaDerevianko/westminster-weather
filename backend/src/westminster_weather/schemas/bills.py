"""Pydantic schemas for Phase 1 (Before Parliament) Government Bill tracking."""

from datetime import date

from pydantic import BaseModel

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
    introduced_date: date | None = None
    department: str | None = None
    outcome: str
    outcome_date: date | None = None
    stop_reason: str | None = None
    is_act: bool
    is_defeated: bool
    withdrawn_date: date | None = None
    has_pre_parliament_data: bool = False


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

