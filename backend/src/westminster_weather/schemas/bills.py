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


class GovernmentBillTrackingResponse(BaseModel):
    id: str
    parliament_bill_id: int
    short_title: str
    long_title: str
    sponsoring_department: str
    as_of_date: date
    earliest_signal_date: date
    first_reading_date: date
    current_phase: LifecyclePhase
    current_stage: BillStage | None = None
    current_stage_order: int
    total_stages: int
    stages: list[TrackedStageNodeSchema]
