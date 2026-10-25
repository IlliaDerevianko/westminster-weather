"""BillOutlook: presentation-ready outputs computed from the passage model's forecast."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

WeatherStatus = Literal["sunny", "partly_cloudy", "rain", "storm"]


class OutlookDistribution(BaseModel):
    method: Literal["simulated_paths"] = Field(
        "simulated_paths",
        description="Counted from the model's simulated paths (no quantile interpolation)",
    )
    n_paths: int
    p_royal_assent: float = Field(..., description="Chance the bill ever becomes law")
    p_fail: float = Field(
        ..., description="Chance the bill dies (defeat, withdrawal, session end, election)"
    )
    royal_assent_date_quantiles: dict[str, date | None] = Field(
        ..., description="q05..q95: Royal Assent date quantiles among paths that pass"
    )


class OutlookDeadline(BaseModel):
    kind: str = Field(
        ...,
        description="christmas_recess, easter_recess, summer_recess, quarter_end_1..4, "
        "session_end or custom",
    )
    label: str
    date: date
    estimated: bool = Field(..., description="Date estimated from past years (not announced)")
    p: float = Field(..., description="Chance of Royal Assent on or before the date")
    sentence: str


class OutlookWeather(BaseModel):
    status: WeatherStatus
    label: str
    target_date: date
    target_label: str
    target_estimated: bool
    p_by_target: float
    p_fail: float
    headline: str


class CurvePoint(BaseModel):
    date: date
    p_law_by_date: float


class OutlookCurve(BaseModel):
    points: list[CurvePoint]
    month_50: date | None = Field(None, description="First month the chance passes 50%")
    month_90: date | None = Field(None, description="First month the chance passes 90%")
    sentence: str


class OutlookPace(BaseModel):
    current_stage: str
    current_stage_name: str
    sitting_days_so_far: int
    typical_sitting_days: float | None
    weeks_ahead_or_behind: int | None = Field(None, description="Positive = behind")
    pace_percentile: float | None = Field(None, description="Share of past bills that were faster")
    n_comparison_bills: int
    sentence: str


class OutlookNextMilestone(BaseModel):
    stage: str
    stage_name: str
    date: date
    date_kind: Literal["scheduled", "expected"]
    p_by_target_without: float | None = None
    p_by_target_with: float | None = None
    sentence: str
    what_if_sentence: str | None = None


class BillOutlook(BaseModel):
    """One bill's outlook on ``forecast_date``; every ``sentence``/``headline`` is ready to
    show as-is."""

    bill_id: int
    short_title: str
    generated_at: datetime
    forecast_date: date
    model_version: str
    recalibrated: bool = Field(..., description="Deadline chances isotonic-recalibrated")
    current_stage: str
    current_stage_name: str
    distribution: OutlookDistribution
    deadlines: list[OutlookDeadline]
    weather: OutlookWeather
    curve: OutlookCurve
    pace: OutlookPace
    next_milestone: OutlookNextMilestone | None
    sentences: list[str] = Field(..., description="All sentences in display order")
