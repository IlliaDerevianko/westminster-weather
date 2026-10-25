"""FastAPI application entrypoint for Westminster Weather (backed by Neon Lakebase Postgres)."""

from datetime import date
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from westminster_weather import __version__
from westminster_weather.config import settings
from westminster_weather.db import check_db_connection
from westminster_weather.domain.bills import (
    TrackedStageNode,
    evaluate_parliament_bill_from_db,
    list_all_bills_from_db,
)
from westminster_weather.domain.stages import (
    GOVERNMENT_BILL_PIPELINE,
)
from westminster_weather.schemas.bills import (
    ApiEvidenceEventSchema,
    BillForecastResponse,
    BillSummarySchema,
    GovernmentBillTrackingResponse,
    MilestoneDateSchema,
    StageNodeSchema,
    TrackedStageNodeSchema,
)
from westminster_weather.schemas.outlook import BillOutlook

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=(
        "Tracks UK Government Bills across Phase 1 (Before Parliament) using real "
        "historical data persisted in Neon Lakebase Postgres."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def _serialize_tracked_nodes(
    tracked_nodes: list[TrackedStageNode],
) -> list[TrackedStageNodeSchema]:
    return [
        TrackedStageNodeSchema(
            stage=t.node.stage,
            phase=t.node.phase,
            order=t.node.order,
            label=t.node.label,
            description=t.node.description,
            stop_risks=list(t.node.stop_risks),
            status=t.status,
            entered_date=t.entered_date,
            completed_date=t.completed_date,
            summary=t.summary,
            source_reference=t.source_reference,
            source_api_url=t.source_api_url,
            web_url=t.web_url,
            visible_events=[
                ApiEvidenceEventSchema(
                    event_date=e.event_date,
                    title=e.title,
                    note=e.note,
                    source_api_url=e.source_api_url,
                    web_url=e.web_url,
                    document_type=e.document_type,
                )
                for e in t.visible_events
            ],
            active_stop_reason=t.active_stop_reason,
            sitting_count=t.sitting_count,
        )
        for t in tracked_nodes
    ]


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    pg_version = await check_db_connection()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": __version__,
        "database": "connected",
        "neon_branch": settings.neon_branch,
        "postgres_version": pg_version.split(",")[0],
    }


@app.get("/api/v1/pipeline/stages", response_model=list[StageNodeSchema], tags=["pipeline"])
async def list_pipeline_stages() -> list[StageNodeSchema]:
    """Return the 6 Phase 1 (Before Parliament) UK Government Bill stages and stop risks."""
    return [
        StageNodeSchema(
            stage=node.stage,
            phase=node.phase,
            order=node.order,
            label=node.label,
            description=node.description,
            stop_risks=list(node.stop_risks),
        )
        for node in GOVERNMENT_BILL_PIPELINE
    ]


@app.get("/api/v1/bills", response_model=list[BillSummarySchema], tags=["bills"])
async def list_bills(
    search: Annotated[
        str | None, Query(description="Filter by title, department, or bill_id")
    ] = None,
    outcome: Annotated[str | None, Query(description="Filter by outcome")] = None,
    house: Annotated[str | None, Query(description="Filter by originating house")] = None,
) -> list[BillSummarySchema]:
    """List all UK Government Bills stored in Neon Lakebase Postgres."""
    rows = await list_all_bills_from_db(search=search, outcome=outcome, house=house)
    return [BillSummarySchema(**r) for r in rows]


@app.get(
    "/api/v1/bills/{bill_id}/forecast",
    response_model=BillForecastResponse,
    tags=["forecast"],
)
def get_bill_forecast(
    bill_id: int,
    as_of: Annotated[
        date | None,
        Query(description="Forecast date (YYYY-MM-DD); defaults to today"),
    ] = None,
) -> BillForecastResponse:
    """Passage forecast for a UK Government Bill: probability of Royal Assent, its timing,
    and the risk of failing at each remaining stage (10,000 simulated paths)."""
    from westminster_weather.model.forecast import ForecastUnavailable, forecast_bill

    as_of = as_of or date.today()
    try:
        response = forecast_bill(bill_id, as_of)
    except ForecastUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _with_stage3(response, bill_id, as_of)


def _with_stage3(response: BillForecastResponse, bill_id: int, as_of: date) -> BillForecastResponse:
    """Fill the stage 3 fields (Royal Assent -> law applies) from the stage 3 XGBoost."""
    from datetime import timedelta

    from westminster_weather.journey import stage3_xgb
    from westminster_weather.schemas.bills import TimelineQuantiles

    s3 = stage3_xgb.forecast(bill_id, as_of)
    if s3 is None:
        return response
    d = s3["days"]
    return response.model_copy(
        update={
            "commencement_probability": round(s3["p_applies_on_royal_assent"], 4),
            "time_to_commencement": TimelineQuantiles(
                p10_days=d[0.1],
                p50_days=d[0.5],
                p90_days=d[0.9],
                estimated_median_date=as_of + timedelta(days=d[0.5]),
                p10_date=as_of + timedelta(days=d[0.1]),
                p90_date=as_of + timedelta(days=d[0.9]),
            ),
        }
    )


@app.get(
    "/api/v1/bills/{bill_id}/outlook",
    response_model=BillOutlook,
    tags=["forecast"],
)
def get_bill_outlook(
    bill_id: int,
    target_date: Annotated[
        date | None,
        Query(
            description="Weather-status target date (YYYY-MM-DD); default: next recess "
            "at least 30 days away"
        ),
    ] = None,
    as_of: Annotated[
        date | None,
        Query(description="Forecast date (YYYY-MM-DD); defaults to today"),
    ] = None,
) -> BillOutlook:
    """Presentation-ready outlook for a bill in Parliament: chance of law by deadlines,
    weather status, weekly curve, pace and next milestone, with sentences ready to show."""
    from westminster_weather.model.forecast import ForecastUnavailable
    from westminster_weather.model.outlook import NotInParliament, build_outlook

    try:
        return build_outlook(bill_id, as_of or date.today(), target_date)
    except (ForecastUnavailable, NotInParliament) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/v1/bills/{bill_id}",
    response_model=GovernmentBillTrackingResponse,
    tags=["bills"],
)
async def get_tracked_bill(
    bill_id: str,
    as_of_date: Annotated[
        date | None,
        Query(description=("Historical date (YYYY-MM-DD) at which to evaluate the bill's state.")),
    ] = None,
) -> GovernmentBillTrackingResponse:
    """Return any UK Government Bill's stage tracking state at `as_of_date` from Neon Postgres."""
    if bill_id.isdigit():
        evaluated = await evaluate_parliament_bill_from_db(int(bill_id), as_of_date=as_of_date)
        if evaluated is None:
            raise HTTPException(status_code=404, detail=f"Bill '{bill_id}' not found")
        return GovernmentBillTrackingResponse(
            id=evaluated["id"],
            parliament_bill_id=evaluated["parliament_bill_id"],
            short_title=evaluated["short_title"],
            long_title=evaluated["long_title"],
            sponsoring_department=evaluated["sponsoring_department"],
            originating_house=evaluated["originating_house"],
            introduced_session_id=evaluated["introduced_session_id"],
            outcome=evaluated["outcome"],
            outcome_date=evaluated["outcome_date"],
            stop_reason=evaluated["stop_reason"],
            as_of_date=evaluated["as_of_date"],
            earliest_signal_date=evaluated["earliest_signal_date"],
            first_reading_date=evaluated["first_reading_date"],
            current_phase=evaluated["current_phase"],
            current_stage=evaluated["current_stage"],
            current_stage_order=evaluated["current_stage_order"],
            total_stages=evaluated["total_stages"],
            stages=_serialize_tracked_nodes(evaluated["tracked_nodes"]),
            milestone_dates=[MilestoneDateSchema(**m) for m in evaluated["milestone_dates"]],
            full_text_url=evaluated["full_text_url"],
            full_text_label=evaluated["full_text_label"],
        )

    raise HTTPException(status_code=404, detail=f"Bill '{bill_id}' not found")
