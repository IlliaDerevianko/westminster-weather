"""FastAPI application entrypoint for Westminster Weather (backed by Neon Lakebase Postgres)."""

from datetime import date
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from westminster_weather import __version__
from westminster_weather.config import settings
from westminster_weather.db import check_db_connection
from westminster_weather.domain.bills import (
    GovernmentBill,
    get_online_safety_bill_from_live_apis,
    sync_online_safety_bill_to_db,
)
from westminster_weather.domain.stages import (
    GOVERNMENT_BILL_PIPELINE,
    LifecyclePhase,
)
from westminster_weather.schemas.bills import (
    ApiEvidenceEventSchema,
    GovernmentBillTrackingResponse,
    StageNodeSchema,
    TrackedStageNodeSchema,
)

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


def _serialize_bill_at_date(
    bill: GovernmentBill, as_of_date: date
) -> GovernmentBillTrackingResponse:
    current_stage, current_order, tracked_nodes = bill.evaluate_at_date(as_of_date)
    return GovernmentBillTrackingResponse(
        id=bill.id,
        parliament_bill_id=bill.parliament_bill_id,
        short_title=bill.short_title,
        long_title=bill.long_title,
        sponsoring_department=bill.sponsoring_department,
        as_of_date=as_of_date,
        earliest_signal_date=bill.earliest_signal_date,
        first_reading_date=bill.first_reading_date,
        current_phase=LifecyclePhase.BEFORE_PARLIAMENT,
        current_stage=current_stage,
        current_stage_order=current_order,
        total_stages=len(GOVERNMENT_BILL_PIPELINE),
        stages=[
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
            )
            for t in tracked_nodes
        ],
    )


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


@app.post("/api/v1/bills/online-safety-bill/sync", tags=["bills"])
async def sync_bill_to_neon() -> dict[str, int | str]:
    """Fetch live GOV.UK & Parliament API records and persist them into Neon Lakebase Postgres."""
    stages_saved, events_saved = await sync_online_safety_bill_to_db()
    return {
        "status": "synced",
        "bill_id": "online-safety-bill",
        "neon_branch": settings.neon_branch,
        "stages_saved": stages_saved,
        "events_saved": events_saved,
    }


@app.get(
    "/api/v1/bills/{bill_id}",
    response_model=GovernmentBillTrackingResponse,
    tags=["bills"],
)
async def get_tracked_bill(
    bill_id: str,
    as_of_date: Annotated[
        date,
        Query(
            description=(
                "Historical date (YYYY-MM-DD) at which to evaluate the bill's Phase 1 state."
            )
        ),
    ] = date(2021, 1, 15),
) -> GovernmentBillTrackingResponse:
    """Return the Online Safety Bill's Phase 1 state at `as_of_date` from Neon Lakebase Postgres."""
    if bill_id != "online-safety-bill":
        raise HTTPException(status_code=404, detail=f"Bill '{bill_id}' not found")
    bill = await get_online_safety_bill_from_live_apis()
    return _serialize_bill_at_date(bill, as_of_date)
