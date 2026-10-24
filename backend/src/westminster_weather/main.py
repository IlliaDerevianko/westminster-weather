"""FastAPI application entrypoint for Westminster Weather."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from westminster_weather import __version__
from westminster_weather.config import settings
from westminster_weather.domain.bills import BILLS_STORE, GovernmentBill
from westminster_weather.domain.stages import GOVERNMENT_BILL_PIPELINE
from westminster_weather.schemas.bills import (
    GovernmentBillTrackingResponse,
    StageNodeSchema,
    TrackedStageNodeSchema,
    UpdateBillStageRequest,
)

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=(
        "Predicts whether UK Government Bills will pass and estimates the timeline "
        "until commencement to support business decision-making."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serialize_bill(bill: GovernmentBill) -> GovernmentBillTrackingResponse:
    tracked_nodes = bill.build_tracked_pipeline()
    return GovernmentBillTrackingResponse(
        id=bill.id,
        short_title=bill.short_title,
        long_title=bill.long_title,
        sponsoring_department=bill.sponsoring_department,
        simulated_as_of_date=bill.simulated_as_of_date,
        current_phase=bill.current_phase,
        current_stage=bill.current_stage,
        current_stage_order=bill.current_node.order,
        total_stages=len(GOVERNMENT_BILL_PIPELINE),
        stopped_reason=bill.stopped_reason,
        passage_probability=bill.passage_probability,
        estimated_months_to_commencement=bill.estimated_months_to_commencement,
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
                active_stop_reason=t.active_stop_reason,
            )
            for t in tracked_nodes
        ],
    )


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "version": __version__}


@app.get("/api/v1/pipeline/stages", response_model=list[StageNodeSchema], tags=["pipeline"])
async def list_pipeline_stages() -> list[StageNodeSchema]:
    """Return the 3-phase UK Government Bill flowchart stages and their stop risks."""
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


@app.get("/api/v1/bills", response_model=list[GovernmentBillTrackingResponse], tags=["bills"])
async def list_tracked_bills() -> list[GovernmentBillTrackingResponse]:
    """Return all tracked Government Bills with their dynamic 24-stage pipeline status."""
    return [_serialize_bill(bill) for bill in BILLS_STORE.values()]


@app.get(
    "/api/v1/bills/{bill_id}",
    response_model=GovernmentBillTrackingResponse,
    tags=["bills"],
)
async def get_tracked_bill(bill_id: str) -> GovernmentBillTrackingResponse:
    """Return a specific tracked Government Bill and its progression across all 24 stages."""
    bill = BILLS_STORE.get(bill_id)
    if not bill:
        raise HTTPException(status_code=404, detail=f"Bill '{bill_id}' not found")
    return _serialize_bill(bill)


@app.patch(
    "/api/v1/bills/{bill_id}/stage",
    response_model=GovernmentBillTrackingResponse,
    tags=["bills"],
)
async def update_bill_stage(
    bill_id: str, payload: UpdateBillStageRequest
) -> GovernmentBillTrackingResponse:
    """Update a bill's current stage (or stop reason) to test dynamic stage tracking."""
    bill = BILLS_STORE.get(bill_id)
    if not bill:
        raise HTTPException(status_code=404, detail=f"Bill '{bill_id}' not found")
    bill.current_stage = payload.current_stage
    bill.stopped_reason = payload.stopped_reason
    return _serialize_bill(bill)
