"""Tests for FastAPI endpoints, tracked Government Bill model, and UK Parliament Bills API."""

from httpx import ASGITransport, AsyncClient

from westminster_weather.clients.parliament_bills import ParliamentBillsClient
from westminster_weather.config import settings
from westminster_weather.domain.bills import StageStatus
from westminster_weather.domain.stages import (
    GOVERNMENT_BILL_PIPELINE,
    BillStage,
    LifecyclePhase,
    StopReason,
)
from westminster_weather.main import app


async def test_health_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == settings.app_name


async def test_pipeline_stages_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/pipeline/stages")
    assert response.status_code == 200
    stages = response.json()
    assert len(stages) == len(GOVERNMENT_BILL_PIPELINE) == 24
    assert stages[0]["stage"] == BillStage.SIGNAL_APPEARS
    assert stages[0]["phase"] == LifecyclePhase.BEFORE_PARLIAMENT
    assert stages[-1]["stage"] == BillStage.AMENDED_OR_REPEALED
    assert stages[-1]["phase"] == LifecyclePhase.AFTER_ROYAL_ASSENT


async def test_tracked_pre_parliament_bill_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/bills/online-safety-bill")
    assert response.status_code == 200
    bill = response.json()
    assert bill["id"] == "online-safety-bill"
    assert bill["short_title"] == "Online Safety Bill"
    assert bill["current_phase"] == LifecyclePhase.BEFORE_PARLIAMENT
    assert bill["current_stage"] == BillStage.DRAFTING
    assert bill["current_stage_order"] == 4
    assert len(bill["stages"]) == 24

    # Orders 1-3 should be COMPLETED, Order 4 should be CURRENT, Orders 5-24 UPCOMING
    for stage_item in bill["stages"]:
        if stage_item["order"] < 4:
            assert stage_item["status"] == StageStatus.COMPLETED
        elif stage_item["order"] == 4:
            assert stage_item["status"] == StageStatus.CURRENT
            assert stage_item["stage"] == BillStage.DRAFTING
        else:
            assert stage_item["status"] == StageStatus.UPCOMING


def test_commencement_orders_stop_risk() -> None:
    commencement_node = next(
        n for n in GOVERNMENT_BILL_PIPELINE if n.stage == BillStage.COMMENCEMENT_ORDERS
    )
    assert StopReason.NEVER_COMMENCED in commencement_node.stop_risks


async def test_parliament_bills_api_returns_government_bills() -> None:
    client = ParliamentBillsClient()
    result = await client.list_government_bills(skip=0, take=5)

    assert "items" in result
    assert "totalResults" in result
    assert result["totalResults"] > 0
    assert len(result["items"]) == 5

    first_bill = result["items"][0]
    assert "billId" in first_bill
    assert "shortTitle" in first_bill
    assert first_bill["billTypeId"] == settings.government_bill_type_id
    assert "currentStage" in first_bill

    stages_result = await client.get_bill_stages(first_bill["billId"])
    assert "items" in stages_result
    assert len(stages_result["items"]) > 0
