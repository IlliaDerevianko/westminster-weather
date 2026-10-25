"""Tests for Phase 1 (Before Parliament) pipeline stages and Neon Lakebase Postgres persistence."""

import pytest
from httpx import ASGITransport, AsyncClient

from westminster_weather.config import settings
from westminster_weather.domain.bills import StageStatus
from westminster_weather.domain.stages import (
    GOVERNMENT_BILL_PIPELINE,
    BillStage,
    LifecyclePhase,
    StopReason,
)
from westminster_weather.main import app


async def test_health_endpoint_verifies_neon_postgres() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == settings.app_name
    assert data["database"] == "connected"
    assert data["neon_branch"] == "production"
    assert "PostgreSQL" in data["postgres_version"]


async def test_phase_one_pipeline_stages_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/pipeline/stages")
    assert response.status_code == 200
    stages = response.json()
    assert len(stages) == len(GOVERNMENT_BILL_PIPELINE) == 6
    assert all(s["phase"] == LifecyclePhase.BEFORE_PARLIAMENT for s in stages)
    assert stages[0]["stage"] == BillStage.SIGNAL_APPEARS
    assert stages[-1]["stage"] == BillStage.READY_FOR_FIRST_READING


def test_phase_one_stop_risks() -> None:
    drafting_node = next(
        n for n in GOVERNMENT_BILL_PIPELINE if n.stage == BillStage.DRAFTING
    )
    assert StopReason.STUCK_IN_DRAFTING in drafting_node.stop_risks


async def test_sync_bill_to_neon_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/bills/online-safety-bill/sync")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "synced"
    assert data["bill_id"] == "online-safety-bill"
    assert data["neon_branch"] == "production"
    assert data["stages_saved"] == 6
    assert data["events_saved"] >= 10


@pytest.mark.parametrize(
    ("as_of_date", "expected_stage", "expected_order"),
    [
        ("2017-11-15", BillStage.SIGNAL_APPEARS, 1),
        ("2018-07-01", BillStage.GOVERNMENT_COMMITS, 2),
        ("2019-06-01", BillStage.CONSULTATION, 3),
        ("2021-01-15", BillStage.DRAFTING, 4),
        ("2021-06-01", BillStage.SLOT_IN_PROGRAMME, 5),
        ("2022-03-17", BillStage.READY_FOR_FIRST_READING, 6),
        ("2026-10-24", BillStage.READY_FOR_FIRST_READING, 6),
    ],
)
async def test_online_safety_bill_historical_dates_from_neon_db(
    as_of_date: str, expected_stage: BillStage, expected_order: int
) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/bills/online-safety-bill", params={"as_of_date": as_of_date}
        )
    assert response.status_code == 200
    bill = response.json()

    assert bill["id"] == "online-safety-bill"
    assert bill["parliament_bill_id"] == 3137
    assert bill["as_of_date"] == as_of_date
    assert bill["current_stage"] == expected_stage
    assert bill["current_stage_order"] == expected_order
    assert len(bill["stages"]) == 6

    for stage_item in bill["stages"]:
        if stage_item["order"] < expected_order:
            assert stage_item["status"] == StageStatus.COMPLETED
            assert stage_item["entered_date"] is not None
        elif stage_item["order"] == expected_order:
            expected_status = (
                StageStatus.COMPLETED
                if expected_stage == BillStage.READY_FOR_FIRST_READING
                else StageStatus.CURRENT
            )
            assert stage_item["status"] == expected_status
            assert stage_item["entered_date"] <= as_of_date
            assert len(stage_item["visible_events"]) >= 1
            for ev in stage_item["visible_events"]:
                assert ev["event_date"] <= as_of_date
        else:
            assert stage_item["status"] == StageStatus.UPCOMING
            assert stage_item["entered_date"] is None
            assert stage_item["visible_events"] == []


async def test_list_all_bills_from_neon_db_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/bills")
    assert response.status_code == 200
    bills = response.json()
    assert len(bills) >= 580
    assert any(b["bill_id"] == 3137 for b in bills)


async def test_get_any_parliament_bill_by_numeric_id_from_neon_db() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/bills/3137")
    assert response.status_code == 200
    data = response.json()
    assert data["parliament_bill_id"] == 3137
    assert "Online Safety" in data["short_title"]
    assert len(data["stages"]) == 22
    phases = {s["phase"] for s in data["stages"]}
    assert phases == {"before_parliament", "inside_parliament", "after_royal_assent"}
    assert len(data["milestone_dates"]) >= 8
    ping_pong = next(s for s in data["stages"] if s["stage"] == "ping_pong")
    assert ping_pong["status"] == StageStatus.COMPLETED
    assert ping_pong["sitting_count"] == 2
    assert "2 ping-pong rounds" in ping_pong["summary"]


async def test_ping_pong_marked_skipped_with_zero_rounds_when_unneeded() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/bills/4090")
    assert response.status_code == 200
    data = response.json()
    ping_pong = next(s for s in data["stages"] if s["stage"] == "ping_pong")
    assert ping_pong["status"] == StageStatus.SKIPPED
    assert ping_pong["sitting_count"] == 0
    assert "0 ping-pong rounds" in ping_pong["summary"]
    comm_orders = next(s for s in data["stages"] if s["stage"] == "commencement_orders")
    assert comm_orders["status"] == StageStatus.SKIPPED
    assert comm_orders["sitting_count"] == 0
    in_force = next(s for s in data["stages"] if s["stage"] == "law_fully_in_force")
    assert in_force["status"] == StageStatus.COMPLETED


async def test_post_royal_assent_stages_and_model_interface_from_neon_db() -> None:
    from datetime import date

    from westminster_weather.data.interface import (
        get_act_commencements,
        get_post_royal_assent_events,
    )

    comm_df = get_act_commencements(date(2026, 10, 24))
    events_df = get_post_royal_assent_events(date(2026, 10, 24))
    assert len(comm_df) >= 520
    assert len(events_df) >= 4500

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/bills/3137")
    assert response.status_code == 200
    data = response.json()
    by_stage = {s["stage"]: s for s in data["stages"]}
    assert by_stage["commencement_orders"]["status"] == StageStatus.COMPLETED
    assert by_stage["commencement_orders"]["sitting_count"] >= 7
    assert by_stage["regulations_drafted"]["status"] == StageStatus.COMPLETED
    assert by_stage["regulations_drafted"]["sitting_count"] >= 4
    assert by_stage["regulations_laid"]["status"] == StageStatus.COMPLETED
    assert by_stage["regulations_laid"]["sitting_count"] >= 7
    # Online Safety Act: some provisions are still prospective on legislation.gov.uk
    # (`act_in_force_status`), so the law is not fully in force yet
    assert by_stage["law_fully_in_force"]["status"] == StageStatus.CURRENT
    assert "prospective" in by_stage["law_fully_in_force"]["summary"]


async def test_zero_regulations_upcoming_with_none_yet_summary_for_bill_3751() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/bills/3751")
    assert response.status_code == 200
    data = response.json()
    by_stage = {s["stage"]: s for s in data["stages"]}
    # Water (Special Measures) Act 2025: no regulations under it, but some provisions are
    # still prospective, so the regulation stages stay upcoming with a "none yet" summary
    assert by_stage["law_fully_in_force"]["status"] == StageStatus.CURRENT
    for st, phrase in (("regulations_drafted", "None drafted yet"),
                       ("regulations_laid", "None laid yet")):
        assert by_stage[st]["status"] == StageStatus.UPCOMING
        assert by_stage[st]["sitting_count"] == 0
        assert by_stage[st]["summary"].startswith(phrase)
        assert by_stage[st]["web_url"] == "https://www.legislation.gov.uk/ukpga/2025/5"


async def test_fast_tracked_bypassed_stages_marked_for_bill_4296() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/bills/4296")
    assert response.status_code == 200
    data = response.json()
    by_stage = {s["stage"]: s for s in data["stages"]}
    # Stages 1..5 in Before Parliament have no separate formal records prior to 1st Reading
    for st in (
        "signal_appears",
        "government_commits",
        "consultation",
        "drafting",
        "slot_in_programme",
    ):
        assert by_stage[st]["status"] == StageStatus.FAST_TRACKED
        assert by_stage[st]["sitting_count"] == 0
        assert "Fast-tracked / bypassed" in by_stage[st]["summary"]
    # Stage 10 (commons_report_stage) had 0 sittings as the bill went straight to 3rd Reading
    assert by_stage["commons_report_stage"]["status"] == StageStatus.FAST_TRACKED
    assert by_stage["commons_report_stage"]["sitting_count"] == 0
    assert "0 sittings" in by_stage["commons_report_stage"]["summary"]





