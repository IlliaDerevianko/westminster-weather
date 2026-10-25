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


@pytest.mark.parametrize(
    ("as_of_date", "expected_order"),
    [
        ("2017-11-15", 1),  # signal appears
        ("2018-07-01", 2),  # government commits
        ("2019-06-01", 3),  # consultation
        ("2021-01-15", 4),  # drafting
        ("2021-06-01", 5),  # slot in the programme
        ("2022-03-17", 7),  # all 6 pre-Parliament stages done; Commons 1st reading
        ("2026-10-24", 7),
    ],
)
async def test_pre_parliament_history_from_tracked_bills_by_bill_id(
    as_of_date: str, expected_order: int
) -> None:
    # Bill 3137 (Online Safety Bill) has stored pre-Parliament records in `tracked_bills`;
    # the generic bill endpoint replays them like any other bill's
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/bills/3137", params={"as_of_date": as_of_date})
    assert response.status_code == 200
    bill = response.json()
    assert bill["parliament_bill_id"] == 3137
    assert bill["as_of_date"] == as_of_date
    pre = [s for s in bill["stages"] if s["phase"] == LifecyclePhase.BEFORE_PARLIAMENT]
    assert [s["order"] for s in pre] == [1, 2, 3, 4, 5, 6]
    if expected_order <= 6:
        assert bill["current_stage_order"] == expected_order

    for stage_item in pre:
        if stage_item["order"] < expected_order:
            assert stage_item["status"] == StageStatus.COMPLETED
            assert stage_item["entered_date"] is not None
        elif stage_item["order"] == expected_order:
            assert stage_item["status"] == StageStatus.CURRENT
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
    assert len(data["stages"]) == 21
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


@pytest.mark.parametrize("bill_id", [33, 40, 45, 57, 3751])
async def test_zero_si_post_royal_assent_stages_are_skipped_or_none_yet(bill_id: int) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/v1/bills/{bill_id}")
    assert response.status_code == 200
    by_stage = {s["stage"]: s for s in response.json()["stages"]}
    fully_in_force = by_stage["law_fully_in_force"]["status"] == StageStatus.COMPLETED
    # A stage with no SIs after Royal Assent is skipped once the Act is fully in force;
    # otherwise it stays upcoming but says none exist yet, with a link to the Act
    for st in ("commencement_orders", "regulations_drafted", "regulations_laid"):
        stage = by_stage[st]
        if stage["sitting_count"] > 0:
            continue
        if fully_in_force:
            assert stage["status"] == StageStatus.SKIPPED
        else:
            assert stage["status"] == StageStatus.UPCOMING
            assert stage["summary"].startswith("None ")
            assert " yet — " in stage["summary"]
            assert stage["web_url"].startswith("https://www.legislation.gov.uk/ukpga/")


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







@pytest.mark.parametrize(
    ("bill_id", "as_of", "expected_url"),
    [
        # An Act: the full Act on legislation.gov.uk
        (3751, None, "https://www.legislation.gov.uk/ukpga/2025/5"),
        # The same Act viewed before its Royal Assent: the bill texts
        (3751, "2024-10-01", "https://bills.parliament.uk/bills/3751/publications"),
        # A bill still in Parliament: the bill texts
        (4305, None, "https://bills.parliament.uk/bills/4305/publications"),
    ],
)
async def test_full_text_url_for_acts_and_bills(
    bill_id: int, as_of: str | None, expected_url: str
) -> None:
    transport = ASGITransport(app=app)
    params = {"as_of_date": as_of} if as_of else {}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/v1/bills/{bill_id}", params=params)
    assert response.status_code == 200
    data = response.json()
    assert data["full_text_url"] == expected_url
    assert data["full_text_label"].startswith("Read the full ")


def test_latest_confirmed_post_ra_stage_rules() -> None:
    from datetime import date

    from westminster_weather.domain.bills import latest_confirmed_post_ra_stage

    today = date(2026, 10, 25)
    # Fully in force wins over everything else
    assert latest_confirmed_post_ra_stage(date(2025, 1, 1), {"regulations_laid"}, today) == (
        BillStage.LAW_FULLY_IN_FORCE
    )
    # In force only in the future: not yet
    assert latest_confirmed_post_ra_stage(date(2027, 1, 1), set(), today) == (
        BillStage.ROYAL_ASSENT
    )
    # Not fully in force: the latest SI stage with an instrument
    assert latest_confirmed_post_ra_stage(
        None, {"commencement_orders", "regulations_drafted"}, today
    ) == BillStage.REGULATIONS_DRAFTED
    assert latest_confirmed_post_ra_stage(None, {"commencement_orders"}, today) == (
        BillStage.COMMENCEMENT_ORDERS
    )
    # Only a later in-force event: "some sections start" is complete
    assert latest_confirmed_post_ra_stage(None, {"law_fully_in_force"}, today) == (
        BillStage.SOME_SECTIONS_START
    )
    assert latest_confirmed_post_ra_stage(None, {"some_sections_start"}, today) == (
        BillStage.ROYAL_ASSENT
    )


async def test_bill_list_latest_stage_matches_bill_page() -> None:
    from datetime import date

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        listed = (await client.get("/api/v1/bills", params={"outcome": "royal_assent"})).json()
        assert len(listed) >= 500
        assert all(b["latest_stage"] for b in listed)
        # Acts at different stages: fully in force, commencement orders, regulations laid
        by_id = {b["bill_id"]: b for b in listed}
        for bill_id in (4207, 3751, 45, 33, 40, 57):
            page = (
                await client.get(
                    f"/api/v1/bills/{bill_id}", params={"as_of_date": date.today().isoformat()}
                )
            ).json()
            done = [s["stage"] for s in page["stages"] if s["status"] == StageStatus.COMPLETED]
            assert by_id[bill_id]["latest_stage"] == done[-1], bill_id
        not_acts = (await client.get("/api/v1/bills", params={"outcome": "in_progress"})).json()
        assert all(b["latest_stage"] is None for b in not_acts)
