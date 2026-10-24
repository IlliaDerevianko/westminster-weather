from datetime import date

import pandas as pd
import pytest

from westminster_weather.data import bills_api
from westminster_weather.data.paths import PROCESSED_DIR


def _payload(bill_id, stages, *, house="Commons", session=30, is_act=False, withdrawn=None):
    return {
        "fetched_at": "2026-10-24T12:00:00+00:00",
        "list_item": {
            "billId": bill_id,
            "billTypeId": 1,
            "shortTitle": f"Bill {bill_id}",
            "originatingHouse": house,
            "introducedSessionId": session,
            "includedSessionIds": [session],
            "isAct": is_act,
            "isDefeated": False,
            "billWithdrawn": withdrawn,
            "lastUpdate": "2026-01-01T00:00:00",
        },
        "detail": {"sponsors": [{"sortOrder": 1, "organisation": {"name": "Home Office"}}]},
        "stages": [
            {
                "id": bill_id * 100 + i,
                "stageId": stage_id,
                "house": "Commons",
                "sessionId": session,
                "sortOrder": i,
                "stageSittings": [
                    {"id": bill_id * 1000 + i * 10 + j, "date": f"{d}T00:00:00"}
                    for j, d in enumerate(dates)
                ],
            }
            for i, (stage_id, dates) in enumerate(stages)
        ],
    }


@pytest.fixture
def synthetic():
    payloads = [
        _payload(1, [(6, ["2016-06-01"]), (49, ["2016-07-01"]), (11, ["2017-01-10"])], is_act=True),
        _payload(2, [(6, ["2016-06-02"]), (8, ["2016-07-02", "2016-07-05"])]),
        _payload(3, [(6, ["2016-06-03"])], withdrawn="2016-09-01T00:00:00"),
        _payload(4, [(6, ["2017-04-01"])], session=31),  # ends 2017-04-27, election 2017-06-08
        _payload(5, [(6, ["2017-04-27"]), (18, ["2017-04-20"])], session=31),
        _payload(6, [(6, ["2018-01-01"])], session=32),  # latest session = current
    ]
    sittings = bills_api.flatten_stage_sittings(payloads)
    sessions = bills_api.session_bounds(sittings)
    return bills_api.flatten_bills(payloads, sittings, sessions).set_index("bill_id"), sittings


def test_stage_sittings_one_row_per_sitting(synthetic):
    _, sittings = synthetic
    assert len(sittings) == 11
    assert sittings["sitting_id"].is_unique
    assert (sittings.query("bill_id == 2")["api_stage_id"] == 8).sum() == 2


def test_outcomes_are_labelled(synthetic):
    bills, _ = synthetic
    assert bills.loc[1, "outcome"] == "royal_assent"
    assert bills.loc[1, "outcome_date"] == date(2017, 1, 10)
    assert bills.loc[2, "outcome"] == "fell_session_end"
    assert bills.loc[3, "outcome"] == "withdrawn"
    assert bills.loc[4, "outcome"] == "fell_dissolution"
    assert bills.loc[6, "outcome"] == "in_progress"
    assert pd.isna(bills.loc[6, "outcome_date"])


def test_bill_columns(synthetic):
    bills, _ = synthetic
    assert bills.loc[1, "introduced_date"] == date(2016, 6, 1)
    assert bills.loc[1, "commons_committee_type"] == "whole_house"
    assert bills.loc[2, "commons_committee_type"] == "public_bill_committee"
    assert bills.loc[5, "carry_over_date"] == date(2017, 4, 20)
    assert bills.loc[1, "department"] == "Home Office"


def test_cache_skips_unchanged_bills():
    item = {"billId": 1, "lastUpdate": "2026-01-01T00:00:00"}
    cached = {"list_item": dict(item), "detail": {"x": 1}}
    assert not bills_api._needs_download(cached, item, detail=True)
    assert bills_api._needs_download(cached, {**item, "lastUpdate": "2026-02-01"}, detail=True)
    assert bills_api._needs_download({**cached, "detail": None}, item, detail=True)
    assert bills_api._needs_download(None, item, detail=False)


# --- checks against the real downloaded data (run `cli fetch` first) -----------------------

needs_data = pytest.mark.skipif(
    not (PROCESSED_DIR / "bills.parquet").exists(), reason="run `cli fetch` first"
)


@needs_data
def test_real_data_counts():
    bills = pd.read_parquet(PROCESSED_DIR / "bills.parquet")
    sittings = pd.read_parquet(PROCESSED_DIR / "stage_sittings.parquet")
    assert len(bills) == 588
    assert bills["bill_id"].is_unique
    assert set(bills["bill_id"]) <= set(sittings["bill_id"])  # every bill has a sitting
    assert 510 <= bills["is_act"].sum() <= 535  # ≈ 521 Acts


@needs_data
def test_every_act_has_royal_assent_sitting():
    bills = pd.read_parquet(PROCESSED_DIR / "bills.parquet")
    sittings = pd.read_parquet(PROCESSED_DIR / "stage_sittings.parquet")
    with_ra = set(sittings.loc[sittings["api_stage_id"] == 11, "bill_id"])
    acts = set(bills.loc[bills["is_act"], "bill_id"])
    assert acts - with_ra == set()
    assert bills.loc[bills["is_act"], "outcome_date"].notna().all()


def test_session_bounds_ignore_stray_dates():
    def rows(session, dates):
        return [{"session_id": session, "event_date": date.fromisoformat(d)} for d in dates]

    sittings = pd.DataFrame(
        rows(30, ["2017-06-22", "2019-10-08", "2019-10-16"])  # 2019-10-16 is a stray date
        + rows(34, ["2019-10-15", "2019-11-05"])
        + rows(35, ["2019-12-18"])
    )
    bounds = bills_api.session_bounds(sittings).set_index("session_id")
    assert bounds.loc[30, "end_date"] == date(2019, 10, 8)
    assert bounds.loc[30, "end_type"] == "prorogation"  # next session began before the election
    assert bounds.loc[34, "end_type"] == "dissolution"  # 2019-12-12 election in the gap
    assert bounds.loc[35, "end_type"] is None  # current session
