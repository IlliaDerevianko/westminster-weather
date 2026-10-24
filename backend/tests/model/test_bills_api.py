"""Tests for flattening UK Parliament Bills API payloads and Neon Postgres dataset counts."""


from datetime import date

import pandas as pd
import pytest

from westminster_weather.data import bills_api, calendar
from westminster_weather.db import fetch_bills_df, fetch_sessions_df, fetch_stage_sittings_df


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
                "house": house,
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


def _label(payloads):
    sittings = bills_api.flatten_stage_sittings(payloads)
    sessions = calendar.sessions(sittings, overrides=pd.DataFrame())
    return bills_api.flatten_bills(payloads, sittings, sessions).set_index("bill_id"), sittings


@pytest.fixture
def synthetic():
    payloads = [
        _payload(
            1, [(6, ["2016-06-01"]), (49, ["2016-07-01"]), (11, ["2017-01-10"])], is_act=True
        ),
        _payload(2, [(6, ["2016-06-02"]), (8, ["2016-07-02", "2016-07-05"])]),
        _payload(3, [(6, ["2016-06-03"])], withdrawn="2016-09-01T00:00:00"),
        _payload(4, [(6, ["2017-04-01"])], session=31),
        _payload(5, [(6, ["2017-04-27"]), (18, ["2017-04-20"])], session=31),
        _payload(6, [(6, ["2018-01-01"])], session=32),
    ]
    return _label(payloads)


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
    assert bills.loc[4, "stop_reason"] == "election_called"
    assert bills.loc[5, "outcome"] == "fell_dissolution"
    assert bills.loc[6, "outcome"] == "in_progress"
    assert pd.isna(bills.loc[6, "outcome_date"])
    assert bills.loc[2, "stop_reason"] == "session_runs_out"


def test_carry_over_survives_prorogation_only():
    bills, _ = _label(
        [
            _payload(8, [(6, ["2016-06-01"]), (18, ["2016-06-10"])], session=30),
            _payload(10, [(6, ["2016-06-02"])], session=30),
            _payload(9, [(6, ["2016-07-01"])], session=31),
        ]
    )
    assert bills.loc[8, "outcome"] == "in_progress"
    assert bills.loc[10, "outcome"] == "fell_session_end"


def test_scheduled_sittings_are_split_off():
    sittings = bills_api.flatten_stage_sittings(
        [_payload(1, [(6, ["2026-10-20"]), (7, ["2026-10-30"])], session=40)]
    )
    past, scheduled = bills_api.split_scheduled(sittings)
    assert list(past["event_date"]) == [date(2026, 10, 20)]
    assert list(scheduled["event_date"]) == [date(2026, 10, 30)]


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


# --- checks against the real downloaded data in Neon Lakebase Postgres -----------------------


def test_real_neon_db_data_counts():
    bills = fetch_bills_df()
    sittings = fetch_stage_sittings_df()
    assert len(bills) >= 580
    assert bills["bill_id"].is_unique
    assert 510 <= bills["is_act"].sum() <= 540
    assert len(sittings) >= 4000


def test_every_act_has_royal_assent_sitting_in_neon_db():
    bills = fetch_bills_df()
    sittings = fetch_stage_sittings_df()
    with_ra = set(sittings.loc[sittings["api_stage_id"] == 11, "bill_id"])
    acts = set(bills.loc[bills["is_act"], "bill_id"])
    assert acts - with_ra == set()
    assert bills.loc[bills["is_act"], "outcome_date"].notna().all()


def test_in_progress_bills_are_current_or_carried_over_in_neon_db():
    """DATA_CONTRACT §8: every in-progress bill is in the current session or carried over."""
    bills = fetch_bills_df()
    current = fetch_sessions_df()["session_id"].max()
    in_progress = bills[bills["outcome"] == "in_progress"]
    ok = (
        in_progress["included_session_ids"].map(lambda s: current in s)
        | in_progress["carry_over_date"].notna()
    )
    assert ok.all()
