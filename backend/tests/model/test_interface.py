"""DATA_CONTRACT §5 interface: correctness on synthetic tables and Neon Postgres."""


from datetime import date

import pandas as pd
import pytest

from westminster_weather.data import bills_api
from westminster_weather.data.interface import DatabaseInterface


def _payload(bill_id, stages, session=30, is_act=False):
    return {
        "fetched_at": "2026-10-24T12:00:00+00:00",
        "list_item": {
            "billId": bill_id,
            "billTypeId": 1,
            "shortTitle": f"Bill {bill_id}",
            "originatingHouse": "Commons",
            "introducedSessionId": session,
            "includedSessionIds": [session],
            "isAct": is_act,
            "isDefeated": False,
            "billWithdrawn": None,
            "lastUpdate": "2026-01-01T00:00:00",
        },
        "detail": {"sponsors": []},
        "stages": [
            {
                "id": bill_id * 100 + i,
                "stageId": stage_id,
                "house": house,
                "sessionId": session,
                "sortOrder": i,
                "stageSittings": [{"id": bill_id * 1000 + i, "date": f"{d}T00:00:00"}],
            }
            for i, (stage_id, house, d) in enumerate(stages)
        ],
    }


@pytest.fixture
def synthetic():
    payloads = [
        _payload(
            1,
            [
                (6, "Commons", "2016-06-01"),
                (7, "Commons", "2016-06-08"),
                (49, "Commons", "2016-06-15"),
                (10, "Commons", "2016-06-15"),
                (1, "Lords", "2016-06-16"),
                (2, "Lords", "2016-06-20"),
                (3, "Lords", "2016-06-27"),
                (4, "Lords", "2016-07-04"),
                (5, "Lords", "2016-07-05"),
                (11, "Unassigned", "2016-07-10"),
            ],
            is_act=True,
        ),
        _payload(2, [(6, "Commons", "2016-06-02"), (7, "Commons", "2016-06-09")]),
        _payload(3, [(6, "Commons", "2016-11-01")], session=31),
    ]
    tables = bills_api.build_tables_from_payloads(payloads, persist_to_db=False)
    return DatabaseInterface(tables=tables)


def _rows(transitions, bill_id):
    return transitions[transitions["bill_id"] == bill_id].set_index("model_state")


def test_skipped_state_has_zero_duration(synthetic):
    t = _rows(synthetic.get_transitions(date(2016, 12, 1)), 1)
    report = t.loc["commons_report_stage"]
    assert report["entered_date"] == report["exited_date"] == date(2016, 6, 15)
    assert report["sitting_days_in_state"] == 0
    assert "ping_pong" not in t.index
    assert t.loc["royal_assent", "entered_date"] == date(2016, 7, 10)
    assert pd.isna(t.loc["royal_assent", "exit_type"])
    assert (t.drop(index="royal_assent")["exit_type"] == "advance").all()


def test_state_in_progress_is_censored(synthetic):
    t = _rows(synthetic.get_transitions(date(2016, 6, 21)), 1)
    assert list(t.index)[-1] == "lords_1st_and_2nd_reading"
    assert pd.isna(t.iloc[-1]["exit_type"]) and pd.isna(t.iloc[-1]["exited_date"])


def test_failure_appears_only_after_it_happens(synthetic):
    end = synthetic.sessions.set_index("session_id").loc[30, "end_date"]
    before = _rows(synthetic.get_transitions(date(2016, 6, 30)), 2)
    assert pd.isna(before.iloc[-1]["exit_type"])
    after = _rows(synthetic.get_transitions(date(2016, 12, 1)), 2)
    assert after.iloc[-1]["exit_type"] == "fail"
    assert after.iloc[-1]["fail_cause"] == "session_end"
    assert after.iloc[-1]["exited_date"] == end


def test_bills_and_covariates_as_of(synthetic):
    bills = synthetic.get_bills(date(2016, 6, 1)).set_index("bill_id")
    assert list(bills.index) == [1]
    assert bills.loc[1, "outcome"] == "in_progress"
    cov = synthetic.get_covariates(date(2016, 6, 20))
    cwh = cov[(cov["bill_id"] == 1) & (cov["feature"] == "committee_whole_house")]
    assert list(cwh["value"]) == [0.0, 1.0]
    same_day = cov[(cov["bill_id"] == 1) & (cov["feature"] == "same_day_2r_3r")]
    assert list(same_day["value"]) == [0.0]


# --- leakage checks on the real data in Neon Lakebase Postgres -----------------------------

AS_OF_DATES = [date(2008, 3, 1), date(2015, 3, 20), date(2019, 10, 1), date(2024, 1, 15)]


@pytest.fixture(scope="module")
def real():
    return DatabaseInterface()


@pytest.mark.parametrize("as_of", AS_OF_DATES)
def test_transitions_never_after_as_of(real, as_of):
    t = real.get_transitions(as_of)
    assert len(t) > 0
    assert (t["entered_date"] <= as_of).all()
    assert (t["exited_date"].dropna() <= as_of).all()
    assert (t["sitting_days_in_state"] >= 0).all()


@pytest.mark.parametrize("as_of", AS_OF_DATES)
def test_bills_in_progress_have_open_current_state(real, as_of):
    t = real.get_transitions(as_of)
    in_progress = real.get_bills(as_of).query("outcome == 'in_progress'")["bill_id"]
    last = t.sort_values("state_order").groupby("bill_id").tail(1).set_index("bill_id")
    current = last.loc[last.index.intersection(in_progress)]
    assert len(current) > 0
    assert current["exit_type"].isna().all()
    assert (current["model_state"] != "royal_assent").all()


@pytest.mark.parametrize("as_of", AS_OF_DATES)
def test_other_functions_never_after_as_of(real, as_of):
    bills = real.get_bills(as_of)
    assert (bills["introduced_date"].dropna() <= as_of).all()
    assert (bills["outcome_date"].dropna() <= as_of).all()
    assert (bills["carry_over_date"].dropna() <= as_of).all()
    assert (real.get_covariates(as_of)["valid_from"] <= as_of).all()
    cal = real.get_calendar(as_of)
    assert (cal["sitting_days"]["date"] <= as_of).all()
    assert (cal["sessions"]["start_date"] <= as_of).all()
    assert (cal["sessions"]["end_date"].dropna() <= as_of).all()
    assert (cal["parliaments"]["dissolution_date"].dropna() <= as_of).all()
    assert real.get_scheduled_sittings(as_of).empty
