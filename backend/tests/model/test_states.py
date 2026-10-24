"""BUILD_PLAN task 3: model states, transitions read through the interface, and checks."""

from datetime import date

import pytest

from westminster_weather.data import bills_api
from westminster_weather.data.interface import DatabaseInterface
from westminster_weather.domain.stages import BillStage, StopReason
from westminster_weather.model import states


def _payload(bill_id, stages, session=30, is_act=False, house="Commons", withdrawn=None):
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
        "detail": {"sponsors": []},
        "stages": [
            {
                "id": bill_id * 100 + i,
                "stageId": stage_id,
                "house": h,
                "sessionId": session,
                "sortOrder": i,
                "stageSittings": [{"id": bill_id * 1000 + i, "date": f"{d}T00:00:00"}],
            }
            for i, (stage_id, h, d) in enumerate(stages)
        ],
    }


LORDS_FIRST_ACT = [
    (1, "Lords", "2016-06-01"),
    (2, "Lords", "2016-06-06"),
    (3, "Lords", "2016-06-13"),
    (4, "Lords", "2016-06-20"),
    (5, "Lords", "2016-06-22"),
    (6, "Commons", "2016-06-23"),
    (7, "Commons", "2016-07-04"),
    (8, "Commons", "2016-07-07"),
    (9, "Commons", "2016-07-12"),
    (10, "Commons", "2016-07-12"),
    (13, "Lords", "2016-07-14"),
    (11, "Unassigned", "2016-07-19"),
]


def _source(payloads):
    tables = bills_api.build_tables_from_payloads(payloads, persist_to_db=False)
    return DatabaseInterface(tables=tables)


@pytest.fixture
def synthetic():
    return _source(
        [
            _payload(1, LORDS_FIRST_ACT, is_act=True, house="Lords"),
            _payload(2, [(6, "Commons", "2016-06-02"), (7, "Commons", "2016-06-09")]),
            _payload(
                4,
                [(6, "Commons", "2016-06-03")],
                withdrawn="2016-06-20T00:00:00",
            ),
            _payload(3, [(6, "Commons", "2016-11-01")], session=31),
            _payload(5, [(6, "Commons", "2004-12-01")], session=18),  # excluded population
        ]
    )


AS_OF = date(2016, 12, 1)


def test_chain_order_depends_on_originating_house():
    lords = states.chain("Lords")
    assert lords[0] == BillStage.LORDS_FIRST_AND_SECOND_READING
    assert lords.index(BillStage.COMMONS_FIRST_READING) == 3
    assert lords[-2:] == [BillStage.PING_PONG, BillStage.ROYAL_ASSENT]
    assert states.STAGE_ID_TO_STATE[14] == BillStage.COMMONS_SECOND_READING


def test_transitions_labelled_with_enums(synthetic):
    t = states.build_transitions(AS_OF, synthetic)
    assert 5 not in set(t["bill_id"])  # session 18 is outside the population
    act = t[t["bill_id"] == 1]
    assert list(act["state"])[0] == BillStage.LORDS_FIRST_AND_SECOND_READING
    assert list(act["state"])[-1] == BillStage.ROYAL_ASSENT
    assert BillStage.PING_PONG in set(act["state"])
    last = t[t["is_last"]].set_index("bill_id")
    assert last.loc[2, "stop_reason"] == StopReason.SESSION_RUNS_OUT.value
    assert last.loc[2, "state"] == BillStage.COMMONS_SECOND_READING  # failure stage
    assert last.loc[4, "stop_reason"] == StopReason.PAUSED_OR_WITHDRAWN.value
    assert last.loc[3, "exit_type"] is None


def test_checks_pass_on_consistent_data(synthetic):
    results = states.check_transitions(AS_OF, synthetic)
    assert all(r.ok for r in results), [r.name for r in results if not r.ok]


def test_checks_catch_an_act_without_royal_assent():
    source = _source(
        [
            _payload(1, [(6, "Commons", "2016-06-01")], is_act=True),  # no stage 11 sitting
            _payload(3, [(6, "Commons", "2016-11-01")], session=31),
        ]
    )
    failed = {r.name for r in states.check_transitions(AS_OF, source) if not r.ok}
    assert failed  # an Act with no Royal Assent date is not a consistent record


def test_checks_catch_royal_assent_on_a_bill_not_labelled_an_act():
    # The Bills API marks Commercial Rent (Coronavirus) Bill (3064) isAct = False although it
    # received Royal Assent on 2022-03-24.
    stages = [(6, "Commons", "2016-06-01"), (11, "Unassigned", "2016-07-10")]
    source = _source(
        [_payload(1, stages, is_act=False), _payload(3, [(6, "Commons", "2016-11-01")], session=31)]
    )
    failed = {r.name for r in states.check_transitions(AS_OF, source) if not r.ok}
    assert "Bills that reached royal_assent but are not labelled Acts" in failed


def test_sanity_report(synthetic):
    report = states.sanity_report(AS_OF, synthetic)
    assert report["bills"] == 4
    assert report["bills_with_ping_pong"] == 1
    assert report["by_final_state"].loc["royal_assent", "royal_assent"] == 1
    assert (report["durations"]["median"].dropna() >= 0).all()
