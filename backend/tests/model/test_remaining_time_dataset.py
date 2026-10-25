import pandas as pd

from westminster_weather.analysis.remaining_time_dataset import (
    SourceData,
    build_dataset,
    feature_columns,
)


def _source() -> SourceData:
    bills = pd.DataFrame(
        [
            {"bill_id": 1, "short_title": "Example Bill", "originating_house": "Commons",
             "introduced_session_id": 30, "department": "Home Office",
             "outcome": "royal_assent", "outcome_date": pd.Timestamp("2020-02-20")},
            # Bills that did not reach Royal Assent must be dropped.
            {"bill_id": 2, "short_title": "Fallen Bill", "originating_house": "Commons",
             "introduced_session_id": 30, "department": "Home Office",
             "outcome": "fell_dissolution", "outcome_date": pd.Timestamp("2020-02-01")},
        ]
    )
    # (stage id, house, date, sort order). 2nd reading, Committee of the whole House and
    # 3rd reading all happen on Jan 10, in that order.
    rows = [
        (6, "Commons", "2020-01-01", 1), (7, "Commons", "2020-01-10", 2),
        (49, "Commons", "2020-01-10", 3), (10, "Commons", "2020-01-10", 4),
        (1, "Lords", "2020-01-12", 5), (2, "Lords", "2020-01-20", 6),
        (3, "Lords", "2020-01-30", 7), (4, "Lords", "2020-02-05", 8),
        (5, "Lords", "2020-02-10", 9), (11, "Lords", "2020-02-20", 10),
    ]
    sittings = pd.DataFrame(
        [
            {"bill_id": b, "bill_type_id": 1, "api_stage_id": sid, "house": h,
             "event_date": pd.Timestamp(d), "sort_order": o}
            for b in (1, 2)
            for sid, h, d, o in rows
        ]
    )
    sessions = pd.DataFrame(
        {"session_id": [30], "parliament_id": [5], "start_date": [pd.Timestamp("2019-12-01")]}
    )
    parliaments = pd.DataFrame(
        {"parliament_id": [5], "first_meeting_date": [pd.Timestamp("2019-12-01")]}
    )
    days = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-10", "2020-01-12", "2020-01-20"]))
    return SourceData(bills, sittings, sessions, parliaments, days)


def test_only_finished_bills_and_target_is_days_to_royal_assent():
    df = build_dataset(_source())
    assert set(df["bill_id"]) == {1}
    row = df[df["anchor"] == "1st House: 1st reading"].iloc[0]
    assert row["target_days"] == 50  # Jan 1 → Feb 20


def test_stage_anchor_does_not_see_later_same_day_stages():
    df = build_dataset(_source()).set_index("anchor")
    at_2r = df.loc["1st House: 2nd reading"]
    assert at_2r["committee_whole_house"] == 0  # held later on Jan 10
    assert at_2r["h1_same_day_2r_3r"] == 0
    assert df.loc["1st House: committee", "committee_whole_house"] == 1
    assert df.loc["1st House: 3rd reading", "h1_same_day_2r_3r"] == 1
    assert pd.isna(at_2r["gap_h2_1r_2r"])  # 2nd House not reached yet
    assert df.loc["2nd House: 2nd reading", "gap_h2_1r_2r"] == 8


def test_time_anchor_uses_only_sittings_up_to_that_date():
    df = build_dataset(_source()).set_index("anchor")
    at_30 = df.loc["30 days in"]  # Jan 31: Lords committee (Jan 30) known, report not
    assert at_30["target_days"] == 20
    assert at_30["gap_h2_2r_cttee"] == 10
    assert pd.isna(at_30["gap_h2_cttee_3r"])
    assert "60 days in" not in df.index  # after Royal Assent


def test_feature_columns_exclude_identifiers_and_target():
    cols = feature_columns(build_dataset(_source()))
    assert "target_days" not in cols and "bill_id" not in cols and "as_of" not in cols
