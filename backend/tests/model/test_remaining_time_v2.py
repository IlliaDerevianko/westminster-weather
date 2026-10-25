import numpy as np
import pandas as pd

from westminster_weather.analysis.remaining_time_dataset import (
    SourceData,
    _days_to_summer_recess,
    build_dataset,
)
from westminster_weather.analysis.remaining_time_v2 import CV_FOLDS, Design, prepare, to_days


def _source() -> SourceData:
    bills = pd.DataFrame(
        [{"bill_id": 1, "short_title": "Northern Ireland (No. 2) Bill",
          "originating_house": "Commons", "introduced_session_id": 30,
          "department": "Northern Ireland Office", "outcome": "royal_assent",
          "outcome_date": pd.Timestamp("2020-02-20")}]
    )
    rows = [
        (6, "Commons", "2020-01-01", 1), (7, "Commons", "2020-01-05", 2),
        (8, "Commons", "2020-01-07", 3), (8, "Commons", "2020-01-08", 4),
        (10, "Commons", "2020-01-10", 5), (1, "Lords", "2020-01-12", 6),
        (2, "Lords", "2020-01-20", 7), (11, "Lords", "2020-02-20", 8),
    ]
    sittings = pd.DataFrame(
        [{"bill_id": 1, "bill_type_id": 1, "api_stage_id": sid, "house": h,
          "event_date": pd.Timestamp(d), "sort_order": o} for sid, h, d, o in rows]
    )
    sessions = pd.DataFrame(
        {"session_id": [30], "parliament_id": [5], "start_date": [pd.Timestamp("2019-12-01")]}
    )
    parliaments = pd.DataFrame(
        {"parliament_id": [5], "first_meeting_date": [pd.Timestamp("2017-06-13")]}
    )
    return SourceData(bills, sittings, sessions, parliaments, pd.Series(dtype="datetime64[ns]"))


def test_extended_features_measure_time_in_each_house_and_sittings():
    df = build_dataset(_source(), extended=True).set_index("anchor")
    at_l2r = df.loc["2nd House: 2nd reading"]
    assert at_l2r["commons_days_so_far"] == 9  # Jan 1 → Commons 3rd reading Jan 10
    assert at_l2r["lords_days_so_far"] == 8  # Jan 12 → Jan 20, still in the Lords
    assert at_l2r["n_h1_committee_sittings"] == 2
    assert at_l2r["first_house_finished"] == 1
    at_cttee = df.loc["1st House: committee"]
    assert at_cttee["n_h1_committee_sittings"] == 1  # the second sitting is not known yet
    assert np.isnan(at_cttee["lords_days_so_far"])
    assert at_cttee["policy_area"] == "northern_ireland"
    assert at_cttee["title_northern_ireland"] == 1 and at_cttee["title_numbered"] == 1
    assert at_cttee["government_majority"] == -16 and at_cttee["coalition_or_minority"] == 1


def test_summer_recess_countdown():
    assert _days_to_summer_recess(pd.Timestamp("2020-07-01")) == (20, 0)
    assert _days_to_summer_recess(pd.Timestamp("2020-08-01")) == (0, 1)
    assert _days_to_summer_recess(pd.Timestamp("2020-10-01"))[0] == 293


def _encoding_rows() -> pd.DataFrame:
    n = 20
    return pd.DataFrame(
        {
            "bill_id": range(n),
            "anchor": ["1st House: 1st reading"] * n,
            "anchor_type": ["stage"] * n,
            "origin_lords": [0] * n,
            "stage_index": [0] * n,
            "elapsed_days": [0] * n,
            "target_days": [100] * (n - 1) + [10_000],
            "policy_area": ["treasury"] * (n - 1) + ["unique_area"],
        }
    )


def test_training_encodings_are_out_of_fold():
    train_enc, _ = prepare(_encoding_rows(), None)
    big = train_enc[train_enc["bill_id"] == 19].iloc[0]
    # Its own 10,000-day target must not feed its rule-of-thumb or area features.
    assert np.isclose(big["b1_log"], np.log1p(100))
    assert np.isclose(big["area_speed_log"], np.log1p(100))
    assert big["area_resid_log"] == 0.0


def test_other_rows_use_all_training_rows():
    train = _encoding_rows()
    other = train.iloc[[0]].assign(bill_id=99)
    _, other_enc = prepare(train, other)
    assert np.isclose(other_enc["b1_log"].iloc[0], np.log1p(100))  # median of all 20


def test_residual_target_round_trips_to_days():
    df = pd.DataFrame({"b1_log": [np.log1p(50.0)]})
    d = Design(features=[], target="resid")
    assert np.isclose(to_days(np.array([0.0]), df, d)[0], 50.0)


def test_cv_folds_validate_on_later_sessions_only():
    for train, val in CV_FOLDS:
        assert max(train) < min(val) and min(train) == 20
