from datetime import date

import pandas as pd

from westminster_weather.analysis.stage_durations import (
    SEGMENTS,
    segment_durations,
    spearman_matrix,
    stage_entry_dates,
)


def _sittings(rows: list[tuple[int, int, date]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"bill_id": b, "bill_type_id": 1, "api_stage_id": sid, "event_date": d}
            for b, sid, d in rows
        ]
    )


def test_entry_dates_use_first_sitting_and_merge_absorbed_stage_ids():
    # 15 (money resolution) is absorbed into Commons 2nd reading; the earliest date wins.
    entry = stage_entry_dates(
        _sittings(
            [
                (1, 6, date(2020, 1, 1)),
                (1, 7, date(2020, 1, 20)),
                (1, 15, date(2020, 1, 10)),
                (1, 99, date(2019, 1, 1)),  # unmapped stage id is ignored
            ]
        )
    )
    assert entry.loc[1, "commons_1st_reading"] == pd.Timestamp("2020-01-01")
    assert entry.loc[1, "commons_2nd_reading"] == pd.Timestamp("2020-01-10")


def test_calendar_and_sitting_day_durations():
    entry = stage_entry_dates(
        _sittings([(1, 6, date(2020, 1, 1)), (1, 7, date(2020, 1, 11)), (2, 6, date(2020, 1, 1))])
    )
    cal = segment_durations(entry, SEGMENTS, sitting_dates=None)
    assert cal.loc[1, "c_1r_2r"] == 10
    assert pd.isna(cal.loc[2, "c_1r_2r"])  # bill 2 never reached 2nd reading

    sitting = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-08", "2020-01-11"])
    sit = segment_durations(entry, SEGMENTS, sitting_dates=sitting)
    assert sit.loc[1, "c_1r_2r"] == 3  # sitting days in (Jan 1, Jan 11]


def test_spearman_matrix_counts_only_bills_with_both_durations():
    df = pd.DataFrame({"a": range(20), "b": [x * 2 for x in range(20)]})
    df.loc[0, "b"] = None
    rho, _, n = spearman_matrix(df)
    assert n.loc["a", "b"] == 19
    assert rho.loc["a", "b"] == 1.0
