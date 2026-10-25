from datetime import date

import pandas as pd

from westminster_weather.analysis.origin_comparison import house_durations


def _sittings(rows: list[tuple[int, int, date]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"bill_id": b, "bill_type_id": 1, "api_stage_id": sid, "event_date": d}
            for b, sid, d in rows
        ]
    )


def test_house_durations_follow_the_originating_house():
    bills = pd.DataFrame(
        {
            "bill_id": [1, 2],
            "originating_house": ["Commons", "Lords"],
            "outcome": ["royal_assent", "in_progress"],
        }
    )
    sittings = _sittings(
        [
            # Bill 1: Commons 1st reading Jan 1, 3rd reading Jan 31; Lords Feb 1 → Mar 2; RA Mar 12.
            (1, 6, date(2020, 1, 1)),
            (1, 10, date(2020, 1, 31)),
            (1, 1, date(2020, 2, 1)),
            (1, 5, date(2020, 3, 2)),
            (1, 11, date(2020, 3, 12)),
            # Bill 2: Lords-first, Lords done in 20 days, Commons not finished yet.
            (2, 1, date(2021, 1, 1)),
            (2, 5, date(2021, 1, 21)),
            (2, 6, date(2021, 1, 22)),
        ]
    )
    d = house_durations(bills, sittings)

    assert d.loc[1, "first_house_days"] == 30  # Commons
    assert d.loc[1, "second_house_days"] == 30  # Lords
    assert d.loc[1, "end_game_days"] == 10
    assert d.loc[1, "total_to_ra_days"] == 71
    assert not d.loc[1, "fast_tracked"]

    assert d.loc[2, "first_house_days"] == 20  # Lords goes first
    assert pd.isna(d.loc[2, "second_house_days"])  # Commons 3rd reading not reached
    assert pd.isna(d.loc[2, "total_to_ra_days"])
