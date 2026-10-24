from datetime import date

import numpy as np
import pandas as pd

from westminster_weather.data import calendar

NO_OVERRIDES = pd.DataFrame(columns=["session_id", "start_date", "end_date", "end_type"])


def _sittings(spec):
    return pd.DataFrame(
        [
            {"session_id": sid, "event_date": date.fromisoformat(d), "house": "Commons"}
            for sid, dates in spec.items()
            for d in dates
        ]
    )


def test_sessions_ignore_stray_dates():
    sittings = _sittings(
        {
            30: ["2017-06-22", "2019-10-08", "2019-10-16"],  # 2019-10-16 is a stray date
            34: ["2019-10-15", "2019-11-05"],
            35: ["2019-12-18"],
        }
    )
    sess = calendar.sessions(sittings, overrides=NO_OVERRIDES).set_index("session_id")
    assert sess.loc[30, "end_date"] == date(2019, 10, 8)
    assert sess.loc[30, "end_type"] == "prorogation"  # 29 days before the 2019 dissolution
    assert sess.loc[34, "end_type"] == "dissolution"  # 1 day before it
    assert sess.loc[35, "end_type"] is None  # current session
    assert sess.loc[35, "parliament_id"] == 6


def test_session_within_14_days_of_dissolution_is_dissolution():
    # 2024 wash-up: prorogued 24 May, dissolved 30 May
    sittings = _sittings({38: ["2023-11-08", "2024-05-24"], 39: ["2024-07-18"]})
    sess = calendar.sessions(sittings, overrides=NO_OVERRIDES).set_index("session_id")
    assert sess.loc[38, "end_type"] == "dissolution"


def test_overrides_and_dissolution_gap():
    # 2024-06-14 falls after the 2024-05-30 dissolution, so it cannot be a real sitting
    sittings = _sittings({38: ["2023-11-08", "2024-05-24", "2024-06-14"], 39: ["2024-07-18"]})
    sess = calendar.sessions(sittings, overrides=NO_OVERRIDES).set_index("session_id")
    assert sess.loc[38, "end_date"] == date(2024, 5, 24)
    days = calendar.sitting_days(sittings)
    assert date(2024, 6, 14) not in set(days["date"])

    override = pd.DataFrame(
        [{"session_id": 38, "start_date": None, "end_date": date(2024, 5, 20), "end_type": None}]
    )
    sess = calendar.sessions(sittings, overrides=override).set_index("session_id")
    assert sess.loc[38, "end_date"] == date(2024, 5, 20)
    assert sess.loc[38, "start_date"] == date(2023, 11, 8)


def test_override_file_sets_2023_24_session():
    overrides = calendar.load_overrides().set_index("session_id")
    assert overrides.loc[38, "end_date"] == date(2024, 5, 24)
    assert overrides.loc[38, "end_type"] == "dissolution"


def test_sitting_days_count_unassigned_for_both_houses():
    sittings = pd.DataFrame(
        [
            {"session_id": 30, "event_date": date(2018, 1, 10), "house": "Commons"},
            {"session_id": 30, "event_date": date(2018, 1, 11), "house": "Unassigned"},
        ]
    )
    days = calendar.sitting_days(sittings)
    assert set(map(tuple, days.to_numpy())) == {
        (date(2018, 1, 10), "Commons"),
        (date(2018, 1, 11), "Commons"),
        (date(2018, 1, 11), "Lords"),
    }
    dates = calendar.sitting_dates(days)
    assert calendar.count_sitting_days(dates, date(2018, 1, 10), date(2018, 1, 11)) == 1
    assert calendar.count_sitting_days(dates, date(2018, 1, 9), date(2018, 1, 9)) == 0


def test_future_sitting_days():
    # history: every Monday-Thursday in 2020-2024 except ISO week 32 (summer recess)
    history = [
        d.date()
        for d in pd.date_range("2020-01-01", "2024-12-31")
        if d.weekday() < 4 and d.isocalendar()[1] != 32
    ]
    days = pd.DataFrame({"date": history, "house": "Commons"})
    prob = calendar.week_sitting_probability(days, as_of=date(2025, 3, 1))
    assert prob[32] == 0.0
    assert prob[10] == 1.0

    future = calendar.future_sitting_days(date(2025, 7, 31), date(2025, 8, 15), prob)
    weeks = {d.isocalendar()[1] for d in future}
    assert 32 not in weeks and 33 in weeks
    assert all(d.weekday() < 4 for d in future)

    rng = np.random.default_rng(0)
    assert calendar.future_sitting_days(date(2025, 7, 31), date(2025, 8, 15), prob, rng) == future
