from datetime import date

import numpy as np
import pandas as pd

from westminster_weather.model import future_calendar


def test_future_weeks_copy_past_clock_patterns():
    # History: Tuesday and Wednesday only, every week except ISO week 32.
    history = pd.DataFrame(
        {
            "date": [
                d.date()
                for d in pd.date_range("2018-01-01", "2023-12-31")
                if d.weekday() in (1, 2) and d.isocalendar()[1] != 32
            ],
            "house": "Commons",
        }
    )
    days = future_calendar.sample_future_days(
        history, date(2024, 1, 1), date(2024, 12, 31), np.random.default_rng(0)
    )
    assert {d.weekday() for d in days} == {1, 2}
    assert 32 not in {d.isocalendar()[1] for d in days}
    assert 95 <= len(days) <= 104  # about 2 days in each of ~51 weeks
    assert all(date(2024, 1, 1) < d <= date(2024, 12, 31) for d in days)
