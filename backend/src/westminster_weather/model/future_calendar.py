"""Future sitting days for the simulator, matching the density of the training clock.

The hazard is fitted per sitting day of the training clock (dates with a government-bill
stage sitting in either House), which averages about 3.5 days in a sitting week and ~36
sitting weeks a year. The generator in ``data/calendar.py`` sits Monday to Thursday in every
likely sitting week, about 20% more days, which made simulated bills too fast in calendar
time. Here each future ISO week copies the actual clock days of the same ISO week in a past
year drawn at random from the last ``HISTORY_YEARS`` complete years before ``as_of`` (weeks
inside a dissolution gap are skipped, because elections are simulated separately).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from westminster_weather.data import calendar

HISTORY_YEARS = 5


def sample_future_days(
    sitting_days: pd.DataFrame,
    as_of: date,
    end: date,
    rng: np.random.Generator,
    parliaments: pd.DataFrame | None = None,
) -> list[date]:
    """Sitting days in ``(as_of, end]``, week by week from past years' clock patterns."""
    gaps = calendar.dissolution_gaps(parliaments)
    years = range(as_of.year - HISTORY_YEARS, as_of.year)
    pattern: dict[tuple[int, int], set[int]] = defaultdict(set)
    for d in pd.to_datetime(sitting_days["date"]).dt.date.unique():
        if d.year in years:
            year, week, weekday = d.isocalendar()
            pattern[(year, week)].add(weekday - 1)

    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for year in years:
        monday = date.fromisocalendar(year, 1, 1)
        while monday.isocalendar()[0] == year:
            week = monday.isocalendar()[1]
            if not calendar.in_dissolution_gap(monday + timedelta(days=2), gaps):
                candidates[week].append((year, week))
            monday += timedelta(weeks=1)

    out: list[date] = []
    monday = as_of - timedelta(days=as_of.weekday())
    while monday <= end:
        week = monday.isocalendar()[1]
        options = candidates.get(week) or candidates.get(min(week, 52)) or []
        if options:
            year_week = options[rng.integers(len(options))]
            for weekday in sorted(pattern.get(year_week, ())):
                d = monday + timedelta(days=weekday)
                if as_of < d <= end:
                    out.append(d)
        monday += timedelta(weeks=1)
    return out
