"""Outlook rules that need no database: weather thresholds, deadlines, curve sentences."""

from datetime import date

import numpy as np
import pandas as pd

from westminster_weather.model import outlook as o


def _dist(ra_dates: list[str], n: int, p_fail: float = 0.0) -> o.Distribution:
    return o.Distribution(
        as_of=date(2026, 10, 25),
        n_paths=n,
        ra_dates=np.sort(np.array(ra_dates, dtype="datetime64[D]")),
        p_fail=p_fail,
        session_end_dates=np.array([], dtype="datetime64[D]"),
    )


def test_weather_thresholds_and_storm_override():
    assert o.weather(0.80, 0.0) == "sunny"
    assert o.weather(0.79, 0.0) == "partly_cloudy"
    assert o.weather(0.50, 0.0) == "partly_cloudy"
    assert o.weather(0.49, 0.0) == "rain"
    assert o.weather(0.05, 0.10) == "rain"  # under 20% with low failure risk
    assert o.weather(0.95, 0.25) == "storm"  # failure risk wins whatever the date


def test_cumulative_curve_counts_paths():
    d = _dist(["2026-11-01", "2026-12-01", "2027-01-01"], n=4)
    assert d.p_royal_assent == 0.75
    assert d.p_by(date(2026, 10, 31)) == 0.0
    assert d.p_by(date(2026, 12, 1)) == 0.5  # on the date counts
    assert d.p_by(date(2030, 1, 1)) == 0.75


def test_next_quarter_ends_skip_today():
    assert o.next_quarter_ends(date(2026, 12, 31)) == [
        date(2027, 3, 31),
        date(2027, 6, 30),
        date(2027, 9, 30),
        date(2027, 12, 31),
    ]


def test_recess_from_recorded_sittings_and_estimate_on_a_thursday():
    sitting = pd.DatetimeIndex(pd.to_datetime(["2025-12-17", "2025-12-18", "2026-01-05"]))
    assert o.recess_start("christmas", 2025, sitting) == date(2025, 12, 18)
    est = o.typical_recess_start("christmas", 2026, sitting)
    assert est.weekday() <= 3 and est.month == 12


def test_curve_sentence_months():
    dates = [f"2027-0{m}-15" for m in (1, 2, 3)] * 3 + ["2027-06-15"]
    c = o.curve_block(_dist(dates, n=10))
    assert c["sentence"] == "Most likely by February 2027; very likely by June 2027."
