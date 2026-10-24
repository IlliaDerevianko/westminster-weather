"""Parliaments, sessions and the sitting-day calendar (BUILD_PLAN task 2, MODEL_SPEC §3).

Sitting days use the MODEL_SPEC §3 fallback: every date on which any bill of any type had a
stage sitting. Sessions are estimated from those dates, then corrected by
``sessions_override.csv``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OVERRIDES_PATH = Path(__file__).with_name("sessions_override.csv")

# Parliaments covering the Bills API data (sessions 18 onwards). ``dissolution_date`` and
# ``election_date`` are those that ended the Parliament.
PARLIAMENTS = pd.DataFrame(
    [
        (1, date(2001, 6, 13), date(2005, 4, 11), date(2005, 5, 5)),
        (2, date(2005, 5, 11), date(2010, 4, 12), date(2010, 5, 6)),
        (3, date(2010, 5, 18), date(2015, 3, 30), date(2015, 5, 7)),
        (4, date(2015, 5, 18), date(2017, 5, 3), date(2017, 6, 8)),
        (5, date(2017, 6, 13), date(2019, 11, 6), date(2019, 12, 12)),
        (6, date(2019, 12, 17), date(2024, 5, 30), date(2024, 7, 4)),
        (7, date(2024, 7, 9), None, None),
    ],
    columns=["parliament_id", "first_meeting_date", "dissolution_date", "election_date"],
)

# A session ending this close before a dissolution died because of the election, not the
# normal end of a session (e.g. the wash-up prorogations of 2010, 2015, 2017 and 2024).
DISSOLUTION_PROXIMITY_DAYS = 14

# Sessions with fewer sitting days than this are flagged by the validation checks.
MIN_PLAUSIBLE_SITTING_DAYS = 50

FUTURE_HISTORY_YEARS = 5
FUTURE_SITTING_WEEKDAYS = (0, 1, 2, 3)  # Monday to Thursday


# --------------------------------------------------------------------------- parliaments


def parliaments() -> pd.DataFrame:
    """The ``parliaments`` reference table (DATA_CONTRACT §2.2)."""
    return PARLIAMENTS.copy()


def dissolution_gaps(parls: pd.DataFrame | None = None) -> list[tuple[date, date]]:
    """Periods with no Parliament: (day after dissolution, day before the next first meeting)."""
    parls = PARLIAMENTS if parls is None else parls
    gaps = []
    for (_, row), (_, nxt) in zip(parls.iterrows(), parls.iloc[1:].iterrows(), strict=False):
        if row["dissolution_date"] is not None:
            gaps.append(
                (
                    row["dissolution_date"] + timedelta(days=1),
                    nxt["first_meeting_date"] - timedelta(days=1),
                )
            )
    return gaps


def in_dissolution_gap(d: date, gaps: list[tuple[date, date]]) -> bool:
    return any(start <= d <= end for start, end in gaps)


# --------------------------------------------------------------------------- sitting days


def sitting_days(all_sittings: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, House) that House sat (DATA_CONTRACT §2.4).

    Royal Assent sittings are recorded as ``Unassigned``; Assent is notified to both Houses,
    so those dates count for both. Dates inside a dissolution gap are data errors and dropped.
    """
    gaps = dissolution_gaps()
    df = all_sittings.dropna(subset=["event_date"])[["event_date", "house"]]
    both = df[~df["house"].isin(["Commons", "Lords"])]
    df = pd.concat(
        [
            df[df["house"].isin(["Commons", "Lords"])],
            both.assign(house="Commons"),
            both.assign(house="Lords"),
        ]
    ).rename(columns={"event_date": "date"})
    df = df[[not in_dissolution_gap(d, gaps) for d in df["date"]]]
    return df.drop_duplicates().sort_values(["date", "house"]).reset_index(drop=True)


def sitting_dates(days: pd.DataFrame, house: str | None = None) -> np.ndarray:
    """Sorted unique sitting dates (``datetime64[D]``) for one House, or either House."""
    if house is not None:
        days = days[days["house"] == house]
    return np.unique(pd.to_datetime(days["date"]).to_numpy().astype("datetime64[D]"))


def count_sitting_days(dates: np.ndarray, after: date, until: date) -> int:
    """Number of sitting days d with ``after < d <= until``."""
    lo = np.searchsorted(dates, np.datetime64(after, "D"), side="right")
    hi = np.searchsorted(dates, np.datetime64(until, "D"), side="right")
    return int(max(hi - lo, 0))


# --------------------------------------------------------------------------- sessions


def _best_cut(earlier: list[date], later: list[date]) -> date:
    """Boundary date between two consecutive sessions that misplaces the fewest sittings.

    A handful of sittings in the API carry dates outside their session (e.g. a 2016 date
    filed under the 2014-15 session), so plain min/max would make sessions overlap."""
    candidates = sorted(set(earlier) | set(later))
    best, best_errors = candidates[0], None
    for cut in candidates:
        errors = sum(d > cut for d in earlier) + sum(d <= cut for d in later)
        if best_errors is None or errors < best_errors:
            best, best_errors = cut, errors
    return best


def _end_type(end: date, parls: pd.DataFrame) -> str:
    window = timedelta(days=DISSOLUTION_PROXIMITY_DAYS)
    dissolutions = parls["dissolution_date"].dropna()
    near = any(end <= d <= end + window for d in dissolutions)
    return "dissolution" if near else "prorogation"


def load_overrides(path: Path = OVERRIDES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"end_type": "string"})
    for col in ("start_date", "end_date"):
        df[col] = pd.to_datetime(df[col]).dt.date
    return df


def sessions(
    all_sittings: pd.DataFrame,
    overrides: pd.DataFrame | None = None,
    parls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The ``sessions`` table (DATA_CONTRACT §2.1).

    Start and end are estimated from sitting dates, then ``overrides`` (non-empty cells only)
    are applied. ``end_type`` is ``'dissolution'`` when the session ended within
    ``DISSOLUTION_PROXIMITY_DAYS`` before a dissolution, otherwise ``'prorogation'``.
    The latest session is current: ``end_date`` and ``end_type`` are NULL.
    """
    parls = PARLIAMENTS if parls is None else parls
    overrides = load_overrides() if overrides is None else overrides
    gaps = dissolution_gaps(parls)
    df = all_sittings.dropna(subset=["event_date"])
    df = df[[not in_dissolution_gap(d, gaps) for d in df["event_date"]]]
    by_session = {sid: sorted(g.unique()) for sid, g in df.groupby("session_id")["event_date"]}
    ids = sorted(by_session)

    starts = {ids[0]: by_session[ids[0]][0]}
    ends: dict[int, date | None] = {ids[-1]: None}
    for prev, nxt in zip(ids, ids[1:], strict=False):
        cut = _best_cut(by_session[prev], by_session[nxt])
        ends[prev] = max((d for d in by_session[prev] if d <= cut), default=cut)
        starts[nxt] = min((d for d in by_session[nxt] if d > cut), default=cut)

    out = pd.DataFrame(
        {
            "session_id": ids,
            "start_date": [starts[s] for s in ids],
            "end_date": [ends[s] for s in ids],
        }
    )
    out["end_type"] = [None if e is None else _end_type(e, parls) for e in out["end_date"]]

    for _, row in overrides.iterrows():
        mask = out["session_id"] == row["session_id"]
        for col in ("start_date", "end_date", "end_type"):
            if pd.notna(row[col]):
                out.loc[mask, col] = row[col]

    out["parliament_id"] = [_parliament_for(d, parls) for d in out["start_date"]]
    return out[["session_id", "parliament_id", "start_date", "end_date", "end_type"]]


def _parliament_for(d: date, parls: pd.DataFrame) -> int | None:
    match = parls[parls["first_meeting_date"] <= d]
    return int(match["parliament_id"].iloc[-1]) if len(match) else None


# --------------------------------------------------------------------------- future days


def week_sitting_probability(days: pd.DataFrame, as_of: date) -> pd.Series:
    """P(either House sits in ISO week w), from the last ``FUTURE_HISTORY_YEARS`` complete
    calendar years before ``as_of``. Weeks inside a dissolution gap are left out, because
    elections are simulated separately."""
    years = range(as_of.year - FUTURE_HISTORY_YEARS, as_of.year)
    gaps = dissolution_gaps()
    sat = {d.isocalendar()[:2] for d in pd.to_datetime(days["date"]).dt.date if d <= as_of}
    hits = np.zeros(54)
    totals = np.zeros(54)
    for year in years:
        monday = date.fromisocalendar(year, 1, 1)
        while monday.isocalendar()[0] == year:
            week = monday.isocalendar()[1]
            if not in_dissolution_gap(monday + timedelta(days=2), gaps):
                totals[week] += 1
                hits[week] += (year, week) in sat
            monday += timedelta(weeks=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(totals > 0, hits / totals, 0.0)
    return pd.Series(prob[1:], index=pd.RangeIndex(1, 54, name="iso_week"))


def future_sitting_days(
    start: date,
    end: date,
    week_probability: pd.Series,
    rng: np.random.Generator | None = None,
) -> list[date]:
    """Sitting days in ``(start, end]`` under the typical pattern (MODEL_SPEC §3).

    A week sits with its historical probability (random if ``rng`` is given, otherwise when
    the probability is at least 0.5); sitting weeks sit Monday to Thursday.
    """
    out = []
    week_sits: dict[tuple[int, int], bool] = {}
    d = start + timedelta(days=1)
    while d <= end:
        if d.weekday() in FUTURE_SITTING_WEEKDAYS:
            key = d.isocalendar()[:2]
            if key not in week_sits:
                p = float(week_probability.get(key[1], 0.0))
                week_sits[key] = bool(rng.random() < p) if rng is not None else p >= 0.5
            if week_sits[key]:
                out.append(d)
        d += timedelta(days=1)
    return out


# --------------------------------------------------------------------------- checks


def session_sitting_day_counts(sess: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """Sitting days (either House) per completed session, flagging implausibly few."""
    dates = sitting_dates(days)
    rows = []
    for _, s in sess.iterrows():
        if s["end_date"] is None or pd.isna(s["end_date"]):
            continue
        n = count_sitting_days(dates, s["start_date"] - timedelta(days=1), s["end_date"])
        rows.append(
            {
                "session_id": s["session_id"],
                "sitting_days": n,
                "flag": n < MIN_PLAUSIBLE_SITTING_DAYS,
            }
        )
    return pd.DataFrame(rows)
