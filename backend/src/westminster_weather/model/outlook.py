"""Presentation-ready bill outlooks computed from the passage model's simulated paths.

Nothing here changes how the model predicts: every number is counted from the same simulated
paths ``forecast_bill`` summarises (``forecast.run_paths``). The cumulative curve is
P(Royal Assent by d) = share of paths that reach Royal Assent on or before d, which equals
p_royal_assent x P(date <= d | passes).

Sections (numbered as in the outlook spec):
0. full distribution: p_royal_assent, 5%-95% quantiles of the Royal Assent date
1. deadline probabilities: next Christmas / Easter / summer recess, next 4 quarter ends,
   expected session end, optional custom date
2. weather status for a target date
3. weekly probability curve for two years
5. pace compared with typical bills (``pace``)
7. next milestone and what-if (``next_milestone``)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from dateutil.easter import easter

from westminster_weather.model.simulate import OUTCOME_REASON, ROYAL_ASSENT, SimResult

# All presentation thresholds in one place.
OUTLOOK_CONFIG: dict[str, float] = {
    "sunny_min": 0.80,  # chance by the target date >= this: "On track"
    "partly_cloudy_min": 0.50,  # >= this: "Changeable"
    "rain_min": 0.20,  # >= this: "Delays likely" (and below it when failure risk is low)
    "storm_fail_min": 0.25,  # chance the bill dies >= this: "Storm warning", whatever the date
    "target_min_days": 30,  # default target: next recess deadline at least this far away
    "curve_weeks": 104,  # weekly curve length (two years)
    "likely": 0.50,  # "most likely by <month>"
    "very_likely": 0.90,  # "very likely by <month>"
}
WEATHER = {
    "sunny": "On track",
    "partly_cloudy": "Changeable",
    "rain": "Delays likely",
    "storm": "Storm warning",
}
RECESS_LABEL = {
    "christmas": "the Christmas recess",
    "easter": "the Easter recess",
    "summer": "the summer recess",
}


def fmt_date(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def pct(p: float) -> str:
    return f"{round(100 * p)}%"


# ------------------------------------------------------------------- 0. distribution


@dataclass
class Distribution:
    """Royal Assent dates of the simulated paths, and how the others ended."""

    as_of: date
    n_paths: int
    ra_dates: np.ndarray  # datetime64[D] of the paths that reach Royal Assent
    p_fail: float  # paths that end in a failure (defeat, withdrawal, session end, election)
    session_end_dates: np.ndarray  # datetime64[D] simulated end of the current session

    @classmethod
    def from_result(cls, result: SimResult) -> Distribution:
        out = result.outcome
        ra = out == ROYAL_ASSENT
        failed = np.isin(out, list(OUTCOME_REASON))
        days = result.days
        s_end = np.clip(result.first_session_end, 1, len(days))
        return cls(
            as_of=result.start.as_of,
            n_paths=len(out),
            ra_dates=np.sort(result.date_of(result.end_day[ra]).astype("datetime64[D]")),
            p_fail=float(failed.mean()),
            session_end_dates=days[s_end - 1].astype("datetime64[D]"),
        )

    @property
    def p_royal_assent(self) -> float:
        return len(self.ra_dates) / self.n_paths

    def p_by(self, d: date) -> float:
        """P(Royal Assent on or before ``d``)."""
        k = np.searchsorted(self.ra_dates, np.datetime64(d, "D"), side="right")
        return float(k) / self.n_paths

    def quantiles(self) -> dict[str, str | None]:
        """Royal Assent date quantiles 5%-95% among paths that pass."""
        if not len(self.ra_dates):
            return {f"q{q:02d}": None for q in range(5, 100, 5)}
        return {
            f"q{q:02d}": str(
                np.percentile(self.ra_dates.astype(int), q, method="lower").astype("datetime64[D]")
            )
            for q in range(5, 100, 5)
        }


# --------------------------------------------------------------------- 1. deadlines


def _sitting_dates(sitting_days: pd.DataFrame) -> pd.DatetimeIndex:
    """Commons sitting days (recess dates are the Commons rising)."""
    if "house" in sitting_days:
        sitting_days = sitting_days[sitting_days["house"].str.lower() == "commons"]
    return pd.DatetimeIndex(sorted(pd.to_datetime(sitting_days["date"]).unique()))


def _last_thursday_on_or_before(d: date) -> date:
    """Estimated recess dates: the Commons usually rises on a Monday-Thursday."""
    return d - timedelta(days=(d.weekday() - 3) % 7) if d.weekday() >= 4 else d


def recess_start(kind: str, year: int, sitting: pd.DatetimeIndex) -> date | None:
    """Last sitting day before the recess in ``year`` from the recorded calendar, or None."""
    if kind == "christmas":
        lo, hi = pd.Timestamp(year, 12, 1), pd.Timestamp(year, 12, 24)
    elif kind == "summer":
        lo, hi = pd.Timestamp(year, 7, 1), pd.Timestamp(year, 8, 31)
    else:
        e = pd.Timestamp(easter(year))
        lo, hi = e - pd.Timedelta(days=21), e - pd.Timedelta(days=1)
    days = sitting[(sitting >= lo) & (sitting <= hi)]
    if not len(days):
        return None
    last = days.max()
    # the recess must actually follow: next sitting day at least 10 days later
    later = sitting[sitting > last]
    if len(later) and (later.min() - last).days < 10:
        return None
    return last.date()


def typical_recess_start(kind: str, year: int, sitting: pd.DatetimeIndex) -> date:
    """Typical date from past years (median day offset), for years with no recorded dates."""
    past = range(year - 12, year)
    if kind == "easter":
        offsets = [
            (e - d).days for y in past if (d := recess_start(kind, y, sitting)) and (e := easter(y))
        ]
        est = easter(year) - timedelta(days=int(np.median(offsets)) if offsets else 5)
        return _last_thursday_on_or_before(est)
    ref = date(year, 12, 25) if kind == "christmas" else date(year, 7, 1)
    offsets = [
        (d - ref.replace(year=y)).days for y in past if (d := recess_start(kind, y, sitting))
    ]
    default = -7 if kind == "christmas" else 21
    return _last_thursday_on_or_before(
        ref + timedelta(days=int(np.median(offsets)) if offsets else default)
    )


def next_recess(kind: str, as_of: date, sitting: pd.DatetimeIndex) -> tuple[date, bool]:
    """(start of the next ``kind`` recess after ``as_of``, estimated?)."""
    for year in (as_of.year, as_of.year + 1, as_of.year + 2):
        actual = recess_start(kind, year, sitting)
        if actual is not None and actual > as_of:
            return actual, False
        if actual is None:
            est = typical_recess_start(kind, year, sitting)
            if est > as_of:
                return est, True
    raise ValueError(f"no {kind} recess found after {as_of}")


def next_quarter_ends(as_of: date, k: int = 4) -> list[date]:
    out, y, q = [], as_of.year, (as_of.month - 1) // 3 + 1
    while len(out) < k:
        end = (pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)).date()
        if end > as_of:
            out.append(end)
        q += 1
        if q > 4:
            q, y = 1, y + 1
    return out


def deadline_list(
    dist: Distribution,
    sitting: pd.DatetimeIndex,
    announced_session_end: date | None = None,
    custom: date | None = None,
) -> list[dict[str, Any]]:
    """Every deadline with its probability and a ready-to-show sentence."""
    as_of = dist.as_of
    items: list[dict[str, Any]] = []
    for kind in ("christmas", "easter", "summer"):
        d, est = next_recess(kind, as_of, sitting)
        items.append(
            {
                "kind": f"{kind}_recess",
                "label": f"before {RECESS_LABEL[kind]}",
                "date": d,
                "estimated": est,
            }
        )
    for i, d in enumerate(next_quarter_ends(as_of), start=1):
        q = (d.month - 1) // 3 + 1
        items.append(
            {
                "kind": f"quarter_end_{i}",
                "label": f"by the end of Q{q} {d.year}",
                "date": d,
                "estimated": False,
            }
        )
    if announced_session_end is not None:
        items.append(
            {
                "kind": "session_end",
                "label": "before the end of the current session",
                "date": announced_session_end,
                "estimated": False,
            }
        )
    elif len(dist.session_end_dates):
        median = pd.Timestamp(
            np.median(dist.session_end_dates.astype(int)).astype("datetime64[D]")
        ).date()
        if median > as_of:
            items.append(
                {
                    "kind": "session_end",
                    "label": "before the expected end of the current session",
                    "date": median,
                    "estimated": True,
                }
            )
    if custom is not None:
        items.append(
            {"kind": "custom", "label": "by the chosen date", "date": custom, "estimated": False}
        )
    for it in sorted(items, key=lambda x: x["date"]):
        it["p"] = dist.p_by(it["date"])
        note = ", estimated" if it["estimated"] else ""
        what = it["label"].replace("before ", "").replace("by ", "")
        it["sentence"] = (
            f"{pct(it['p'])} chance of Royal Assent {it['label']} ({fmt_date(it['date'])}{note})."
            if it["kind"] != "custom"
            else f"{pct(it['p'])} chance of Royal Assent by {fmt_date(it['date'])}."
        )
        it.setdefault("what", what)
    return sorted(items, key=lambda x: x["date"])


# ----------------------------------------------------------------------- 2. weather


def default_target(deadlines: list[dict[str, Any]], as_of: date) -> dict[str, Any]:
    """The next recess deadline at least ``target_min_days`` away."""
    soon = as_of + timedelta(days=int(OUTLOOK_CONFIG["target_min_days"]))
    recesses = [d for d in deadlines if d["kind"].endswith("_recess") and d["date"] >= soon]
    return min(recesses, key=lambda d: d["date"])


def weather(p_by_target: float, p_fail: float) -> str:
    c = OUTLOOK_CONFIG
    if p_fail >= c["storm_fail_min"]:
        return "storm"
    if p_by_target >= c["sunny_min"]:
        return "sunny"
    if p_by_target >= c["partly_cloudy_min"]:
        return "partly_cloudy"
    return "rain"  # 20-50%, or under 20% with low failure risk (high risk is a storm above)


def weather_block(dist: Distribution, target: dict[str, Any], short_title: str) -> dict[str, Any]:
    p = dist.p_by(target["date"])
    status = weather(p, dist.p_fail)
    when = f"{target['label']} ({fmt_date(target['date'])})"
    if status == "storm":
        headline = (
            f"Storm warning: {pct(dist.p_fail)} chance {short_title} fails before "
            f"becoming law; {pct(p)} chance of Royal Assent {when}."
        )
    else:
        headline = f"{WEATHER[status]}: {pct(p)} chance of Royal Assent {when}."
    return {
        "status": status,
        "label": WEATHER[status],
        "target_date": target["date"],
        "target_label": target["label"],
        "target_estimated": target.get("estimated", False),
        "p_by_target": p,
        "p_fail": dist.p_fail,
        "headline": headline,
    }


# ------------------------------------------------------------------------- 3. curve


def curve_block(dist: Distribution) -> dict[str, Any]:
    weeks = int(OUTLOOK_CONFIG["curve_weeks"])
    points = [
        {
            "date": dist.as_of + timedelta(weeks=w),
            "p_law_by_date": dist.p_by(dist.as_of + timedelta(weeks=w)),
        }
        for w in range(weeks + 1)
    ]

    def first_month(threshold: float) -> date | None:
        hit = next((p["date"] for p in points if p["p_law_by_date"] >= threshold), None)
        return None if hit is None else hit.replace(day=1)

    m50, m90 = first_month(OUTLOOK_CONFIG["likely"]), first_month(OUTLOOK_CONFIG["very_likely"])
    if m50 is None:
        sentence = (
            f"Less than a 50% chance of becoming law within two years "
            f"({pct(points[-1]['p_law_by_date'])} by {fmt_date(points[-1]['date'])})."
        )
    elif m90 is None:
        sentence = (
            f"Most likely by {m50:%B %Y}; less than 90% likely within two years "
            f"({pct(points[-1]['p_law_by_date'])})."
        )
    else:
        sentence = f"Most likely by {m50:%B %Y}; very likely by {m90:%B %Y}."
    return {"points": points, "month_50": m50, "month_90": m90, "sentence": sentence}
