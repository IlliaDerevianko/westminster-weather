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

    def p_law_by(self, d: date) -> float:
        """P(Royal Assent on or before ``d``) as shown: after recalibration, if any."""
        return calibrate(self.p_by(d))

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
        it["p"] = dist.p_law_by(it["date"])
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
    p = dist.p_law_by(target["date"])
    status = weather(p, dist.p_fail)
    est = ", estimated" if target.get("estimated") else ""
    when = f"{target['label']} ({fmt_date(target['date'])}{est})"
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
            "p_law_by_date": dist.p_law_by(dist.as_of + timedelta(weeks=w)),
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


# -------------------------------------------------------------------------- 5. pace

STAGE_NAMES = {
    "commons_1st_reading": "Commons 1st reading",
    "commons_2nd_reading": "Commons 2nd reading",
    "commons_committee": "Commons committee stage",
    "commons_report_stage": "Commons report stage",
    "commons_3rd_reading": "Commons 3rd reading",
    "lords_1st_and_2nd_reading": "Lords 1st and 2nd reading",
    "lords_committee": "Lords committee stage",
    "lords_report_and_3rd_reading": "Lords report and 3rd reading",
    "ping_pong": "ping-pong",
    "royal_assent": "Royal Assent",
}


def stage_name(state: str) -> str:
    return STAGE_NAMES.get(state, state.replace("_", " "))


def pace_block(
    bill_id: int, as_of: date, trans_all: pd.DataFrame, bills: pd.DataFrame, dates: np.ndarray
) -> dict[str, Any]:
    """Sitting days from 1st reading to the bill's current stage, against bills that had
    finished before ``as_of`` and reached the same stage (same originating House)."""
    from westminster_weather.data.calendar import count_sitting_days

    mine = trans_all[trans_all["bill_id"] == bill_id].sort_values("state_order")
    first, current = mine.iloc[0], mine.iloc[-1]
    state = current["model_state"]
    base = {"current_stage": state, "current_stage_name": stage_name(state)}
    if len(mine) == 1:
        return {
            **base,
            "sitting_days_so_far": 0,
            "typical_sitting_days": None,
            "weeks_ahead_or_behind": None,
            "pace_percentile": None,
            "n_comparison_bills": 0,
            "sentence": f"Just introduced ({stage_name(state)}): too early to compare its "
            "pace with past bills.",
        }
    mine_days = count_sitting_days(dates, first["entered_date"], current["entered_date"])

    done = bills[(bills["outcome"] != "in_progress") & (bills["outcome_date"] < as_of)]
    peers = trans_all[
        trans_all["bill_id"].isin(done["bill_id"])
        & (trans_all["originating_house"] == current["originating_house"])
    ]
    starts = peers.sort_values("state_order").groupby("bill_id")["entered_date"].first()
    reached = peers[peers["model_state"] == state].groupby("bill_id")["entered_date"].first()
    peer_days = np.array([count_sitting_days(dates, starts[b], reached[b]) for b in reached.index])
    if len(peer_days) < 10:
        return {
            **base,
            "sitting_days_so_far": mine_days,
            "typical_sitting_days": None,
            "weeks_ahead_or_behind": None,
            "pace_percentile": None,
            "n_comparison_bills": len(peer_days),
            "sentence": f"Too few past bills reached {stage_name(state)} to compare pace.",
        }

    typical = float(np.median(peer_days))
    # sitting days per sitting week over the year before as_of
    year = dates[
        (dates > np.datetime64(as_of - timedelta(days=365), "D"))
        & (dates <= np.datetime64(as_of, "D"))
    ]
    weeks_with_sittings = len({pd.Timestamp(d).isocalendar()[:2] for d in year}) or 1
    per_week = max(len(year) / weeks_with_sittings, 1.0)
    weeks = round((mine_days - typical) / per_week)
    slower_than = float(np.mean(peer_days < mine_days))
    if weeks > 0:
        lead = f"Running {weeks} week{'s' if weeks != 1 else ''} behind a typical bill"
    elif weeks < 0:
        lead = f"Running {-weeks} week{'s' if weeks != -1 else ''} ahead of a typical bill"
    else:
        lead = "Running at the pace of a typical bill"
    sentence = f"{lead} at {stage_name(state)} (slower than {pct(slower_than)} of past bills)."
    return {
        **base,
        "sitting_days_so_far": mine_days,
        "typical_sitting_days": typical,
        "weeks_ahead_or_behind": weeks,
        "pace_percentile": slower_than,
        "n_comparison_bills": len(peer_days),
        "sentence": sentence,
    }


# ------------------------------------------------------------- 7. next milestone, what-if


def _change(before: float, after: float) -> str:
    if round(100 * after) > round(100 * before):
        return f"rises from {pct(before)} to {pct(after)}"
    if round(100 * after) < round(100 * before):
        return f"falls from {pct(before)} to {pct(after)}"
    return f"stays at {pct(after)}"


def next_milestone_block(
    bill_id: int, as_of: date, base: Any, target: dict[str, Any], n: int
) -> dict[str, Any] | None:
    """The next stage and its date (scheduled sitting, else the model's median entry date),
    and how the chance by the target date changes if it happens on that date."""
    from westminster_weather.model.forecast import run_paths, scheduled_entry

    start = base.start
    k = start.k0 + 1
    if k >= len(start.chain):
        return None
    state = start.chain[k]
    sched = scheduled_entry(start, bill_id, as_of, base.src)
    if sched is not None and sched[0] == k:
        when, kind = sched[1], "scheduled"
    else:
        reached = base.result.entry_day[:, k] >= 0
        if not reached.any():
            return None
        day = int(np.percentile(base.result.entry_day[reached, k], 50, method="lower"))
        when, kind = pd.Timestamp(base.result.date_of(day)).date(), "expected"
    out = {
        "stage": state,
        "stage_name": stage_name(state),
        "date": when,
        "date_kind": kind,
        "p_by_target_without": None,
        "p_by_target_with": None,
    }
    shown = f"{fmt_date(when)} ({kind})"
    out["sentence"] = f"Next: {stage_name(state)}, {shown}."
    if state == "royal_assent" or when <= as_of:
        return out
    free = run_paths(
        bill_id, as_of, variant=base.variant, n=n, models=base.models, use_schedule=False
    )
    forced = run_paths(
        bill_id, as_of, variant=base.variant, n=n, models=base.models, next_entry=(k, when)
    )
    before = calibrate(Distribution.from_result(free.result).p_by(target["date"]))
    after = calibrate(Distribution.from_result(forced.result).p_by(target["date"]))
    if kind == "scheduled":
        lead = f"If {stage_name(state)} happens on {fmt_date(when)} as scheduled"
    else:  # pinning the median date rules out the faster paths
        lead = f"If {stage_name(state)} only happens on {fmt_date(when)} (its expected date)"
    out.update(
        p_by_target_without=before,
        p_by_target_with=after,
        what_if_sentence=f"{lead}, the chance of law {target['label']} {_change(before, after)}.",
    )
    return out


# ------------------------------------------------------------------ recalibration hook

_CALIBRATOR: Any = None


def calibrate(p: float) -> float:
    """Deadline probabilities after isotonic recalibration (fitted on training-era forecasts
    only by ``outlook_calibration``, stored in ``reports/deadline_calibration.json``), or
    unchanged when the calibration check did not call for it."""
    global _CALIBRATOR
    if _CALIBRATOR is None:
        import json

        from westminster_weather.model.outlook_calibration import METRICS_JSON

        iso = None
        if METRICS_JSON.exists():
            iso = json.loads(METRICS_JSON.read_text()).get("isotonic")
        _CALIBRATOR = (np.array(iso["x"]), np.array(iso["y"])) if iso else False
    if _CALIBRATOR is False:
        return float(p)
    x, y = _CALIBRATOR
    return float(np.interp(p, x, y))


def is_recalibrated() -> bool:
    calibrate(0.5)
    return _CALIBRATOR is not False


# ---------------------------------------------------------------------- assembling


class NotInParliament(ValueError):
    """The bill already reached Royal Assent or failed by the forecast date."""


def build_outlook(
    bill_id: int, as_of: date, target_date: date | None = None, n: int | None = None
) -> Any:
    """The full ``BillOutlook`` for a bill in Parliament on ``as_of``."""
    from datetime import UTC, datetime

    from westminster_weather.data.calendar import sitting_dates
    from westminster_weather.model import states
    from westminster_weather.model.forecast import run_paths
    from westminster_weather.model.simulate import N_PATHS
    from westminster_weather.schemas.outlook import BillOutlook

    n = n or N_PATHS
    base = run_paths(bill_id, as_of, n=n)
    if base.start is None:
        raise NotInParliament(f"bill {bill_id} is not in Parliament on {as_of}")
    src = base.src
    cal = src.get_calendar(as_of)
    dist = Distribution.from_result(base.result)
    deadlines = deadline_list(
        dist, _sitting_dates(cal["sitting_days"]), base.start.announced_session_end, target_date
    )
    target = (
        next(d for d in deadlines if d["kind"] == "custom")
        if target_date
        else default_target(deadlines, as_of)
    )
    trans_all = states.build_transitions(as_of, src)
    pace = pace_block(
        bill_id, as_of, trans_all, src.get_bills(as_of), sitting_dates(cal["sitting_days"])
    )
    nxt = next_milestone_block(bill_id, as_of, base, target, n)
    curve = curve_block(dist)
    wthr = weather_block(dist, target, base.bill["short_title"])
    fitted = getattr(base.models.failure, "fitted_as_of", None)
    sentences = [wthr["headline"], curve["sentence"]]
    sentences += [d["sentence"] for d in deadlines]
    sentences.append(pace["sentence"])
    if nxt is not None:
        sentences += [nxt["sentence"]] + (
            [nxt["what_if_sentence"]] if nxt.get("what_if_sentence") else []
        )
    state = base.start.chain[base.start.k0]
    return BillOutlook(
        bill_id=bill_id,
        short_title=base.bill["short_title"],
        generated_at=datetime.now(UTC),
        forecast_date=as_of,
        model_version=f"passage-{base.variant}-{fitted}+outlook-v1",
        recalibrated=is_recalibrated(),
        current_stage=state,
        current_stage_name=stage_name(state),
        distribution={
            "n_paths": dist.n_paths,
            "p_royal_assent": dist.p_royal_assent,
            "p_fail": dist.p_fail,
            "royal_assent_date_quantiles": dist.quantiles(),
        },
        deadlines=[
            {k: d[k] for k in ("kind", "label", "date", "estimated", "p", "sentence")}
            for d in deadlines
        ],
        weather=wthr,
        curve=curve,
        pace=pace,
        next_milestone=nxt,
        sentences=sentences,
    )
