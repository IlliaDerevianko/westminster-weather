"""Failure routes (BUILD_PLAN task 8, MODEL_SPEC §4.2, §4.3, §4.5).

- **Session end S:** an announced prorogation, else a draw from past session lengths (in
  sitting days) conditional on the session having lasted as long as it has. Only sessions
  ended by prorogation are used: sessions cut short by an election are the dissolution
  route below, so including them would count election risk twice.
- **What a prorogation does to an unfinished bill:** the government times the session end
  around its bills, so reaching S does not simply kill a bill. A bill still in progress at S
  **dies**, is **carried over** (keeps its stage and time in stage and continues in the next
  session) or is **rushed through** (the session is extended until it finishes, at
  end-of-session speed), with probabilities learned from government bills in progress in
  the last ``SESSION_END_WINDOW`` sitting days before past prorogations, separately for bills
  still in their first House and bills in the second House or ping-pong. A bill whose
  carry-over motion is already agreed is carried over with certainty.
- **Dissolution E:** an announced dissolution, else a per-sitting-day hazard by years since
  the last general election (buckets <3, 3-4, 4-4.5, 4.5+), with a hard cap 5 years after
  the Parliament first met. Five dissolutions in the data give few events per bucket, so
  each bucket rate is smoothed (+0.5 events, +1 day), which is the bucket-only logistic
  hazard with a weak prior.
- **Carry-over:** a bill with a carry-over motion in the current session survives the
  session end S (a new S is drawn for the next session); a dissolution ends it anyway.
- **Defeat:** Commons votes have no data yet (BUILD_PLAN task 7 skipped), so MODEL_SPEC
  §4.3's fallback applies: constant P(defeat) = 0.002 at each Commons vote stage.
- **Withdrawal:** constant hazard per bill sitting day = withdrawals / bill sitting days at
  risk, with no covariates (only a handful of cases).

All data is read through the DATA_CONTRACT §5 interface as of a date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from westminster_weather.data import calendar
from westminster_weather.domain.stages import BillStage
from westminster_weather.model import features, states
from westminster_weather.model.states import DataSource

logger = logging.getLogger(__name__)

YEARS_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 3.0, "<3"),
    (3.0, 4.0, "3-4"),
    (4.0, 4.5, "4-4.5"),
    (4.5, np.inf, "4.5+"),
)
MAX_PARLIAMENT_YEARS = 5
DEFEAT_PROBABILITY_FALLBACK = 0.002
VOTE_STATES: frozenset[str] = frozenset(
    {
        BillStage.COMMONS_SECOND_READING.value,
        BillStage.COMMONS_REPORT_STAGE.value,
        BillStage.COMMONS_THIRD_READING.value,
    }
)
# A session that has already outlasted every past session is assumed to end within this
# many more sitting days (uniformly).
OVERRUN_SITTING_DAYS = 20
# Bills in progress this many sitting days before a prorogation are the cases for the
# session-end outcome probabilities.
SESSION_END_WINDOW = 30
FIRST_HOUSE, SECOND_HOUSE = "first_house", "second_house_or_ping_pong"
SESSION_END_OUTCOMES = ("died", "carried_over", "rushed_through")


def years_bucket(years: float) -> str:
    for lo, hi, label in YEARS_BUCKETS:
        if lo <= years < hi:
            return label
    return YEARS_BUCKETS[-1][2]


def _either_house_dates(src: Any, as_of: date) -> np.ndarray:
    return calendar.sitting_dates(src.get_calendar(as_of)["sitting_days"])


# --------------------------------------------------------------------------- session end


@dataclass
class SessionLengths:
    """Lengths in sitting days of past sessions that ended by prorogation."""

    lengths: np.ndarray
    session_ids: list[int]

    def sample(self, sitting_days_so_far: int, rng: np.random.Generator, n: int) -> np.ndarray:
        """Total session lengths for ``n`` paths, given the session has lasted
        ``sitting_days_so_far`` (MODEL_SPEC §4.2: sample from lengths ≥ current length)."""
        longer = self.lengths[self.lengths >= sitting_days_so_far]
        if len(longer) == 0:
            return sitting_days_so_far + rng.integers(1, OVERRUN_SITTING_DAYS + 1, size=n)
        return rng.choice(longer, size=n)

    def describe(self) -> str:
        return (
            f"{len(self.lengths)} past sessions ended by prorogation; an average session lasts "
            f"{self.lengths.mean():.0f} sitting days (median {np.median(self.lengths):.0f}, "
            f"range {self.lengths.min()}-{self.lengths.max()})"
        )


def fit_session_lengths(as_of: date, data: DataSource | None = None) -> SessionLengths:
    src = states._source(data)
    cal = src.get_calendar(as_of)
    dates = calendar.sitting_dates(cal["sitting_days"])
    sess = cal["sessions"]
    done = sess[
        (sess["end_type"] == "prorogation")
        & sess["end_date"].notna()
        & (sess["session_id"] >= states.MIN_SESSION_ID)
    ]
    lengths = np.array(
        [
            calendar.count_sitting_days(dates, start - timedelta(days=1), end)
            for start, end in zip(done["start_date"], done["end_date"], strict=True)
        ]
    )
    return SessionLengths(lengths, [int(s) for s in done["session_id"]])


# --------------------------------------------------------------------------- prorogation


def house_group(state: str, originating_house: str) -> str:
    """Whether a bill in ``state`` is still in its first House."""
    first = states.chain(originating_house)[: 5 if originating_house != "Lords" else 3]
    return FIRST_HOUSE if state in {s.value for s in first} else SECOND_HOUSE


@dataclass
class SessionEndOutcomes:
    """P(died / carried over / rushed through) for a bill still in progress when a session
    ends by prorogation, by ``house_group``."""

    probabilities: dict[str, dict[str, float]]
    counts: dict[str, dict[str, int]]

    def describe(self) -> str:
        parts = []
        for group, p in self.probabilities.items():
            n = sum(self.counts[group].values())
            parts.append(
                f"{group.replace('_', ' ')} ({n} cases): dies {100 * p['died']:.0f}%, "
                f"carried over {100 * p['carried_over']:.0f}%, "
                f"rushed through {100 * p['rushed_through']:.0f}%"
            )
        return "; ".join(parts)


def session_end_cases(as_of: date, data: DataSource | None = None) -> pd.DataFrame:
    """Every (prorogation, government bill) where the bill was in progress at the start of
    the last ``SESSION_END_WINDOW`` sitting days before a prorogation known on ``as_of``,
    with its stage group then and what happened by the prorogation."""
    src = states._source(data)
    cal = src.get_calendar(as_of)
    dates = calendar.sitting_dates(cal["sitting_days"])
    sess = cal["sessions"]
    sess = sess[
        (sess["end_type"] == "prorogation")
        & sess["end_date"].notna()
        & (sess["session_id"] >= states.MIN_SESSION_ID)
    ]
    bills = src.get_bills(as_of)
    bills = bills[bills["introduced_session_id"] >= states.MIN_SESSION_ID].set_index("bill_id")
    trans = states.build_transitions(as_of, src)
    ra = trans[trans["model_state"] == "royal_assent"].set_index("bill_id")["entered_date"]

    rows = []
    for s in sess.itertuples():
        i_end = np.searchsorted(dates, np.datetime64(s.end_date, "D"), side="right") - 1
        window_start = dates[max(i_end - SESSION_END_WINDOW + 1, 0)].astype(date)
        for bill_id, b in bills.iterrows():
            if b["introduced_date"] > s.end_date:
                continue
            assent = ra.get(bill_id)
            if assent is not None and assent < window_start:
                continue
            failed_on = b["outcome_date"] if b["outcome"] != "royal_assent" else None
            if failed_on is not None and failed_on < window_start:
                continue
            t = trans[(trans["bill_id"] == bill_id) & (trans["entered_date"] <= window_start)]
            stage = (
                t.iloc[-1]["model_state"]
                if len(t)
                else trans.loc[trans["bill_id"] == bill_id, "model_state"].iloc[0]
            )
            if assent is not None and assent <= s.end_date:
                fate = "rushed_through"  # passed before the session ended
            elif b["outcome"] == "withdrawn" and b["outcome_date"] <= s.end_date:
                fate = "rushed_through"  # neither died at the session end nor carried over
            elif b["outcome"] == "fell_session_end" and b["outcome_date"] == s.end_date:
                fate = "died"
            else:
                fate = "carried_over"
            rows.append(
                {
                    "session_id": s.session_id,
                    "bill_id": bill_id,
                    "stage": stage,
                    "group": house_group(stage, b["originating_house"]),
                    "fate": fate,
                }
            )
    return pd.DataFrame(rows, columns=["session_id", "bill_id", "stage", "group", "fate"])


def fit_session_end_outcomes(as_of: date, data: DataSource | None = None) -> SessionEndOutcomes:
    cases = session_end_cases(as_of, data)
    probabilities, counts = {}, {}
    for group in (FIRST_HOUSE, SECOND_HOUSE):
        fates = cases.loc[cases["group"] == group, "fate"]
        n = max(len(fates), 1)
        counts[group] = {o: int((fates == o).sum()) for o in SESSION_END_OUTCOMES}
        probabilities[group] = {o: counts[group][o] / n for o in SESSION_END_OUTCOMES}
    return SessionEndOutcomes(probabilities, counts)


# --------------------------------------------------------------------------- dissolution


@dataclass
class DissolutionHazard:
    """P(Parliament is dissolved on a sitting day), by years since the last election."""

    rates: dict[str, float]
    events: dict[str, int]
    days: dict[str, int]

    def daily_probability(self, years_since_election: float) -> float:
        return self.rates[years_bucket(years_since_election)]

    def describe(self) -> str:
        parts = []
        for _, _, b in YEARS_BUCKETS:
            per_100 = 100 * self.rates[b]
            parts.append(
                f"{b} years: {per_100:.2f}% per sitting day "
                f"({self.events[b]} dissolutions in {self.days[b]} sitting days)"
            )
        return "; ".join(parts)


def _election_before(parls: pd.DataFrame) -> dict[int, date]:
    """Election that created each Parliament (the one that ended the previous Parliament)."""
    ordered = parls.sort_values("first_meeting_date")
    out = {}
    for prev, cur in zip(ordered.itertuples(), ordered.iloc[1:].itertuples(), strict=False):
        if prev.election_date is not None and not pd.isna(prev.election_date):
            out[cur.parliament_id] = prev.election_date
    return out


def fit_dissolution(as_of: date, data: DataSource | None = None) -> DissolutionHazard:
    """One observation per sitting day of each Parliament (from the calendar's first day);
    the last sitting day before a dissolution is the event."""
    src = states._source(data)
    cal = src.get_calendar(as_of)
    dates = calendar.sitting_dates(cal["sitting_days"])
    parls = cal["parliaments"]
    elections = _election_before(parls)
    events = {b: 0 for *_, b in YEARS_BUCKETS}
    days = {b: 0 for *_, b in YEARS_BUCKETS}
    for p in parls.itertuples():
        if p.parliament_id not in elections:
            continue
        start = np.datetime64(p.first_meeting_date, "D")
        dissolved = p.dissolution_date is not None and not pd.isna(p.dissolution_date)
        end = np.datetime64(p.dissolution_date if dissolved else as_of, "D")
        span = dates[(dates >= start) & (dates <= end)]
        election = elections[p.parliament_id]
        for i, d in enumerate(span):
            years = (d.astype(date) - election).days / 365.25
            b = years_bucket(years)
            days[b] += 1
            events[b] += int(dissolved and i == len(span) - 1)
    rates = {b: (events[b] + 0.5) / (days[b] + 1) for b in days}
    return DissolutionHazard(rates, events, days)


# --------------------------------------------------------------------------- withdrawal


@dataclass
class WithdrawalHazard:
    rate: float
    withdrawals: int
    days_at_risk: int

    def describe(self) -> str:
        per_year = self.rate * 150  # about 150 sitting days a year
        return (
            f"{self.withdrawals} withdrawals in {self.days_at_risk:,} bill sitting days: "
            f"{100 * self.rate:.4f}% per sitting day (about {100 * per_year:.2f}% per year)"
        )


def fit_withdrawal(
    as_of: date, data: DataSource | None = None, sessions: tuple[int, ...] | None = None
) -> WithdrawalHazard:
    src = states._source(data)
    rows = features.build_training_rows(as_of, src).rows
    trans = states.build_transitions(as_of, src)
    if sessions is not None:
        rows = rows[rows["introduced_session_id"].isin(sessions)]
        trans = trans[trans["introduced_session_id"].isin(sessions)]
    withdrawals = int((trans["fail_cause"] == "withdrawn").sum())
    return WithdrawalHazard(withdrawals / len(rows), withdrawals, len(rows))


# --------------------------------------------------------------------------- bundle


@dataclass
class FailureModel:
    session_lengths: SessionLengths
    dissolution: DissolutionHazard
    withdrawal: WithdrawalHazard
    session_end: SessionEndOutcomes | None = None
    defeat_probability: float = DEFEAT_PROBABILITY_FALLBACK
    fitted_as_of: date | None = None
    notes: list[str] = field(default_factory=list)

    def p_defeat(self, state: str) -> float:
        """P(defeat) when a bill leaves ``state`` (Commons vote stages only)."""
        return self.defeat_probability if state in VOTE_STATES else 0.0

    def describe(self) -> list[str]:
        return [
            f"Session end: {self.session_lengths.describe()}.",
            f"At a prorogation, an unfinished bill: {self.session_end.describe()}.",
            f"Dissolution: {self.dissolution.describe()}. "
            f"Hard cap {MAX_PARLIAMENT_YEARS} years after a Parliament first meets.",
            f"Withdrawal: {self.withdrawal.describe()}.",
            f"Defeat: {100 * self.defeat_probability:.1f}% at each Commons vote stage "
            "(2nd reading, report, 3rd reading); fixed fallback, no vote data yet.",
            *self.notes,
        ]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> FailureModel:
        return joblib.load(path)


def fit_failure(
    as_of: date, data: DataSource | None = None, sessions: tuple[int, ...] | None = None
) -> FailureModel:
    """Fit every failure component on data available on ``as_of``."""
    logger.warning(
        "No Commons division data (task 7 skipped): P(defeat) fixed at %s per vote stage",
        DEFEAT_PROBABILITY_FALLBACK,
    )
    return FailureModel(
        session_lengths=fit_session_lengths(as_of, data),
        dissolution=fit_dissolution(as_of, data),
        withdrawal=fit_withdrawal(as_of, data, sessions),
        session_end=fit_session_end_outcomes(as_of, data),
        fitted_as_of=as_of,
        notes=[
            "Carry-over: a bill with a carry-over motion already agreed in the current session "
            "is carried over at the session end; an election still ends it."
        ],
    )


def carried_over_in_session(carry_over_date: date | None, session_start: date) -> bool:
    """Whether the carry-over rule protects a bill at the end of the session that began
    ``session_start`` (its motion must have been agreed in that session)."""
    return (
        carry_over_date is not None
        and not pd.isna(carry_over_date)
        and (carry_over_date >= session_start)
    )


def plot_components(model: FailureModel, pp: Any, path: Path) -> Path:
    """Session lengths, dissolution risk by Parliament age and ping-pong length."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    lengths = np.sort(model.session_lengths.lengths)
    ax1.bar(range(len(lengths)), lengths, color="#4c72b0")
    ax1.axhline(lengths.mean(), color="black", ls="--", lw=1, label=f"average {lengths.mean():.0f}")
    ax1.set_xticks([])
    ax1.set_ylabel("Sitting days")
    ax1.set_title(f"Length of {len(lengths)} past sessions ended by prorogation")
    ax1.legend()

    buckets = [b for *_, b in YEARS_BUCKETS]
    rates = [100 * model.dissolution.rates[b] for b in buckets]
    ax2.bar(buckets, rates, color="#c44e52")
    for i, b in enumerate(buckets):
        ax2.text(i, rates[i], f"{model.dissolution.events[b]} of {model.dissolution.days[b]} days",
                 ha="center", va="bottom", fontsize=8)  # fmt: skip
    ax2.set_xlabel("Years since the last general election")
    ax2.set_ylabel("% chance per sitting day")
    ax2.set_title("Risk Parliament is dissolved")

    ax3.hist(pp.durations, bins=range(0, int(pp.durations.max()) + 2), color="#55a868")
    ax3.set_xlabel("Sitting days from ping-pong to Royal Assent")
    ax3.set_ylabel("Bills")
    ax3.set_title(f"Ping-pong length ({100 * pp.probability:.0f}% of Acts have ping-pong)")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
