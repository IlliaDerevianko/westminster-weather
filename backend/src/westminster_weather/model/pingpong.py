"""Ping-pong (BUILD_PLAN task 8, MODEL_SPEC §4.4), in the form the data interface supports.

- **Whether ping-pong happens:** π_pp = share of completed bills (Acts) that went through
  ping-pong. MODEL_SPEC's covariates (Lords defeats, amendments agreed, size in clauses)
  are not available yet, so only the intercept is fitted; the share by originating House is
  reported for information.
- **Length:** MODEL_SPEC splits ping-pong into rounds, gaps between rounds and a final gap
  to Royal Assent. Counting rounds needs each ping-pong sitting, which the DATA_CONTRACT §5
  interface does not expose (``get_transitions`` gives only ping-pong entry and the Royal
  Assent date). Until it does, the simulator draws the **total** sitting days from entering
  ping-pong to Royal Assent from its empirical distribution. With no covariates this has the
  same distribution as rounds × gaps + final gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from westminster_weather.data import calendar
from westminster_weather.domain.stages import BillStage
from westminster_weather.model import states
from westminster_weather.model.states import DataSource

PP = BillStage.PING_PONG.value
RA = BillStage.ROYAL_ASSENT.value


@dataclass
class PingPongModel:
    probability: float
    probability_by_house: dict[str, float]
    n_acts: int
    durations: np.ndarray  # sitting days from entering ping-pong to Royal Assent

    def sample_duration(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(self.durations, size=n)

    def describe(self) -> list[str]:
        d = self.durations
        by_house = ", ".join(f"{h} {100 * p:.0f}%" for h, p in self.probability_by_house.items())
        return [
            f"Ping-pong happens for {100 * self.probability:.0f}% of bills that pass "
            f"({self.n_acts} Acts; by originating House: {by_house}).",
            f"Ping-pong to Royal Assent takes a median of {np.median(d):.0f} sitting days "
            f"(80% between {np.percentile(d, 10):.0f} and {np.percentile(d, 90):.0f}; "
            f"{len(d)} bills).",
        ]


def fit_pingpong(
    as_of: date, data: DataSource | None = None, sessions: tuple[int, ...] | None = None
) -> PingPongModel:
    """Fit on bills that reached Royal Assent by ``as_of`` (optionally only from
    ``sessions``)."""
    src = states._source(data)
    trans = states.build_transitions(as_of, src)
    if sessions is not None:
        trans = trans[trans["introduced_session_id"].isin(sessions)]
    acts = trans[trans["model_state"] == RA][["bill_id", "originating_house", "entered_date"]]
    acts = acts.rename(columns={"entered_date": "ra_date"})
    pp = trans[trans["model_state"] == PP][["bill_id", "entered_date"]]
    acts = acts.merge(pp, on="bill_id", how="left")
    acts["had_pp"] = acts["entered_date"].notna()

    dates = calendar.sitting_dates(src.get_calendar(as_of)["sitting_days"])
    with_pp = acts[acts["had_pp"]]
    durations = np.array(
        [
            calendar.count_sitting_days(dates, entered, ra)
            for entered, ra in zip(with_pp["entered_date"], with_pp["ra_date"], strict=True)
        ]
    )
    return PingPongModel(
        probability=float(acts["had_pp"].mean()),
        probability_by_house=acts.groupby("originating_house")["had_pp"].mean().to_dict(),
        n_acts=len(acts),
        durations=durations,
    )
