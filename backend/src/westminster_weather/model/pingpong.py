"""Ping-pong (BUILD_PLAN task 8, MODEL_SPEC §4.4).

- **Whether ping-pong happens:** π_pp = share of completed bills (Acts) that went through
  ping-pong. MODEL_SPEC's covariates (Lords defeats, amendments agreed, size in clauses)
  are not available yet, so only the intercept is fitted; the share by originating House is
  reported for information.
- **Rounds:** a round is one sitting of stage 12, 13 or 42 (``model_state = ping_pong`` in
  ``get_stage_sittings``). ``N − 1 ~ Poisson(λ)`` with ``λ = exp(θ_0)``; with no Lords-defeat
  data the maximum-likelihood ``λ`` is the mean number of extra rounds.
- **Timing:** gaps between consecutive rounds and the gap from the last round to Royal
  Assent are drawn from their empirical distributions (sitting days, either House).
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
    extra_rounds_rate: float  # λ: mean rounds after the first
    round_gaps: np.ndarray  # sitting days between consecutive rounds
    final_gaps: np.ndarray  # sitting days from the last round to Royal Assent
    rounds: np.ndarray  # observed rounds per bill
    durations: np.ndarray  # observed sitting days from first round to Royal Assent

    def sample_rounds(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return 1 + rng.poisson(self.extra_rounds_rate, size=n)

    def sample_duration(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sitting days from entering ping-pong to Royal Assent for ``n`` paths: the gaps
        between ``N`` rounds plus the final gap."""
        rounds = self.sample_rounds(rng, n)
        extra = rounds - 1
        gaps = rng.choice(self.round_gaps, size=int(extra.sum())) if extra.sum() else []
        owner = np.repeat(np.arange(n), extra)
        between = np.bincount(owner, weights=gaps, minlength=n) if len(owner) else np.zeros(n)
        return between.astype(int) + rng.choice(self.final_gaps, size=n)

    def describe(self) -> list[str]:
        by_house = ", ".join(f"{h} {100 * p:.0f}%" for h, p in self.probability_by_house.items())
        d = self.durations
        return [
            f"Ping-pong happens for {100 * self.probability:.0f}% of bills that pass "
            f"({self.n_acts} Acts; by originating House: {by_house}).",
            f"Rounds: 1 + Poisson({self.extra_rounds_rate:.2f}); observed median "
            f"{np.median(self.rounds):.0f}, max {self.rounds.max()} rounds.",
            f"Gap between rounds: median {np.median(self.round_gaps):.0f} sitting days; "
            f"last round to Royal Assent: median {np.median(self.final_gaps):.0f} sitting days.",
            f"In total, ping-pong to Royal Assent takes a median of {np.median(d):.0f} sitting "
            f"days (80% between {np.percentile(d, 10):.0f} and {np.percentile(d, 90):.0f}; "
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
    entered = trans[trans["model_state"] == PP][["bill_id", "entered_date"]]
    acts = acts.merge(entered, on="bill_id", how="left")
    acts["had_pp"] = acts["entered_date"].notna()

    dates = calendar.sitting_dates(src.get_calendar(as_of)["sitting_days"])
    sittings = src.get_stage_sittings(as_of)
    sittings = sittings[sittings["model_state"] == PP]

    rounds, round_gaps, final_gaps, durations = [], [], [], []
    for bill in acts[acts["had_pp"]].itertuples():
        days = sorted(
            sittings.loc[
                (sittings["bill_id"] == bill.bill_id) & (sittings["event_date"] <= bill.ra_date),
                "event_date",
            ]
        )
        if not days:
            continue
        rounds.append(len(days))
        round_gaps += [
            calendar.count_sitting_days(dates, a, b) for a, b in zip(days, days[1:], strict=False)
        ]
        final_gaps.append(calendar.count_sitting_days(dates, days[-1], bill.ra_date))
        durations.append(calendar.count_sitting_days(dates, days[0], bill.ra_date))

    rounds_arr = np.array(rounds)
    return PingPongModel(
        probability=float(acts["had_pp"].mean()),
        probability_by_house=acts.groupby("originating_house")["had_pp"].mean().to_dict(),
        n_acts=len(acts),
        extra_rounds_rate=float(rounds_arr.mean() - 1),
        round_gaps=np.array(round_gaps),
        final_gaps=np.array(final_gaps),
        rounds=rounds_arr,
        durations=np.array(durations),
    )
