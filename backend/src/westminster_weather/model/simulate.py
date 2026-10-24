"""Monte Carlo simulator (BUILD_PLAN task 9, MODEL_SPEC §5).

Simulates ``n`` paths of one bill from its state on ``as_of``, all paths together, one future
sitting day at a time. Each day, in this order (MODEL_SPEC §5):

1. Dissolution: past the path's dissolution day E → fails with ``ELECTION_CALLED``.
2. Session end (prorogation): past the path's session end S, an unfinished bill **dies**
   (``SESSION_RUNS_OUT``), is **carried over** (keeps its state and time in state; a new
   session end is drawn for the next session, with no sitting days counted for the gap) or is
   **rushed through** (the session is extended until it finishes, at end-of-session speed:
   its days-to-session-end bucket stays at 0-5). Probabilities depend on whether the bill is
   still in its first House (``failure.SessionEndOutcomes``). A carry-over motion already
   agreed on ``as_of`` means carried over with certainty.
3. Withdrawal: constant daily hazard.
4. Ping-pong: a path in ping-pong reaches Royal Assent on its drawn day.
5. Advance: the logistic hazard gives P(move on today) from the state, time in state and
   sitting days left in the path's session; each path's odds are multiplied by its own
   speed factor exp(u), u ~ Normal(0, ``speed_sd``), drawn once per path. A bill can move
   through several states on the same day (each new state gets a τ = 0 chance, as in the
   training rows). Leaving a Commons
   vote stage risks defeat. Leaving the second House's last state leads to ping-pong with
   probability π_pp, otherwise straight to Royal Assent.

``days_to_session_end`` is recomputed every day from each path's own simulated session end
(the TASK 9 REQUIREMENT in ``features.py``); the training-only "41+" fallback is never used.
Time-varying features other than τ and days to session end are frozen at ``as_of``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from westminster_weather.domain.stages import BillStage, StopReason
from westminster_weather.model import failure, features, future_calendar
from westminster_weather.model.failure import MAX_PARLIAMENT_YEARS, VOTE_STATES, FailureModel
from westminster_weather.model.hazard import HazardModel
from westminster_weather.model.pingpong import PingPongModel

N_PATHS = 10_000
PP = BillStage.PING_PONG.value
RA = BillStage.ROYAL_ASSENT.value

# Path outcome codes.
RUNNING, ROYAL_ASSENT = -1, 0
OUTCOME_REASON: dict[int, StopReason] = {
    1: StopReason.SESSION_RUNS_OUT,
    2: StopReason.ELECTION_CALLED,
    3: StopReason.PAUSED_OR_WITHDRAWN,
    4: StopReason.DEFEATED,
}
SESSION_END, ELECTION, WITHDRAWN, DEFEATED = 1, 2, 3, 4

TAU_EDGES = np.array([1, 6, 21, 61])  # τ buckets 0 | 1-5 | 6-20 | 21-60 | 61+
END_EDGES = np.array([6, 16, 41])  # days-to-end buckets 0-5 | 6-15 | 16-40 | 41+


@dataclass
class StartState:
    """Where a bill stands on ``as_of`` and the context the simulation needs."""

    bill_id: int
    as_of: date
    chain: list[str]
    k0: int  # index of the current state in ``chain``
    tau0: int  # sitting days already spent in the current state
    shared_features: dict[str, float]
    carry_over_active: bool  # carry-over motion agreed in the current session
    session_days_so_far: int
    announced_session_end: date | None
    announced_dissolution: date | None
    election_date: date  # election that created the current Parliament
    parliament_first_meeting: date
    parliaments: pd.DataFrame | None = None  # as known on as_of (for dissolution gaps)
    # Live forecasts only: (chain index, date) of the first later stage that already has a
    # future-dated sitting scheduled. Never set in backtests (announcement dates unknown).
    scheduled_entry: tuple[int, date] | None = None


@dataclass
class Models:
    hazard: HazardModel
    failure: FailureModel
    pingpong: PingPongModel
    # Optional LightGBM quantile model; its days to Royal Assent are averaged with the
    # simulator's (Part 1 step 5).
    quantile: Any = None


@dataclass
class SimResult:
    start: StartState
    days: np.ndarray  # future sitting dates, days[j - 1] is day j
    outcome: np.ndarray  # per path: RUNNING, ROYAL_ASSENT or a failure code
    end_day: np.ndarray  # day of Royal Assent or failure
    fail_state: np.ndarray  # chain index of the state the path failed in (-1 if none)
    entry_day: np.ndarray  # [path, chain index] day the state was entered (-1 if never)
    first_session_end: np.ndarray  # S before any carry-over
    carried: np.ndarray  # path was carried over into a later session
    rushed: np.ndarray  # path was rushed through at a session end
    meta: dict[str, Any] = field(default_factory=dict)

    def date_of(self, day: np.ndarray) -> np.ndarray:
        return self.days[np.asarray(day) - 1]


def probability_table(
    hazard: HazardModel, chain: list[str], shared: dict[str, float]
) -> np.ndarray:
    """P(advance) for every (chain state, τ bucket, days-to-end bucket); NaN for states the
    hazard does not cover (ping-pong, Royal Assent)."""
    tau_labels = [label for *_, label in features.TAU_BUCKETS]
    end_labels = [label for *_, label in features.END_BUCKETS]
    rows = []
    for k, state in enumerate(chain):
        if state in (PP, RA):
            continue
        for ti, tau in enumerate(tau_labels):
            for ei, end in enumerate(end_labels):
                rows.append(
                    {
                        "k": k, "ti": ti, "ei": ei, "state": state, "tau_bucket": tau,
                        "days_to_session_end_bucket": end, **shared,
                    }
                )  # fmt: skip
    frame = pd.DataFrame(rows)
    table = np.full((len(chain), len(tau_labels), len(end_labels)), np.nan)
    table[frame["k"], frame["ti"], frame["ei"]] = hazard.predict(frame)
    return table


def _future_days(
    start: StartState, sitting_days: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Sitting days from ``as_of`` to the 5-year limit of this Parliament, at the density of
    the training clock (``future_calendar``)."""
    cap = start.parliament_first_meeting + timedelta(days=round(365.25 * MAX_PARLIAMENT_YEARS))
    parls = (
        start.parliaments
        if start.parliaments is not None
        else pd.DataFrame(
            columns=["parliament_id", "first_meeting_date", "dissolution_date", "election_date"]
        )
    )
    days = future_calendar.sample_future_days(sitting_days, start.as_of, cap, rng, parls)
    return np.array(days, dtype="datetime64[D]")


def _day_index(days: np.ndarray, d: date) -> int:
    """Last future day on or before ``d`` (0 = before the first future day)."""
    return int(np.searchsorted(days, np.datetime64(d, "D"), side="right"))


def _dissolution_days(
    start: StartState, days: np.ndarray, failure: FailureModel, rng: np.random.Generator, n: int
) -> np.ndarray:
    """Day E after which each path's Parliament no longer exists."""
    if start.announced_dissolution is not None:
        return np.full(n, _day_index(days, start.announced_dissolution))
    years = (days - np.datetime64(start.election_date, "D")).astype(int) / 365.25
    p = np.array([failure.dissolution.daily_probability(y) for y in years])
    survive = np.cumprod(1 - p)
    u = rng.random(n)
    event = np.searchsorted(-survive, -u, side="left") + 1  # first day survival drops below u
    return np.minimum(event, len(days))  # the 5-year cap is the last generated day


def _session_end_days(
    start: StartState, days: np.ndarray, failure: FailureModel, rng: np.random.Generator, n: int
) -> np.ndarray:
    """Day S (the session's last sitting day) for each path."""
    if start.announced_session_end is not None:
        return np.full(n, _day_index(days, start.announced_session_end))
    total = failure.session_lengths.sample(start.session_days_so_far, rng, n)
    return total - start.session_days_so_far


def simulate(
    start: StartState,
    models: Models,
    sitting_days: pd.DataFrame,
    n: int = N_PATHS,
    seed: int | None = None,
) -> SimResult:
    """Run ``n`` paths. ``sitting_days`` is the calendar known on ``as_of``. The random seed
    defaults to one fixed per (bill, ``as_of``), so forecasts are reproducible."""
    rng = np.random.default_rng(
        seed if seed is not None else [start.bill_id, start.as_of.toordinal()]
    )
    days = _future_days(start, sitting_days, rng)
    forced_k, forced_day = -1, -1
    if start.scheduled_entry is not None:
        forced_k, when = start.scheduled_entry
        when64 = np.datetime64(when, "D")
        if when64 not in days:
            days = np.sort(np.append(days, when64))
        forced_day = int(np.searchsorted(days, when64)) + 1
    horizon = len(days)
    chain = start.chain
    n_states = len(chain)
    k_pp, k_ra = chain.index(PP), chain.index(RA)
    k_last_second = k_pp - 1
    vote = np.array([s in VOTE_STATES for s in chain])
    table = probability_table(models.hazard, chain, start.shared_features)
    p_defeat = models.failure.defeat_probability
    h_w = models.failure.withdrawal.rate
    pi_pp = models.pingpong.probability
    # Per-path speed: daily advance odds × exp(u), u ~ Normal(0, speed_sd), fixed per path.
    speed_sd = getattr(models.hazard, "speed_sd", 0.0)
    speed = np.exp(rng.normal(0.0, speed_sd, n)) if speed_sd > 0 else np.ones(n)

    state = np.full(n, start.k0)
    tau = np.full(n, start.tau0)
    outcome = np.full(n, RUNNING)
    end_day = np.zeros(n, dtype=int)
    fail_state = np.full(n, -1)
    entry_day = np.full((n, n_states), -1)
    entry_day[:, start.k0] = 0
    ra_due = np.full(n, np.iinfo(np.int64).max)
    if start.k0 == k_pp:
        ra_due[:] = _remaining_pingpong(models.pingpong, start.tau0, rng, n)
    E = _dissolution_days(start, days, models.failure, rng, n)
    S = _session_end_days(start, days, models.failure, rng, n)
    first_S = S.copy()
    carry_available = np.full(n, start.carry_over_active)
    carried = np.zeros(n, dtype=bool)
    rushed = np.zeros(n, dtype=bool)
    first_house = np.zeros(n_states, dtype=bool)
    first_house[: 3 if start.chain[0].startswith("lords") else 5] = True  # first House states
    outcomes = models.failure.session_end
    q = {
        name: np.array(
            [
                outcomes.probabilities[failure.FIRST_HOUSE][name],
                outcomes.probabilities[failure.SECOND_HOUSE][name],
            ]
        )
        if outcomes is not None
        else np.array([1.0, 1.0]) * (name == "died")
        for name in failure.SESSION_END_OUTCOMES
    }
    never = np.iinfo(np.int64).max // 2

    def finish(mask: np.ndarray, code: int, day: int) -> None:
        outcome[mask] = code
        end_day[mask] = day
        if code != ROYAL_ASSENT:
            fail_state[mask] = state[mask]
        else:
            entry_day[mask, k_ra] = day

    for j in range(1, horizon + 1):
        alive = outcome == RUNNING
        if not alive.any():
            break
        finish(alive & (j > E), ELECTION, j)
        alive = outcome == RUNNING
        ended = alive & (j > S)
        if ended.any():
            group = np.where(first_house[state], 0, 1)
            u = rng.random(n)
            certain = ended & carry_available
            dies = ended & ~certain & (u < q["died"][group])
            carry = ended & ~dies & (certain | (u < q["died"][group] + q["carried_over"][group]))
            rush = ended & ~dies & ~carry
            if carry.any():
                new_session = models.failure.session_lengths.sample(0, rng, int(carry.sum()))
                S[carry] = j - 1 + new_session
                carry_available[carry] = False
                carried[carry] = True
            S[rush] = never
            rushed[rush] = True
            finish(dies, SESSION_END, j)
        alive = outcome == RUNNING
        finish(alive & (rng.random(n) < h_w), WITHDRAWN, j)
        alive = outcome == RUNNING
        finish(alive & (state == k_pp) & (j >= ra_due), ROYAL_ASSENT, j)

        alive = outcome == RUNNING
        tau[alive] += 1
        active = alive & (state < k_pp)
        end_bucket = np.digitize(np.maximum(S - j, 0), END_EDGES)
        end_bucket[rushed] = 0  # extended session: end-of-session speed until it finishes
        if forced_k >= 0:
            behind = alive & (state < forced_k)
            if j == forced_day and behind.any():
                # The scheduled stage happens today: skipped states are passed on the same day.
                rows = np.flatnonzero(behind)
                for k in range(start.k0 + 1, forced_k + 1):
                    unset = entry_day[rows, k] < 0
                    entry_day[rows[unset], k] = j
                state[behind] = forced_k
                tau[behind] = 0
        for _ in range(n_states):
            if not active.any():
                break
            p = np.zeros(n)
            idx = np.flatnonzero(active)
            p[idx] = table[state[idx], np.digitize(tau[idx], TAU_EDGES), end_bucket[idx]]
            if speed_sd > 0:
                odds = p[idx] / (1 - p[idx]) * speed[idx]
                p[idx] = odds / (1 + odds)
            if j < forced_day:
                p[state == forced_k - 1] = 0.0  # the scheduled stage cannot start earlier
            moves = active & (rng.random(n) < p)
            defeated = moves & vote[state] & (rng.random(n) < p_defeat)
            finish(defeated, DEFEATED, j)
            moves &= ~defeated

            to_end = moves & (state == k_last_second)
            to_pp = to_end & (rng.random(n) < pi_pp)
            finish(to_end & ~to_pp, ROYAL_ASSENT, j)
            if to_pp.any():
                state[to_pp] = k_pp
                entry_day[to_pp, k_pp] = j
                ra_due[to_pp] = j + models.pingpong.sample_duration(rng, int(to_pp.sum()))
                finish(to_pp & (ra_due <= j), ROYAL_ASSENT, j)

            step = moves & ~to_end
            state[step] += 1
            entry_day[np.flatnonzero(step), state[step]] = j
            tau[step] = 0
            active = step  # the new state gets its own same-day (τ = 0) chance

    # The last generated day is the Parliament's 5-year limit: anything still open dies then.
    unresolved = int((outcome == RUNNING).sum())
    finish(outcome == RUNNING, ELECTION, horizon)
    return SimResult(
        start, days, outcome, end_day, fail_state, entry_day, first_S, carried, rushed,
        meta={"horizon_days": horizon, "unresolved_at_cap": unresolved},
    )  # fmt: skip


def _remaining_pingpong(
    model: PingPongModel, elapsed: int, rng: np.random.Generator, n: int
) -> np.ndarray:
    """Days until Royal Assent for a bill already ``elapsed`` sitting days into ping-pong:
    draws of total ping-pong length conditional on being at least ``elapsed`` + 1."""
    draws = model.sample_duration(rng, 20 * n)
    longer = draws[draws > elapsed]
    if len(longer) == 0:
        return np.full(n, 1)
    return rng.choice(longer, size=n) - elapsed
