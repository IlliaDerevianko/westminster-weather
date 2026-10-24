"""BUILD_PLAN task 9: simulator and forecast API."""

import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from westminster_weather.domain.stages import StopReason
from westminster_weather.model import failure, forecast, pingpong, simulate, states
from westminster_weather.model.splits import TRAIN_CUTOFF

AS_OF = date(2020, 1, 10)
CHAIN = [s.value for s in states.chain("Commons")]
HISTORY = pd.DataFrame(
    {
        "date": [d.date() for d in pd.date_range("2014-01-01", AS_OF) if d.weekday() < 4],
        "house": "Commons",
    }
)


class FakeHazard:
    """P(advance) from a rule on the design row, so tests control the simulator."""

    departments: list[str] = []

    def __init__(self, rule):
        self.rule = rule

    def predict(self, frame):
        return np.array([self.rule(r) for r in frame.itertuples()], dtype=float)


DIES = {"died": 1.0, "carried_over": 0.0, "rushed_through": 0.0}
CARRIED = {"died": 0.0, "carried_over": 1.0, "rushed_through": 0.0}
RUSHED = {"died": 0.0, "carried_over": 0.0, "rushed_through": 1.0}


def _models(
    rule, session_length=100, pi_pp=0.0, withdrawal=0.0, defeat=0.0, dissolution=0.0,
    first=DIES, second=DIES,
):  # fmt: skip
    rates = {b: dissolution for *_, b in failure.YEARS_BUCKETS}
    session_end = failure.SessionEndOutcomes(
        {failure.FIRST_HOUSE: first, failure.SECOND_HOUSE: second}, {}
    )
    fail = failure.FailureModel(
        session_lengths=failure.SessionLengths(np.array([session_length]), [1]),
        dissolution=failure.DissolutionHazard(
            rates, dict.fromkeys(rates, 0), dict.fromkeys(rates, 1)
        ),
        withdrawal=failure.WithdrawalHazard(withdrawal, 0, 1),
        session_end=session_end,
        defeat_probability=defeat,
    )
    pp = pingpong.PingPongModel(
        probability=pi_pp, probability_by_house={}, n_acts=1, extra_rounds_rate=0.0,
        round_gaps=np.array([1]), final_gaps=np.array([2]), rounds=np.array([1]),
        durations=np.array([2]),
    )  # fmt: skip
    return simulate.Models(FakeHazard(rule), fail, pp)


def _start(carry=False, k0=0):
    return simulate.StartState(
        bill_id=1, as_of=AS_OF, chain=CHAIN, k0=k0, tau0=0, shared_features={},
        carry_over_active=carry, session_days_so_far=0, announced_session_end=None,
        announced_dissolution=None, election_date=date(2019, 12, 12),
        parliament_first_meeting=date(2019, 12, 17),
    )  # fmt: skip


def _run(models, start=None, n=2000):
    return simulate.simulate(start or _start(), models, HISTORY, n=n, seed=0)


def test_certain_advance_chains_through_every_state_on_day_one():
    result = _run(_models(lambda r: 1.0))
    assert (result.outcome == simulate.ROYAL_ASSENT).all()
    assert (result.end_day == 1).all()  # several states can be passed on the same day


def test_no_advance_fails_at_session_end():
    result = _run(_models(lambda r: 0.0, session_length=30))
    assert (result.outcome == simulate.SESSION_END).all()
    assert (result.end_day == 31).all()
    assert (result.fail_state == 0).all()  # failed in the state it was stuck in


def test_carry_over_survives_one_session_end():
    result = _run(_models(lambda r: 0.0, session_length=30), _start(carry=True))
    assert (result.outcome == simulate.SESSION_END).all()
    assert (result.end_day == 61).all()


def test_prorogation_can_carry_over_a_bill():
    result = _run(_models(lambda r: 0.0, session_length=30, first=CARRIED))
    assert result.carried.all()
    assert (result.outcome != simulate.SESSION_END).all()  # only the election ends it


def _late_rush(r):
    """Moves only after 61+ sitting days in a state and in the last 0-5 days of a session,
    so a bill can only finish if its session is extended."""
    return 1.0 if r.tau_bucket == "61+" and r.days_to_session_end_bucket == "0-5" else 0.0


def test_rushed_bills_finish_at_end_of_session_speed():
    result = _run(_models(_late_rush, session_length=3, first=RUSHED))
    assert result.rushed.all()
    assert (result.outcome == simulate.ROYAL_ASSENT).all()
    assert (result.end_day > 3).all()  # finished in the extended session


def test_session_end_outcome_depends_on_house():
    models = _models(_late_rush, session_length=3, first=DIES, second=RUSHED)
    in_lords = _run(models, _start(k0=CHAIN.index("lords_committee")))
    assert in_lords.rushed.all() and (in_lords.outcome == simulate.ROYAL_ASSENT).all()
    in_commons = _run(models, _start(k0=CHAIN.index("commons_committee")))
    assert (in_commons.outcome == simulate.SESSION_END).all()


def test_days_to_session_end_is_recomputed_from_each_paths_session_end():
    # Bills only move in the last 0-5 sitting days of the simulated session.
    rule = lambda r: 1.0 if r.days_to_session_end_bucket == "0-5" else 0.0  # noqa: E731
    result = _run(_models(rule, session_length=50))
    assert (result.outcome == simulate.ROYAL_ASSENT).all()
    assert (result.end_day >= 45).all() and (result.end_day <= 50).all()


def test_failure_routes_and_pingpong():
    result = _run(_models(lambda r: 1.0, pi_pp=1.0))
    assert (result.end_day == 3).all()  # ping-pong final gap of 2 sitting days
    result = _run(_models(lambda r: 1.0, defeat=1.0))
    assert (result.outcome == simulate.DEFEATED).all()
    assert (result.fail_state == CHAIN.index("commons_2nd_reading")).all()
    result = _run(_models(lambda r: 0.0, withdrawal=1.0))
    assert (result.outcome == simulate.WITHDRAWN).all()
    result = _run(_models(lambda r: 0.0, dissolution=1.0))
    assert (result.outcome == simulate.ELECTION).all()


def test_same_seed_same_paths():
    models = _models(lambda r: 0.05)
    a, b = _run(models), _run(models)
    np.testing.assert_array_equal(a.end_day, b.end_day)


def test_summary_probabilities_sum_to_one():
    result = _run(_models(lambda r: 0.03, session_length=60, withdrawal=0.002, defeat=0.05))
    summary = forecast.summarise(result, AS_OF)
    total = summary["passage_probability"] + sum(summary["primary_stop_risks"].values())
    assert total == pytest.approx(1.0)
    for stage in summary["stages"]:
        assert 0 <= stage.p_reach <= 1


# --- real data and fitted models ----------------------------------------------------------

needs_models = pytest.mark.skipif(
    not (forecast.MODELS_DIR / "hazard_train.joblib").exists(), reason="run `cli train` first"
)


@needs_models
def test_real_forecast_is_fast_consistent_and_leak_free():
    started = time.perf_counter()
    response = forecast.forecast_bill(3064, date(2021, 12, 1))
    assert time.perf_counter() - started < 5
    assert response.model_variant == "train"
    total = response.passage_probability + sum(response.primary_stop_risks.values())
    assert total == pytest.approx(1.0)
    assert response.time_to_royal_assent.p10_days <= response.time_to_royal_assent.p90_days
    assert response.stages[0].model_state.value == "commons_2nd_reading"
    again = forecast.forecast_bill(3064, date(2021, 12, 1))
    assert again.passage_probability == response.passage_probability


@needs_models
def test_resolved_bills_and_dates_before_training_cutoff():
    done = forecast.forecast_bill(3064, date(2022, 6, 1))
    assert done.passage_probability == 1.0
    assert done.time_to_royal_assent.estimated_median_date == date(2022, 3, 24)
    fell = forecast.forecast_bill(3462, date(2024, 9, 1))  # Renters (Reform) Bill, 2024 election
    assert fell.passage_probability == 0.0
    assert fell.primary_stop_risks[StopReason.ELECTION_CALLED] == 1.0
    with pytest.raises(forecast.ForecastUnavailable):
        forecast.forecast_bill(3064, TRAIN_CUTOFF.replace(year=2018))


def test_per_path_speed_spread_widens_timing():
    narrow = _models(lambda r: 0.05, session_length=10_000)
    wide = _models(lambda r: 0.05, session_length=10_000)
    wide.hazard.speed_sd = 1.0
    a, b = _run(narrow, n=4000), _run(wide, n=4000)
    spread = lambda r: np.subtract(*np.percentile(r.end_day, [90, 10]))  # noqa: E731
    assert spread(b) > spread(a)
