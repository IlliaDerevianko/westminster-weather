"""BUILD_PLAN task 8: failure routes and ping-pong."""

from datetime import date

import numpy as np
import pytest
from test_states import LORDS_FIRST_ACT, _payload, _source

from westminster_weather.model import failure, pingpong
from westminster_weather.model.splits import TRAIN_CUTOFF, TRAIN_SESSIONS


def test_session_length_draws_are_conditional_on_length_so_far():
    lengths = failure.SessionLengths(np.array([120, 150, 260]), [1, 2, 3])
    rng = np.random.default_rng(0)
    assert set(lengths.sample(140, rng, 200)) == {150, 260}
    overrun = lengths.sample(300, rng, 200)
    assert overrun.min() > 300 and overrun.max() <= 300 + failure.OVERRUN_SITTING_DAYS


def test_years_buckets():
    assert failure.years_bucket(1.0) == "<3"
    assert failure.years_bucket(3.5) == "3-4"
    assert failure.years_bucket(4.2) == "4-4.5"
    assert failure.years_bucket(4.9) == "4.5+"


def test_carry_over_and_defeat_rules():
    assert failure.carried_over_in_session(date(2026, 3, 1), date(2025, 7, 1))
    assert not failure.carried_over_in_session(date(2025, 3, 1), date(2025, 7, 1))
    assert not failure.carried_over_in_session(None, date(2025, 7, 1))
    model = failure.FailureModel(None, None, None)
    assert model.p_defeat("commons_2nd_reading") == failure.DEFEAT_PROBABILITY_FALLBACK
    assert model.p_defeat("lords_committee") == 0.0


def test_pingpong_on_synthetic_data():
    source = _source(
        [
            _payload(1, LORDS_FIRST_ACT, is_act=True, house="Lords"),  # ping-pong 07-14 -> 07-19
            _payload(3, [(6, "Commons", "2016-11-01")], session=31),
        ]
    )
    model = pingpong.fit_pingpong(date(2016, 12, 1), source)
    assert model.n_acts == 1 and model.probability == 1.0
    assert list(model.durations) == [1]  # 07-19 is the next sitting day after 07-14


# --- fitted on the real data in Neon Postgres ---------------------------------------------


@pytest.fixture(scope="module")
def trained():
    return failure.fit_failure(TRAIN_CUTOFF, sessions=TRAIN_SESSIONS)


def test_session_lengths_use_prorogued_sessions_before_cutoff(trained):
    assert set(trained.session_lengths.session_ids) <= set(TRAIN_SESSIONS)
    assert 27 not in trained.session_lengths.session_ids  # ended by dissolution (2015)
    assert trained.session_lengths.lengths.min() > 50


def test_dissolution_risk_rises_with_parliament_age(trained):
    rates = trained.dissolution.rates
    assert rates["4.5+"] > rates["4-4.5"] > rates["<3"]
    assert sum(trained.dissolution.events.values()) == 3  # 2010, 2015, 2017 by Oct 2019


def test_withdrawal_rate_is_small(trained):
    assert 0 < trained.withdrawal.rate < 0.001


def test_pingpong_share_is_about_half():
    model = pingpong.fit_pingpong(TRAIN_CUTOFF, sessions=TRAIN_SESSIONS)
    assert 0.3 < model.probability < 0.7
    assert (model.durations >= 0).all()
