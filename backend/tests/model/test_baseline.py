"""BUILD_PLAN task 5: baselines B0 and B1."""

from datetime import date

import pytest
from test_states import _payload, _source

from westminster_weather.model import baseline
from westminster_weather.model.splits import TEST_SESSIONS, TRAIN_CUTOFF, TRAIN_SESSIONS

ACT_STAGES = [
    (6, "Commons", "{y}-01-10"),
    (7, "Commons", "{y}-01-20"),
    (8, "Commons", "{y}-02-01"),
    (9, "Commons", "{y}-02-10"),
    (10, "Commons", "{y}-02-10"),
    (1, "Lords", "{y}-02-11"),
    (2, "Lords", "{y}-02-20"),
    (3, "Lords", "{y}-03-01"),
    (4, "Lords", "{y}-03-10"),
    (5, "Lords", "{y}-03-12"),
    (11, "Unassigned", "{y}-04-10"),
]


def _act(bill_id, year, session):
    stages = [(s, h, d.format(y=year)) for s, h, d in ACT_STAGES]
    return _payload(bill_id, stages, session=session, is_act=True)


@pytest.fixture(scope="module")
def source():
    return _source(
        [
            _act(1, 2016, 28),  # training Act: 1st reading 2016-01-10, RA 2016-04-10 (91 days)
            _act(2, 2020, 35),  # test Act
            _payload(3, [(6, "Commons", "2021-01-05")], session=36),  # current session
        ]
    )


def test_fit_uses_training_sessions_only(source):
    fitted = baseline.fit(source)
    assert fitted.n_train_acts == 1
    assert fitted.b0_days == {"Commons": 91}
    # one training bill per cell is below MIN_BILLS_PER_CELL, so B1 falls back to the state
    assert fitted.b1_days == {}
    assert fitted.b1_state_days["lords_committee"] == 40  # 2016-03-01 -> 2016-04-10


def test_predictions():
    fitted = baseline.Baselines(
        {"Commons": 100.0}, {("Commons", "lords_committee"): 30.0}, {"lords_committee": 50.0}, 1
    )
    assert fitted.predict_b0("Commons", date(2020, 1, 1), date(2020, 2, 1)) == date(2020, 4, 10)
    # B0 never forecasts a date before the forecast is made
    assert fitted.predict_b0("Commons", date(2020, 1, 1), date(2020, 6, 1)) == date(2020, 6, 1)
    assert fitted.predict_b1("Commons", "lords_committee", date(2020, 3, 1)) == date(2020, 3, 31)
    assert fitted.predict_b1("Lords", "lords_committee", date(2020, 3, 1)) == date(2020, 4, 20)


def test_forecast_points_cover_test_acts_only(source):
    points = baseline.forecast_points(date(2021, 6, 1), source)
    assert set(points["bill_id"]) == {2}
    assert "royal_assent" not in set(points["model_state"])
    assert (points["forecast_as_of"] <= points["ra_date"]).all()


def test_evaluate_reports_mae(source):
    _, metrics = baseline.evaluate(baseline.fit(source), date(2021, 6, 1), source)
    assert metrics["n_test_bills"] == 1
    assert metrics["B0"]["mae_days"] >= 0 and metrics["B1"]["mae_days"] >= 0


def test_split_is_time_ordered():
    assert max(TRAIN_SESSIONS) < min(TEST_SESSIONS)
    assert TRAIN_CUTOFF < date(2019, 10, 15)  # first test session (34) began 2019-10-15
