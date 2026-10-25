"""BUILD_PLAN task 10: backtest points and metrics."""

from datetime import date

import pandas as pd
import pytest

from westminster_weather.model import baseline, evaluate
from westminster_weather.model.splits import TEST_SESSIONS


def _results():
    base = {
        "originating_house": "Commons",
        "introduced_date": date(2020, 1, 1),
        "start_state": "lords_committee",
        "session_end": date(2021, 4, 29),
    }
    rows = [
        # became law on 2020-06-01; forecast median 10 days early, window covers it
        dict(bill_id=1, as_of=date(2020, 3, 1), outcome="royal_assent",
             outcome_date=date(2020, 6, 1), ra=True, ra_before_session_end=True,
             passage_probability=0.9, p_ra_before_session_end=0.8,
             p10=date(2020, 5, 1), p50=date(2020, 5, 22), p90=date(2020, 7, 1)),
        # became law later than the window
        dict(bill_id=2, as_of=date(2020, 3, 1), outcome="royal_assent",
             outcome_date=date(2020, 9, 1), ra=True, ra_before_session_end=True,
             passage_probability=0.6, p_ra_before_session_end=0.5,
             p10=date(2020, 4, 1), p50=date(2020, 5, 1), p90=date(2020, 6, 1)),
        # failed
        dict(bill_id=3, as_of=date(2020, 3, 1), outcome="fell_dissolution",
             outcome_date=date(2020, 4, 1), ra=False, ra_before_session_end=False,
             passage_probability=0.2, p_ra_before_session_end=0.1,
             p10=None, p50=None, p90=None),
    ]  # fmt: skip
    return pd.DataFrame([{**base, **r} for r in rows])


def test_score_on_known_results():
    fitted = baseline.Baselines({"Commons": 100.0}, {}, {"lords_committee": 60.0}, 1)
    m = evaluate.score(_results(), fitted)
    assert m["n_forecasts"] == 3 and m["n_forecasts_for_acts"] == 2
    assert m["mae_days"]["model"] == pytest.approx((10 + 123) / 2)
    assert m["coverage_p10_p90"] == 0.5
    expected_brier = ((0.9 - 1) ** 2 + (0.6 - 1) ** 2 + (0.2 - 0) ** 2) / 3
    assert m["brier_passage_probability"] == pytest.approx(expected_brier, abs=1e-4)
    assert 0 <= m["concordance_index"] <= 1


def test_backtest_points_are_test_bills_before_their_outcome():
    points = evaluate.backtest_points(date(2026, 10, 24))
    assert set(points["introduced_session_id"]) <= set(TEST_SESSIONS)
    assert (points["as_of"] < points["outcome_date"]).all()
    assert not points.duplicated(["bill_id", "as_of"]).any()
    assert "in_progress" not in set(points["outcome"])
