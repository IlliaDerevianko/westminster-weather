import numpy as np
import pandas as pd

from westminster_weather.analysis.remaining_time_models import (
    CV_FOLDS,
    TEST_SESSIONS,
    TRAIN_SESSIONS,
    add_pace,
    baseline_b0,
    baseline_b1,
    cv_splits,
)


def test_train_and_test_sessions_do_not_overlap_and_test_is_later():
    assert not TRAIN_SESSIONS & TEST_SESSIONS
    assert max(TRAIN_SESSIONS) < min(TEST_SESSIONS)


def test_cv_folds_always_validate_on_later_sessions():
    for train, val in CV_FOLDS:
        assert max(train) < min(val)
        assert (train | val) <= TRAIN_SESSIONS
    df = pd.DataFrame({"session_id": [20, 25, 26, 28, 30]})
    (tr, va), *_ = cv_splits(df)
    assert list(df.loc[tr, "session_id"]) == [20, 25] and list(df.loc[va, "session_id"]) == [26]


def _rows():
    return pd.DataFrame(
        {
            "anchor": ["1st House: 1st reading"] * 3 + ["2nd House: 1st reading"] * 2,
            "anchor_type": ["stage"] * 5,
            "origin_lords": [0, 0, 1, 0, 0],
            "stage_index": [0, 0, 0, 5, 5],
            "elapsed_days": [0, 0, 0, 40, 60],
            "target_days": [100, 200, 300, 50, 70],
        }
    )


def test_b1_uses_median_at_same_point_and_origin():
    train = _rows()
    test = train.iloc[[0, 2, 3]]
    assert list(baseline_b1(train, test)) == [150, 300, 60]


def test_b0_subtracts_elapsed_and_never_goes_below_one_day():
    train = _rows()
    test = pd.DataFrame({"origin_lords": [0, 0], "elapsed_days": [50, 500]})
    assert list(baseline_b0(train, test)) == [100, 1]


def test_pace_compares_with_training_median_at_same_stage():
    train = _rows()
    test = pd.DataFrame({"stage_index": [5], "origin_lords": [0], "elapsed_days": [100]})
    pace = add_pace(train, test)["pace_vs_typical"].iloc[0]
    assert np.isclose(pace, 101 / 51)  # training median at stage 5 = 50 days
