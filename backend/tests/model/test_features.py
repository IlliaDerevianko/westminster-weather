"""BUILD_PLAN task 4: feature registry, hazard training rows and no data leakage."""

from datetime import date

import pandas as pd
import pytest
from test_states import LORDS_FIRST_ACT, _payload, _source

from westminster_weather.model import features

CUTOFF = date(2016, 12, 1)


@pytest.fixture(scope="module")
def synthetic():
    return _source(
        [
            _payload(1, LORDS_FIRST_ACT, is_act=True, house="Lords"),
            _payload(2, [(6, "Commons", "2016-06-02"), (7, "Commons", "2016-06-09")]),
            _payload(4, [(6, "Commons", "2016-06-03")], withdrawn="2016-06-20T00:00:00"),
            _payload(3, [(6, "Commons", "2016-11-01")], session=31),
        ]
    )


@pytest.fixture(scope="module")
def rows(synthetic):
    return features.build_training_rows(CUTOFF, synthetic).rows


def _state_rows(rows, bill_id, state):
    return rows[(rows["bill_id"] == bill_id) & (rows["state"] == state)].sort_values("date")


def test_rows_run_from_entry_to_advance(rows):
    r = _state_rows(rows, 1, "lords_1st_and_2nd_reading")
    assert r["date"].iloc[0] == pd.Timestamp("2016-06-01")
    assert r["date"].iloc[-1] == pd.Timestamp("2016-06-13")  # entered Lords committee
    assert list(r["tau"]) == list(range(len(r)))
    assert list(r["y"]) == [0] * (len(r) - 1) + [1]
    assert r["tau_bucket"].iloc[0] == "0"


def test_same_day_transition_is_one_row_with_tau_zero(rows):
    r = _state_rows(rows, 1, "commons_report_stage")  # report and 3rd reading on 2016-07-12
    assert len(r) == 1
    assert (r["tau"].iloc[0], r["y"].iloc[0]) == (0, 1)


def test_failed_and_censored_states_have_no_positive(rows):
    failed = _state_rows(rows, 2, "commons_2nd_reading")
    assert len(failed) > 1 and failed["y"].sum() == 0
    censored = _state_rows(rows, 3, "commons_1st_reading")
    assert censored["y"].sum() == 0
    assert censored["date"].max() <= pd.Timestamp(CUTOFF)


def test_no_rows_for_ping_pong_or_royal_assent(rows):
    assert not set(rows["state"]) & {"ping_pong", "royal_assent"}


def test_time_varying_features_use_only_earlier_days(rows):
    r = _state_rows(rows, 1, "lords_1st_and_2nd_reading").set_index("date")
    # the 1st reading sitting on 2016-06-01 is not visible on that day, only from the next day
    assert r.loc[pd.Timestamp("2016-06-01"), "n_stage_sittings_cum"] == 0
    assert r.loc[pd.Timestamp("2016-06-06"), "n_stage_sittings_cum"] == 1


def test_registry_matches_rows_on_synthetic_data(synthetic, rows):
    _assert_rows_match_registry(rows, synthetic, n=len(rows))


def _assert_rows_match_registry(rows, data, n, seed=0):
    sample = rows.sample(n=min(n, len(rows)), random_state=seed)
    for row in sample.itertuples(index=False):
        before = features.row_as_of(row.date)
        on_day = pd.Timestamp(row.date).date()
        for name in features.COVARIATE_FEATURES:
            expected = features.feature_value(name, row.bill_id, before, data)
            assert getattr(row, name) == (expected or 0.0), (name, row)
        assert row.originating_house_lords == features.feature_value(
            "originating_house_lords", row.bill_id, on_day, data
        )
        assert row.introduced_months_into_session == pytest.approx(
            features.feature_value("introduced_months_into_session", row.bill_id, on_day, data)
        )
        assert row.parliament_age_years == pytest.approx(
            features.feature_value("parliament_age_years", row.bill_id, on_day, data)
        )


# --- leakage checks on the real data in Neon Postgres ------------------------------------


@pytest.fixture(scope="module")
def real_rows():
    return features.build_training_rows(date(2019, 12, 31))


def test_real_rows_counts(real_rows):
    assert len(real_rows.rows) > 20_000  # 26,197 on the Oct 2026 data
    assert 0.05 < real_rows.positive_rate < 0.2
    assert real_rows.rows["date"].max() <= pd.Timestamp(date(2019, 12, 31))


def test_real_row_features_use_no_later_data(real_rows):
    """Every sampled row's features equal the registry value computed with information
    available before that row's day (days_to_session_end is the documented exception)."""
    _assert_rows_match_registry(real_rows.rows, None, n=12)


def test_real_rows_do_not_change_with_a_later_cutoff(real_rows):
    early = features.build_training_rows(date(2015, 1, 1)).rows
    late = real_rows.rows
    keys = ["bill_id", "state", "date"]
    cols = ["tau", "y", *features.COVARIATE_FEATURES, "parliament_age_years"]
    merged = early.merge(late[late["date"] <= pd.Timestamp(date(2015, 1, 1))], on=keys)
    assert len(merged) == len(early)
    for col in cols:
        assert (merged[f"{col}_x"] == merged[f"{col}_y"]).all(), col
