"""BUILD_PLAN task 6: advance hazard."""

import numpy as np
import pandas as pd
import pytest

from westminster_weather.model import hazard
from westminster_weather.model.splits import TRAIN_SESSIONS


def _rows():
    return pd.DataFrame(
        {
            "state": ["commons_1st_reading", "lords_committee", "lords_committee"],
            "tau_bucket": ["0", "6-20", "61+"],
            "days_to_session_end_bucket": ["41+", "0-5", "16-40"],
            "carry_over": [0.0, 1.0, 0.0],
            "n_stage_sittings_cum": [1.0, 10.0, 30.0],
            "y": [1, 0, 0],
        }
    )


def test_design_matrix_encodes_cells_and_end_buckets():
    rows = _rows()
    spec = hazard.DesignSpec(shared=["carry_over", "n_stage_sittings_cum"])
    spec.means["n_stage_sittings_cum"], spec.sds["n_stage_sittings_cum"] = 10.0, 10.0
    m = pd.DataFrame(spec.matrix(rows), columns=spec.columns)
    assert m.loc[0, "cell:commons_1st_reading|0"] == 1
    assert m.loc[1, "cell:lords_committee|6-20"] == 1
    assert m.iloc[:, : len(hazard.STATE_LEVELS) * len(hazard.TAU_LEVELS)].sum(axis=1).eq(1).all()
    assert list(m["end:0-5"]) == [0, 1, 0]
    assert "end:41+" not in m.columns  # reference bucket
    assert list(m["n_stage_sittings_cum"]) == [-0.9, 0.0, 2.0]  # standardised


@pytest.fixture(scope="module")
def fitted():
    built = hazard.training_rows()
    return built, hazard.fit_hazard(built)


def test_trains_on_training_sessions_only(fitted):
    built, model = fitted
    assert set(built.rows["introduced_session_id"]) <= set(TRAIN_SESSIONS)
    val = model.validation
    assert max(val["fit_sessions"]) < min(val["validation_sessions"])


def test_beats_simple_references_on_validation(fitted):
    val = fitted[1].validation
    assert val["log_loss_best"] < val["log_loss_state_and_time_only"]
    assert val["log_loss_best"] < val["log_loss_constant_rate"]


def test_end_of_session_rush_has_sensible_signs(fitted):
    coefs = fitted[1].coefficients()
    assert coefs["end:0-5"] > coefs["end:6-15"] > 0
    assert coefs["end:16-40"] > 0
    assert coefs["same_day_2r_3r"] > 0  # emergency legislation moves fast


def test_save_and_load_round_trip(fitted, tmp_path):
    built, model = fitted
    path = model.save(tmp_path / "hazard.joblib")
    loaded = hazard.HazardModel.load(path)
    sample = built.rows.head(50)
    np.testing.assert_allclose(loaded.predict(sample), model.predict(sample))
