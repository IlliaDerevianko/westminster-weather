"""Part 1 step 5: LightGBM quantile model averaged with the simulator."""

from datetime import date

import numpy as np
import pandas as pd

from westminster_weather.model import quantile


def test_quantile_model_predicts_sorted_days_and_averages():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {"bill_id": range(400), "as_of": date(2018, 1, 1), "stage": rng.integers(0, 8, 400)}
    )
    y = 300 - 30 * x["stage"] + rng.normal(0, 20, 400)
    model = quantile.fit(x, y.to_numpy())
    pred = model.predict_days(x.head(5))
    assert (np.diff(pred, axis=1) >= 0).all() and (pred >= 0).all()

    results = pd.DataFrame(
        {"bill_id": [0], "as_of": [date(2018, 1, 1)], "p10": [date(2018, 2, 1)],
         "p50": [date(2018, 3, 1)], "p90": [date(2018, 4, 1)]}
    )  # fmt: skip
    avg = quantile.apply(results, x, model, "average")
    lgbm_p50 = date(2018, 1, 1) + pd.Timedelta(days=round(model.predict_days(x.head(1))[0, 1]))
    expected = (
        pd.Timestamp(date(2018, 3, 1))
        + (pd.Timestamp(lgbm_p50) - pd.Timestamp(date(2018, 3, 1))) / 2
    )
    assert avg["p50"].iloc[0] == expected.date()
