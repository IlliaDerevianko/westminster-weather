"""Direct quantile model for time to Royal Assent (Part 1, step 5).

LightGBM quantile regressions predict calendar days from ``as_of`` to Royal Assent
(10th, 50th and 90th percentiles) from the bill's position on ``as_of``: the simulator's start
state (stage, time in stage, Tier 1 features, pace, session days so far, carry-over) plus the
calendar month and days since introduction. Trained only on forecast points from training-era
sessions, for bills that reached Royal Assent; it complements the simulator's dates and does
not change pass probabilities.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "random_state": 0,
}


def start_features(start: Any, bill: pd.Series) -> dict[str, Any]:
    """Quantile-model inputs from a simulator start state."""
    as_of = start.as_of
    return {
        "bill_id": start.bill_id,
        "as_of": as_of,
        "stage": start.k0 + (10 if bill["originating_house"] == "Lords" else 0),
        "tau0": start.tau0,
        "session_days_so_far": start.session_days_so_far,
        "carry_over_active": float(start.carry_over_active),
        "month": as_of.month,
        "days_since_introduced": (as_of - bill["introduced_date"]).days,
        **start.shared_features,
    }


def _features_one(args: tuple[int, date, Any]) -> dict[str, Any] | None:
    from westminster_weather.model import forecast, states

    bill_id, as_of, hazard = args
    start, bill, _ = forecast.start_state(bill_id, as_of, states._source(None), hazard)
    return None if start is None else start_features(start, bill)


def point_features(points: pd.DataFrame, hazard: Any, workers: int = 6) -> pd.DataFrame:
    """Forecast-time features for every (bill, as_of) point, in parallel processes."""
    jobs = [(int(b), a, hazard) for b, a in zip(points["bill_id"], points["as_of"], strict=True)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(_features_one, jobs, chunksize=16) if r is not None]
    return pd.DataFrame(rows)


@dataclass
class QuantileModel:
    columns: list[str]
    models: dict[float, Any]

    def predict_days(self, X: pd.DataFrame) -> np.ndarray:
        """[n, 3] days to Royal Assent at the 10th, 50th and 90th percentiles (sorted)."""
        preds = np.column_stack([self.models[q].predict(X[self.columns]) for q in QUANTILES])
        return np.sort(np.maximum(preds, 0), axis=1)


def fit(features: pd.DataFrame, days_to_ra: pd.Series) -> QuantileModel:
    import lightgbm as lgb

    columns = [c for c in features.columns if c not in ("bill_id", "as_of")]
    models = {}
    for q in QUANTILES:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGBM_PARAMS)
        model.fit(features[columns], days_to_ra)
        models[q] = model
    return QuantileModel(columns, models)


def apply(
    results: pd.DataFrame, features: pd.DataFrame, model: QuantileModel, how: str
) -> pd.DataFrame:
    """Replace the simulator's p10/p50/p90 dates by the quantile model's (``how='lgbm'``) or by
    the average of the two (``how='average'``)."""
    out = results.merge(features[["bill_id", "as_of"]], on=["bill_id", "as_of"], how="left")
    merged = out.merge(features, on=["bill_id", "as_of"], how="left")
    pred = model.predict_days(merged)
    as_of = pd.to_datetime(out["as_of"])
    for i, col in enumerate(("p10", "p50", "p90")):
        lgbm_date = as_of + pd.to_timedelta(np.round(pred[:, i]), unit="D")
        if how == "lgbm":
            new = lgbm_date
        else:
            sim = pd.to_datetime(out[col])
            new = sim + (lgbm_date - sim) / 2
            new = new.where(sim.notna(), lgbm_date)
        out[col] = new.dt.date
    return out
