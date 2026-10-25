"""Stage 3 forecast for the API: days from ``as_of`` until an Act's law applies (XGBoost).

Serves the two-step model evaluated in ``analysis.commencement_xgb`` (same features and
target): a classifier for "applies on Royal Assent day", a median regressor for the days left
otherwise, and p10 / p90 quantile regressors for the window. Reads the database only
(``act_commencements``, ``act_texts``, ``post_royal_assent_events`` and reference tables).

No future data: models are fitted per month on checkpoints before the month's first day whose
outcome was known by then, and each forecast uses only events dated on or before ``as_of``.
Bills without Royal Assent by ``as_of`` get no stage 3 forecast (the Act's text, the model's
main input, does not exist yet).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from xgboost import XGBClassifier, XGBRegressor

from westminster_weather.analysis import commencement_xgb as cx

log = logging.getLogger(__name__)
MIN_TRAIN_ACTS = 50


@dataclass
class Stage3Models:
    cut: pd.Timestamp
    cols: list[str]
    departments: list[str]
    same_day: XGBClassifier
    q: dict[float, XGBRegressor]  # 0.1, 0.5, 0.9 on log1p(days) for "not applied yet" rows


_DATA: tuple[pd.DataFrame, pd.DataFrame] | None = None


def _data() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(Acts with features, post-Royal-Assent events), loaded once per process. None, and
    retried on the next call, if a source table is missing or the database is unreachable
    (e.g. ``act_texts`` before `clients.legislation_texts` has run): stage 3 then has no forecast
    instead of failing the whole bill forecast."""
    global _DATA
    if _DATA is None:
        try:
            t = cx.load_tables()
        except SQLAlchemyError as exc:
            log.warning("stage 3 forecast unavailable: %s", str(exc).splitlines()[0])
            return None
        events = t["events"].assign(event_date=pd.to_datetime(t["events"]["event_date"]))
        _DATA = cx.act_features(t), events
    return _DATA


def _dept_cols(df: pd.DataFrame, departments: list[str]) -> pd.DataFrame:
    df = df.copy()
    for i, d in enumerate(departments):
        df[f"dept_{i}"] = (df["department"] == d).astype(float)
    return df


@cache
def _models(cut: pd.Timestamp) -> Stage3Models | None:
    data = _data()
    if data is None:
        return None
    acts, events = data
    rows = cx.build_rows(acts, events)
    known = rows[(rows["as_of"] < cut) & (rows["applied"] < cut)]
    if known["bill_id"].nunique() < MIN_TRAIN_ACTS:
        return None
    departments = list(known["department"].value_counts().head(cx.TOP_DEPARTMENTS).index)
    known = _dept_cols(known, departments)
    cols = cx.STATIC + cx.DYNAMIC + [f"dept_{i}" for i in range(len(departments))]
    p = cx._params()

    at_ra = known[known["checkpoint"] == "Royal Assent"]
    same_day = XGBClassifier(
        **{**p, "min_child_weight": 1.0},
        objective="binary:logistic",
        tree_method="hist",
        random_state=0,
    )
    same_day.fit(cx._x(at_ra, cols), (at_ra["target_days"] <= 0).astype(int))

    pos = known[known["target_days"] > 0]
    q = {}
    for alpha in (0.1, 0.5, 0.9):
        m = XGBRegressor(
            **p,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            tree_method="hist",
            random_state=0,
        )
        m.fit(cx._x(pos, cols), np.log1p(pos["target_days"]), sample_weight=cx._weights(pos))
        q[alpha] = m
    return Stage3Models(cut, cols, departments, same_day, q)


def _row(act: pd.Series, events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    ev = events[(events["bill_id"] == act["bill_id"]) & (events["event_date"] <= as_of)]
    row = act.to_dict()
    row.update(
        days_since_ra=float((as_of - act["royal_assent_date"]).days),
        regs_laid_so_far=float((ev["stage"] == "regulations_laid").sum()),
        regs_drafted_so_far=float((ev["stage"] == "regulations_drafted").sum()),
    )
    return pd.DataFrame([row])


def _awaiting_first_order(act: pd.Series, events: pd.DataFrame) -> bool:
    """A recent Act with no commencement order whose text leaves commencement to regulations:
    its "applied" date in the data is only Royal Assent, the first order is still to come."""
    data_end = events["event_date"].max()
    return bool(
        act["commencement_orders_count"] == 0
        and act["c_appoint"] == 1
        and act["royal_assent_date"] > data_end - pd.Timedelta(days=cx.RECENT_DAYS)
    )


def forecast(bill_id: int, as_of: date) -> dict[str, Any] | None:
    """Stage 3 for ``bill_id`` at ``as_of``: ``p_applies_on_royal_assent`` and ``days``
    (p10 / p50 / p90 days from ``as_of`` until the law applies; negative once it has applied).
    None if the bill has no Royal Assent by ``as_of`` or too little earlier data."""
    data = _data()
    if data is None:
        return None
    acts, events = data
    match = acts[acts["bill_id"] == bill_id]
    if match.empty:
        return None
    act = match.iloc[0]
    as_of_ts = pd.Timestamp(as_of).normalize()
    ra = act["royal_assent_date"]
    if pd.isna(ra) or as_of_ts < ra:
        return None
    applied = act["applied"]
    if _awaiting_first_order(act, events):
        applied = pd.NaT
    if pd.notna(applied) and applied <= as_of_ts:  # already applied: known outcome
        d = (applied - as_of_ts).days
        return {
            "applied": True,
            "p_applies_on_royal_assent": float(applied <= ra),
            "days": {0.1: d, 0.5: d, 0.9: d},
            "model_cut": None,
        }

    models = _models(as_of_ts.to_period("M").to_timestamp())
    if models is None:
        return None
    x = cx._x(_dept_cols(_row(act, events, as_of_ts), models.departments), models.cols)
    at_ra = as_of_ts == ra
    p0 = float(models.same_day.predict_proba(x)[0, 1]) if at_ra else 0.0
    later = {a: float(np.expm1(m.predict(x)[0]).clip(1)) for a, m in models.q.items()}
    later[0.9] = max(later[0.9], later[0.5])
    later[0.1] = min(later[0.1], later[0.5])
    # mixture of "applies today" (mass p0 at 0 days) and the regressors' window otherwise
    days = {a: 0 if p0 >= a else round(later[a]) for a in (0.1, 0.5, 0.9)}
    return {
        "applied": False,
        "p_applies_on_royal_assent": p0,
        "days": days,
        "model_cut": models.cut.date(),
    }
