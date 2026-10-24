"""Tier 1 features and hazard training rows (BUILD_PLAN task 4, MODEL_SPEC §4.1 and §6).

All data comes through the DATA_CONTRACT §5 interface. Two views of the same features:

- ``FEATURES``: the registry, one function per feature, ``(bill_id, as_of, data) -> value``,
  using only information available on ``as_of``. Used for live forecasts and to test the
  training rows for leakage.
- ``build_training_rows``: the vectorised bill × sitting-day rows for the advance hazard.

Timing rule for training rows: a row on sitting day ``t`` sees time-varying features as they
stood at the end of the previous day (``as_of = t - 1 day``), so the sitting that moves a bill
on day ``t`` can never leak into that day's features.

Clock: sitting days on which either House sat (MODEL_SPEC §3), from government-bill sittings
only (DATA_SCOPE.md).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from westminster_weather.data import calendar
from westminster_weather.domain.stages import BillStage
from westminster_weather.model import states
from westminster_weather.model.states import DataSource

TAU_BUCKETS: tuple[tuple[int, float, str], ...] = (
    (0, 0, "0"),
    (1, 5, "1-5"),
    (6, 20, "6-20"),
    (21, 60, "21-60"),
    (61, np.inf, "61+"),
)
END_BUCKETS: tuple[tuple[int, float, str], ...] = (
    (0, 5, "0-5"),
    (6, 15, "6-15"),
    (16, 40, "16-40"),
    (41, np.inf, "41+"),
)
END_REFERENCE_BUCKET = "41+"

# MODEL_SPEC §4.1: the training session end is the actual end date moved earlier by a random
# 0-20 sitting days (no announced dates exist yet), so the model cannot learn the exact end.
END_JITTER_MAX_SITTING_DAYS = 20

TOP_DEPARTMENTS = 10
DAYS_PER_MONTH = 365.25 / 12

# States whose advance is modelled by the hazard. Ping-pong rounds and the gap to Royal
# Assent are modelled separately (MODEL_SPEC §4.4); Royal Assent is absorbing.
HAZARD_STATES: tuple[BillStage, ...] = tuple(
    s for s in states.chain("Commons") if s not in (BillStage.PING_PONG, BillStage.ROYAL_ASSENT)
)

# Time-varying features published by ``get_covariates`` (value 0 until they first change).
COVARIATE_FEATURES = (
    "n_stage_sittings_cum",
    "committee_whole_house",
    "same_day_2r_3r",
    "carry_over",
)


def bucket(values: pd.Series, buckets: tuple[tuple[int, float, str], ...]) -> pd.Series:
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    for lo, hi, label in buckets:
        out[(values >= lo) & (values <= hi)] = label
    return out


def department_column(name: str) -> str:
    return "dept_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --------------------------------------------------------------------------- registry


def _bill(bill_id: int, as_of: date, data: Any) -> pd.Series | None:
    bills = data.get_bills(as_of)
    match = bills[bills["bill_id"] == bill_id]
    return match.iloc[0] if len(match) else None


def _covariate(name: str) -> Callable[[int, date, Any], float | None]:
    def value(bill_id: int, as_of: date, data: Any) -> float | None:
        if _bill(bill_id, as_of, data) is None:
            return None
        cov = data.get_covariates(as_of)
        rows = cov[(cov["bill_id"] == bill_id) & (cov["feature"] == name)]
        rows = rows[rows["valid_from"] <= as_of].sort_values("valid_from")
        return float(rows["value"].iloc[-1]) if len(rows) else 0.0

    return value


def _originating_house_lords(bill_id: int, as_of: date, data: Any) -> float | None:
    bill = _bill(bill_id, as_of, data)
    return None if bill is None else float(bill["originating_house"] == "Lords")


def _department(bill_id: int, as_of: date, data: Any) -> str | None:
    bill = _bill(bill_id, as_of, data)
    return None if bill is None else bill["department"]


def _introduced_months_into_session(bill_id: int, as_of: date, data: Any) -> float | None:
    bill = _bill(bill_id, as_of, data)
    if bill is None:
        return None
    sessions = data.get_calendar(as_of)["sessions"].set_index("session_id")
    start = sessions.loc[bill["introduced_session_id"], "start_date"]
    return (bill["introduced_date"] - start).days / DAYS_PER_MONTH


def _parliament_age_years(bill_id: int, as_of: date, data: Any) -> float | None:
    parls = data.get_calendar(as_of)["parliaments"]
    first = parls.loc[parls["first_meeting_date"] <= as_of, "first_meeting_date"].max()
    return (as_of - first).days / 365.25


def _days_to_session_end(bill_id: int, as_of: date, data: Any) -> float | None:
    """Sitting days to an announced prorogation or dissolution, if one is known on ``as_of``.

    None when nothing is announced: the simulator then draws the session end (MODEL_SPEC
    §4.2). Training rows use the jittered actual end instead (see ``build_training_rows``).
    """
    announced = data.get_calendar(as_of)["announced"]
    if announced.empty:
        return None
    ends = announced[
        announced["date_type"].isin(["prorogation", "dissolution"])
        & (announced["target_date"] >= as_of)
    ]
    if ends.empty:
        return None
    days = data.get_calendar(as_of)["sitting_days"]
    prob = calendar.week_sitting_probability(days, as_of)
    return float(len(calendar.future_sitting_days(as_of, ends["target_date"].min(), prob)))


@dataclass(frozen=True)
class Feature:
    name: str
    tier: int
    kind: str  # "fixed" (known at introduction) or "time_varying"
    description: str
    value: Callable[[int, date, Any], Any]


FEATURES: dict[str, Feature] = {
    f.name: f
    for f in (
        Feature(
            "originating_house_lords", 1, "fixed", "Bill started in the Lords",
            _originating_house_lords,
        ),
        Feature(
            "department", 1, "fixed", "Sponsoring department (top 10 one-hot + other)",
            _department,
        ),
        Feature(
            "introduced_months_into_session", 1, "fixed",
            "Months from session start to 1st reading", _introduced_months_into_session,
        ),
        Feature(
            "n_stage_sittings_cum", 1, "time_varying", "Stage sittings so far",
            _covariate("n_stage_sittings_cum"),
        ),
        Feature(
            "committee_whole_house", 1, "time_varying",
            "Committee of the whole House has happened", _covariate("committee_whole_house"),
        ),
        Feature(
            "same_day_2r_3r", 1, "time_varying", "Commons 2nd and 3rd reading on the same day",
            _covariate("same_day_2r_3r"),
        ),
        Feature(
            "carry_over", 1, "time_varying", "Carry-over motion agreed",
            _covariate("carry_over"),
        ),
        Feature(
            "parliament_age_years", 1, "time_varying", "Years since the Parliament first met",
            _parliament_age_years,
        ),
        Feature(
            "days_to_session_end", 1, "time_varying",
            "Sitting days to the session end (bucketed in the model)", _days_to_session_end,
        ),
    )
}  # fmt: skip


def feature_value(name: str, bill_id: int, as_of: date, data: DataSource | None = None) -> Any:
    """Registry lookup: the value of one feature for one bill, as known on ``as_of``."""
    return FEATURES[name].value(bill_id, as_of, states._source(data))


# --------------------------------------------------------------------------- training rows


@dataclass
class TrainingRows:
    rows: pd.DataFrame
    feature_columns: list[str]
    departments: list[str] = field(default_factory=list)
    as_of_cutoff: date | None = None

    @property
    def positive_rate(self) -> float:
        return float(self.rows["y"].mean()) if len(self.rows) else float("nan")


def _row_skeleton(trans: pd.DataFrame, cal: np.ndarray, cutoff: date) -> pd.DataFrame:
    """One row per (bill, state, sitting day at risk) with ``tau`` and ``y`` (MODEL_SPEC §4.1).

    - Entry day: a row with ``tau = 0`` (``y = 1`` if the bill moved on the same day).
    - Each later sitting day up to and including the exit day: ``tau`` = sitting days since
      entry; ``y = 1`` only on the day the next state is entered.
    - Failed states: rows up to and including the failure date, all ``y = 0``.
    - Censored states (still open at the cutoff): rows up to the cutoff, all ``y = 0``.
    """
    cutoff64 = np.datetime64(cutoff, "D")
    bill_ids, state_vals, dates, taus, ys = [], [], [], [], []
    for row in trans.itertuples(index=False):
        entered = np.datetime64(row.entered_date, "D")
        if row.exit_type == "advance":
            end = np.datetime64(row.exited_date, "D")
        elif row.exit_type == "fail":
            end = np.datetime64(row.exited_date, "D")
        else:
            end = cutoff64
        lo = np.searchsorted(cal, entered, side="right")
        hi = np.searchsorted(cal, end, side="left")
        between = cal[lo:hi]  # sitting days strictly between entry and exit
        if row.exit_type == "advance":
            if end == entered:
                days, y = np.array([entered]), np.array([1])
            else:
                days = np.concatenate([[entered], between, [end]])
                y = np.zeros(len(days), dtype=int)
                y[-1] = 1
        else:
            last = cal[hi : hi + 1] if hi < len(cal) and cal[hi] == end else cal[0:0]
            days = np.concatenate([[entered], between, last])
            y = np.zeros(len(days), dtype=int)
        n = len(days)
        bill_ids.append(np.full(n, row.bill_id))
        state_vals.append(np.full(n, row.model_state, dtype=object))
        dates.append(days)
        taus.append(np.arange(n))
        ys.append(y)

    return pd.DataFrame(
        {
            "bill_id": np.concatenate(bill_ids),
            "state": np.concatenate(state_vals),
            "date": np.concatenate(dates).astype("datetime64[ns]"),
            "tau": np.concatenate(taus),
            "y": np.concatenate(ys),
        }
    )


def _add_covariates(rows: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Time-varying covariates as they stood strictly before each row's date."""
    cov = cov.assign(valid_from=pd.to_datetime(cov["valid_from"]))
    rows = rows.sort_values("date")
    for name in COVARIATE_FEATURES:
        series = (
            cov[cov["feature"] == name][["bill_id", "valid_from", "value"]]
            .sort_values("valid_from")
            .rename(columns={"value": name})
        )
        rows = pd.merge_asof(
            rows,
            series,
            left_on="date",
            right_on="valid_from",
            by="bill_id",
            allow_exact_matches=False,
        ).drop(columns="valid_from")
        rows[name] = rows[name].fillna(0.0)
    return rows


def _add_session_end(
    rows: pd.DataFrame, sessions: pd.DataFrame, cal: np.ndarray, seed: int
) -> pd.DataFrame:
    """``days_to_session_end``: sitting days from the row date to the (jittered) end of the
    session the row falls in. NaN when the session had not ended by the cutoff."""
    sess = sessions.assign(start=pd.to_datetime(sessions["start_date"])).sort_values("start")
    rows = pd.merge_asof(
        rows.sort_values("date"),
        sess[["start", "session_id", "end_date"]].rename(columns={"session_id": "row_session"}),
        left_on="date",
        right_on="start",
    ).drop(columns="start")

    end_index = {}
    for sid, end in zip(sess["session_id"], sess["end_date"], strict=True):
        if end is None or pd.isna(end):
            continue
        idx = np.searchsorted(cal, np.datetime64(end, "D"), side="right") - 1
        jitter = np.random.default_rng([seed, int(sid)]).integers(
            0, END_JITTER_MAX_SITTING_DAYS + 1
        )
        end_index[sid] = idx - jitter

    row_index = np.searchsorted(cal, rows["date"].to_numpy().astype("datetime64[D]"))
    perceived_end = rows["row_session"].map(end_index)
    rows["days_to_session_end"] = np.clip(perceived_end - row_index, 0, None)
    return rows.drop(columns=["end_date"])


def build_training_rows(
    as_of_cutoff: date, data: DataSource | None = None, seed: int = 0
) -> TrainingRows:
    """Bill × sitting-day rows for the pooled advance hazard (MODEL_SPEC §4.1), using only
    data available on ``as_of_cutoff``.

    Columns: ``bill_id, date, introduced_session_id, state, tau, tau_bucket,
    days_to_session_end, days_to_session_end_bucket, <Tier 1 features>, y``.
    """
    src = states._source(data)
    bills = src.get_bills(as_of_cutoff)
    bills = bills[bills["introduced_session_id"] >= states.MIN_SESSION_ID]
    cal_tables = src.get_calendar(as_of_cutoff)
    cal = calendar.sitting_dates(cal_tables["sitting_days"])
    trans = states.build_transitions(as_of_cutoff, src)
    trans = trans[trans["model_state"].isin([s.value for s in HAZARD_STATES])]

    rows = _row_skeleton(trans, cal, as_of_cutoff)
    rows = _add_covariates(rows, src.get_covariates(as_of_cutoff))

    # Fixed features, known at introduction.
    sessions = cal_tables["sessions"]
    start = dict(zip(sessions["session_id"], sessions["start_date"], strict=True))
    fixed = bills[["bill_id", "introduced_session_id", "originating_house", "department"]].copy()
    fixed["originating_house_lords"] = (fixed["originating_house"] == "Lords").astype(float)
    fixed["introduced_months_into_session"] = [
        (intro - start[sid]).days / DAYS_PER_MONTH
        for intro, sid in zip(bills["introduced_date"], bills["introduced_session_id"], strict=True)
    ]
    departments = fixed["department"].dropna().value_counts().head(TOP_DEPARTMENTS).index.tolist()
    dept_cols = []
    for dept in departments:
        col = department_column(dept)
        fixed[col] = (fixed["department"] == dept).astype(float)
        dept_cols.append(col)
    rows = rows.merge(fixed.drop(columns=["originating_house", "department"]), on="bill_id")

    # Parliament age on the row date (first-meeting dates are known on the day).
    parls = cal_tables["parliaments"].assign(
        first=lambda p: pd.to_datetime(p["first_meeting_date"])
    )
    rows = pd.merge_asof(
        rows.sort_values("date"), parls[["first"]].sort_values("first"), left_on="date",
        right_on="first",
    )  # fmt: skip
    rows["parliament_age_years"] = (rows["date"] - rows["first"]).dt.days / 365.25
    rows = rows.drop(columns="first")

    rows = _add_session_end(rows, sessions, cal, seed)
    rows["tau_bucket"] = bucket(rows["tau"], TAU_BUCKETS)
    rows["days_to_session_end_bucket"] = bucket(rows["days_to_session_end"], END_BUCKETS).fillna(
        END_REFERENCE_BUCKET
    )

    feature_columns = [
        "originating_house_lords",
        "introduced_months_into_session",
        *COVARIATE_FEATURES,
        "parliament_age_years",
        *dept_cols,
    ]
    columns = [
        "bill_id",
        "date",
        "introduced_session_id",
        "state",
        "tau",
        "tau_bucket",
        "days_to_session_end",
        "days_to_session_end_bucket",
        *feature_columns,
        "y",
    ]
    rows = rows[columns].sort_values(["bill_id", "date", "state"]).reset_index(drop=True)
    return TrainingRows(rows, feature_columns, departments, as_of_cutoff)


def row_as_of(row_date: date | pd.Timestamp) -> date:
    """The ``as_of`` whose information a training row on ``row_date`` may use."""
    return pd.Timestamp(row_date).date() - timedelta(days=1)
