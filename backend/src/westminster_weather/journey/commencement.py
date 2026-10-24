"""Bill journey stage 3: from Royal Assent to commencement (commencement orders).

What the data supports. ``act_commencements`` (one row per Act, Government Bills only) and
``post_royal_assent_events`` record, for every Act since late 2006, its Royal Assent date and
the dates of its commencement orders (statutory instruments that bring sections into force).
``is_fully_in_force`` is True for all Acts and ``law_fully_in_force_date`` is just the latest
commencement order date (or Royal Assent when there is none), so the data cannot say when an Act
is *fully* in force. This module therefore models only what is observed:

1. **Needs commencement orders?** and **when is the first one?** -- the take-up mixture cure
   model (``takeup.model.TakeUpModel``) in calendar weeks from Royal Assent with an all-ones
   mask (ministers can make an order on any day, so there is no sitting calendar). "Cured" =
   never needs an order (the Act commences on Royal Assent or on fixed dates); the event is the
   first commencement order. ``z`` features, all known at Royal Assent: top departments,
   Lords-first, log stage sittings, years from first reading to Royal Assent, ping-pong,
   months into the session and years since the election (standardised), draft / laid
   regulations already published by Royal Assent (dropped when too rare). They move only
   P(needs orders): the weekly hazard has no covariates (giving departments etc. hazard
   shifts too was tried and scored worse), so the first-order timing curve is shared by all
   Acts, conditioned on the weeks already passed.
2. **When are the orders finished?** -- the week T of the LATEST commencement order, among
   Acts with orders. T can only fall in a week with an order, so a weekly hazard of T itself
   (tried first) put too little weight on "the order just seen was the last" and was badly
   calibrated after the first order. Instead ``OrderModel`` is a weekly discrete-time hazard
   (L2 logistic, as ``takeup.hazard``) of a FURTHER order in week t > t1, with recency
   features known at the week start (weeks since the last order, orders so far, weeks since
   Royal Assent) and the static features; T is the last week of simulated order paths
   (``N_SIMS``, seeded by Act and as_of, up to ``FINISH_HORIZON`` = 520 weeks). Training rows
   are only weeks observed by the cut, so censoring is exact (an Act whose latest order is
   recent simply contributes fewer weeks). For SCORING, an Act's orders count as finished at
   its latest order only when ``FINISH_QUIET_WEEKS`` (104) weeks followed with no further order
   before the data end.
3. **Forecasts at ``as_of``** (``forecast_act``), using only events dated <= ``as_of``:
   - before any order, s whole weeks after Royal Assent: P(needs orders) = the cure model's
     ``pi_now`` (long silence lowers it), the first-order timing quantiles given an order
     comes (> s), and the finish timing from order paths started at first-order weeks drawn
     from the cure model's conditional timing distribution;
   - after the first order (needs orders = 1, first order known), with the latest order seen
     in week L0: order paths are simulated from the current week; ``p_finished_already`` =
     the share of paths with no further order, ``p_by_week`` = P(T <= 52 / 104 / 208).
   80% windows [p10, p90] are CQR-calibrated (``takeup.conformal``, scale-normalised scores)
   on Acts with Royal Assent in the 4 (else 6, 8) years before the fit cut, predicted at
   their Royal Assent by models fitted only on data visible at the start of that window;
   fewer than ``MIN_CALIBRATION`` observed outcomes -> raw windows and ``calibrated = False``.

Weeks: week t covers days 7(t-1)..7t-1 after Royal Assent (an order on the Royal Assent day is
in week 1). Nothing is written to any database; data is read with plain SELECTs in a read-only
transaction (``load_tables``).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from westminster_weather.takeup import model as tm
from westminster_weather.takeup.conformal import (
    MIN_CALIBRATION,
    CQRCalibrator,
    conformal_quantile,
)
from westminster_weather.takeup.signals import department_slug

log = logging.getLogger(__name__)

FIRST_HORIZON = tm.T_MAX  # weeks followed for the first order (5 years; the cure model's)
FINISH_HORIZON = 520  # weeks followed for the latest order (10 years)
FINISH_QUIET_WEEKS = 104  # an Act's orders are 'finished' after this long with no new order
NO_ORDER_KNOWN_WEEKS = 156  # no order this long after Royal Assent -> 'never needed' (scoring)
CALIBRATION_YEARS = 4  # validation window for the finish model's C
CALIBRATION_YEARS_TRY = (4, 6, 8)  # CQR calibration windows, widened until enough outcomes
TRAIN_CUT = date(2019, 1, 1)
TOP_DEPARTMENTS = 6
MIN_DEPT_ACTS = 8
MIN_POSITIVE = 8  # binary features with fewer positives in training are dropped
SINCE_EDGES = (1, 5, 13, 27, 53, 105, 209, 365)  # weeks since Royal Assent bins
GAP_EDGES = (1, 5, 13, 27, 53, 105)  # weeks since the last order bins
N_SIMS = 1000  # simulated order paths per forecast
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_C = 0.1
QUANTILES = (0.1, 0.5, 0.9)
FINISH_BY_WEEKS = (52, 104, 208)  # orders_finished.p_by_week: P(latest order by week H)
PING_PONG_STAGES = (12, 13, 42)  # consideration of Lords / Commons amendments, Lords message
CONTINUOUS = (
    "z_log_stage_sittings",
    "z_years_1r_to_ra",
    "z_months_into_session",
    "z_years_since_election",
)
BINARY = ("z_lords_first", "z_ping_pong", "z_draft_regs_by_ra", "z_laid_regs_by_ra")
_EPOCH = pd.Timestamp("1970-01-01")

# ----------------------------------------------------------------------------- data


def _day(value: Any) -> float:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    return float((pd.Timestamp(value).normalize() - _EPOCH).days)


def _date(day: float | None) -> date | None:
    if day is None or not np.isfinite(day):
        return None
    return (_EPOCH + pd.Timedelta(int(day), unit="D")).date()


def _week(day: float, ra: float) -> int:
    """Week number of ``day`` after Royal Assent (1 = days 0-6)."""
    return int(np.floor((day - ra) / 7.0)) + 1


@dataclass
class CommencementData:
    """Acts, their static Royal Assent features and post-Royal-Assent event dates.

    ``acts``: one row per Act (bill_id, act_title, royal_assent_date, department, ra_day and
    the raw features). ``orders`` / ``drafts`` / ``laid``: bill_id -> sorted day numbers.
    ``data_end``: the last event date on record (outcomes are not observed after it)."""

    acts: pd.DataFrame
    orders: dict[int, np.ndarray]
    drafts: dict[int, np.ndarray]
    laid: dict[int, np.ndarray]
    data_end: pd.Timestamp

    def orders_until(self, bill_id: int, day: float) -> np.ndarray:
        o = self.orders.get(int(bill_id), np.array([]))
        return o[o <= day]

    def act(self, bill_id: int) -> pd.Series:
        hit = self.acts[self.acts["bill_id"] == int(bill_id)]
        if hit.empty:
            raise KeyError(f"Act for bill {bill_id} not found in act_commencements")
        return hit.iloc[0]


def _event_days(ev: pd.DataFrame, stage: str) -> dict[int, np.ndarray]:
    e = ev[ev["stage"] == stage]
    out: dict[int, np.ndarray] = {}
    for bid, g in e.groupby("bill_id"):
        days = np.sort(np.array([_day(x) for x in g["event_date"]], dtype=float))
        out[int(bid)] = days[np.isfinite(days)]
    return out


def build_data(tables: dict[str, pd.DataFrame]) -> CommencementData:
    """``CommencementData`` from tables ``act_commencements``, ``post_royal_assent_events``,
    ``bills``, ``stage_sittings``, ``sessions``, ``parliaments`` (pure; for tests too).

    Commencement-order dates come from the events table (``stage = 'commencement_orders'``);
    an Act with no such events but a ``first_commencement_order_date`` gets that date (and the
    latest) from ``act_commencements``."""
    act = tables["act_commencements"].copy()
    ev = tables["post_royal_assent_events"].copy()
    bills = tables["bills"]
    ss = tables["stage_sittings"]
    sessions = tables["sessions"]
    parls = tables["parliaments"]

    orders = _event_days(ev, "commencement_orders")
    for r in act.itertuples(index=False):
        bid = int(r.bill_id)
        if bid not in orders:
            days = [
                _day(getattr(r, c, None))
                for c in ("first_commencement_order_date", "latest_commencement_order_date")
            ]
            days = sorted({d for d in days if np.isfinite(d)})
            if days:
                orders[bid] = np.array(days, dtype=float)
    drafts = _event_days(ev, "regulations_drafted")
    laid = _event_days(ev, "regulations_laid")

    ends = [_day(x) for x in ev["event_date"]] if len(ev) else []
    for c in ("royal_assent_date", "latest_commencement_order_date"):
        if c in act:
            ends += [_day(x) for x in act[c]]
    ends = [e for e in ends if np.isfinite(e)]
    data_end = _EPOCH + pd.Timedelta(int(max(ends)), unit="D")

    b = bills.drop_duplicates("bill_id").set_index("bill_id")
    sess_start = np.sort(np.array([_day(x) for x in sessions["start_date"]], dtype=float))
    sess_start = sess_start[np.isfinite(sess_start)]
    elections = np.sort(np.array([_day(x) for x in parls["election_date"]], dtype=float))
    elections = elections[np.isfinite(elections)]
    ss = ss.assign(_d=[_day(x) for x in ss["event_date"]])

    rows = []
    for r in act.itertuples(index=False):
        bid = int(r.bill_id)
        ra = _day(r.royal_assent_date)
        if not np.isfinite(ra):
            continue
        dept = b["department"].get(bid) if "department" in b else None
        house = b["originating_house"].get(bid) if "originating_house" in b else None
        sit = ss[(ss["bill_id"] == bid) & (ss["_d"] <= ra)]
        first_reading = sit.loc[sit["api_stage_id"].isin((1, 6)), "_d"].min()
        if first_reading is None or pd.isna(first_reading):
            intro = b["introduced_date"].get(bid) if "introduced_date" in b else None
            first_reading = _day(intro)
        s_idx = np.searchsorted(sess_start, ra, side="right") - 1
        e_idx = np.searchsorted(elections, ra, side="right") - 1
        rows.append(
            {
                "bill_id": bid,
                "act_title": getattr(r, "act_title", None),
                "royal_assent_date": pd.Timestamp(_date(ra)),
                "ra_day": ra,
                "department": None if dept is None or pd.isna(dept) else str(dept),
                "lords_first": float(str(house).lower() == "lords"),
                "n_stage_sittings": float(len(sit)),
                "days_1r_to_ra": float(ra - first_reading)
                if first_reading is not None and np.isfinite(first_reading)
                else np.nan,
                "ping_pong": float(sit["api_stage_id"].isin(PING_PONG_STAGES).any()),
                "months_into_session": (ra - sess_start[s_idx]) / 30.4375 if s_idx >= 0 else np.nan,
                "years_since_election": (ra - elections[e_idx]) / 365.25 if e_idx >= 0 else np.nan,
                "draft_regs_by_ra": float((drafts.get(bid, np.array([])) <= ra).any()),
                "laid_regs_by_ra": float((laid.get(bid, np.array([])) <= ra).any()),
            }
        )
    acts = pd.DataFrame(rows).sort_values(["ra_day", "bill_id"]).reset_index(drop=True)
    return CommencementData(acts, orders, drafts, laid, pd.Timestamp(data_end).normalize())


def load_tables() -> dict[str, pd.DataFrame]:
    """The six source tables from production with plain SELECTs in a read-only transaction
    (the ``westminster_weather.db.fetch_*`` adapters are not used: they may seed tables)."""
    from sqlalchemy import text

    from westminster_weather import db as app_db

    queries = {
        "act_commencements": "SELECT * FROM act_commencements",
        "post_royal_assent_events": (
            "SELECT bill_id, stage, event_date, document_type FROM post_royal_assent_events"
        ),
        "bills": (
            "SELECT bill_id, short_title, originating_house, introduced_date, department FROM bills"
        ),
        "stage_sittings": "SELECT bill_id, api_stage_id, event_date FROM stage_sittings",
        "sessions": "SELECT session_id, start_date FROM sessions",
        "parliaments": "SELECT parliament_id, election_date FROM parliaments",
    }
    engine = app_db.get_sync_pooled_engine()
    out: dict[str, pd.DataFrame] = {}
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(postgresql_readonly=True)
            for name, q in queries.items():
                res = conn.execute(text(q))
                out[name] = pd.DataFrame(res.mappings().all(), columns=list(res.keys()))
            conn.rollback()
    finally:
        engine.dispose()
    if out["parliaments"].empty:
        out["parliaments"] = pd.DataFrame(
            [(p[0], p[3]) for p in app_db.DEFAULT_PARLIAMENTS],
            columns=["parliament_id", "election_date"],
        )
    return out


_loaded: dict[str, Any] = {"data": None}
_load_lock = threading.Lock()


def load_data(refresh: bool = False) -> CommencementData:
    """``build_data(load_tables())``, loaded once per process (``refresh`` reloads)."""
    with _load_lock:
        if _loaded["data"] is None or refresh:
            _loaded["data"] = build_data(load_tables())
            _FORECASTERS.clear()
        return _loaded["data"]


# ------------------------------------------------------------------------- features


@dataclass
class StaticFeatures:
    """Royal Assent features: top departments chosen and continuous ones standardised on the
    training Acts; rare binary features dropped."""

    departments: list[str] = field(default_factory=list)
    mu: dict[str, float] = field(default_factory=dict)
    sd: dict[str, float] = field(default_factory=dict)
    binary: list[str] = field(default_factory=list)

    @staticmethod
    def raw(acts: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dept": [department_slug(x) for x in acts["department"]],
                "z_lords_first": acts["lords_first"].to_numpy(float),
                "z_ping_pong": acts["ping_pong"].to_numpy(float),
                "z_draft_regs_by_ra": acts["draft_regs_by_ra"].to_numpy(float),
                "z_laid_regs_by_ra": acts["laid_regs_by_ra"].to_numpy(float),
                "z_log_stage_sittings": np.log1p(acts["n_stage_sittings"].to_numpy(float)),
                "z_years_1r_to_ra": acts["days_1r_to_ra"].to_numpy(float) / 365.25,
                "z_months_into_session": np.minimum(
                    acts["months_into_session"].to_numpy(float), 24.0
                ),
                "z_years_since_election": np.minimum(
                    acts["years_since_election"].to_numpy(float), 5.5
                ),
            },
            index=acts.index,
        )

    def fit(self, acts: pd.DataFrame) -> StaticFeatures:
        r = self.raw(acts)
        counts = r["dept"].value_counts()
        self.departments = [d for d, n in counts.items() if n >= MIN_DEPT_ACTS][:TOP_DEPARTMENTS]
        for c in CONTINUOUS:
            v = r[c].to_numpy(float)
            mu = float(np.nanmean(v)) if np.isfinite(v).any() else 0.0
            sd = float(np.nanstd(v)) if np.isfinite(v).any() else 1.0
            self.mu[c], self.sd[c] = mu, sd if sd > 1e-9 else 1.0
        self.binary = [c for c in BINARY if r[c].sum() >= MIN_POSITIVE]
        return self

    @property
    def columns(self) -> list[str]:
        return [*(f"z_dept_{d}" for d in self.departments), *self.binary, *CONTINUOUS]

    def transform(self, acts: pd.DataFrame) -> pd.DataFrame:
        r = self.raw(acts)
        out = pd.DataFrame(index=acts.index)
        for d in self.departments:
            out[f"z_dept_{d}"] = (r["dept"] == d).astype(float)
        for c in self.binary:
            out[c] = r[c]
        for c in CONTINUOUS:  # missing -> the training mean (0 after standardising)
            out[c] = ((r[c] - self.mu[c]) / self.sd[c]).fillna(0.0)
        return out


# ------------------------------------------------------- order-process hazard model


class OrderModel:
    """Weekly hazard of a FURTHER commencement order, after an Act's first order.

    One row per Act-week t > t1 (first-order week) that is fully observed by the cut, up to
    ``FINISH_HORIZON``; y = 1 if at least one order falls in week t. Features use only orders
    in weeks before t: weeks-since-Royal-Assent bins and log t, weeks since the last order
    (bins and log), log1p orders so far, and the static Royal Assent features; L2 logistic
    (sklearn, standardised), C chosen on the last ``CALIBRATION_YEARS`` before the cut.
    Censoring is exact: an Act contributes only weeks seen by the cut, so recent Acts whose
    orders may continue are not mislabelled as finished."""

    def __init__(self, zcols: Sequence[str], c_grid: Sequence[float] = C_GRID) -> None:
        self.zcols = list(zcols)
        self.c_grid = tuple(c_grid)
        self.C = DEFAULT_C
        self.mu = self.sd = None
        self.coef: np.ndarray | None = None
        self.n_rows = self.n_events = 0
        self.feature_names = [
            *(f"since_ra_{e}+" for e in SINCE_EDGES[1:]),
            "log_week",
            *(f"gap_{e}+" for e in GAP_EDGES[1:]),
            "log_gap",
            "log1p_orders",
            *self.zcols,
        ]

    def features(self, t: np.ndarray, gap: np.ndarray, k: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Rows for weeks ``t`` with ``gap`` weeks since the last order and ``k`` orders so far
        (arrays of one length) and static features ``z``."""
        t = np.asarray(t, dtype=float)
        gap = np.asarray(gap, dtype=float)
        since = np.searchsorted(SINCE_EDGES, t, side="right") - 1
        gb = np.searchsorted(GAP_EDGES, gap, side="right") - 1
        cols = [
            *((since == j).astype(float) for j in range(1, len(SINCE_EDGES))),
            np.log(t),
            *((gb == j).astype(float) for j in range(1, len(GAP_EDGES))),
            np.log(np.maximum(gap, 1.0)),
            np.log1p(np.asarray(k, dtype=float)),
            *(np.full(len(t), float(v)) for v in np.asarray(z, dtype=float)),
        ]
        return np.column_stack(cols)

    @staticmethod
    def order_weeks(data: CommencementData, bill_id: int, ra: float, until: float) -> np.ndarray:
        return np.array([_week(d, ra) for d in data.orders_until(bill_id, until)], dtype=int)

    def fit(
        self,
        data: CommencementData,
        acts: pd.DataFrame,
        Z: pd.DataFrame,
        cut: float,
        val_from: float | None = None,
    ) -> OrderModel:
        Xs, ys, starts = [], [], []
        for k_row, a in acts.iterrows():
            ra = float(a["ra_day"])
            ow = self.order_weeks(data, a["bill_id"], ra, cut)
            if not len(ow):
                continue
            last_full = min(int(np.floor((cut - ra + 1) / 7.0)), FINISH_HORIZON)
            weeks = np.arange(ow[0] + 1, last_full + 1)
            if not len(weeks):
                continue
            n_before = np.searchsorted(ow, weeks, side="left")  # orders in weeks < t
            last_before = ow[np.maximum(n_before - 1, 0)]
            X = self.features(weeks, weeks - last_before, n_before, Z.loc[k_row, self.zcols])
            Xs.append(X)
            ys.append(np.isin(weeks, ow).astype(float))
            starts.append(ra + 7.0 * (weeks - 1))
        if not ys or sum(y.sum() for y in ys) == 0:
            raise ValueError("no further commencement orders to fit the order model on")
        X, y, ws = np.concatenate(Xs), np.concatenate(ys), np.concatenate(starts)
        self.n_rows, self.n_events = len(y), int(y.sum())
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        Xn = (X - self.mu) / self.sd
        if val_from is not None and len(self.c_grid) > 1:
            tr, va = ws < val_from, ws >= val_from
            if y[tr].sum() >= 10 and y[va].sum() >= 5:
                best = None
                for C in self.c_grid:
                    m = self._logit(C).fit(Xn[tr], y[tr])
                    p = np.clip(m.predict_proba(Xn[va])[:, 1], 1e-12, 1 - 1e-12)
                    ll = -np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log1p(-p))
                    if best is None or ll < best[0]:
                        best = (ll, C)
                self.C = best[1]
        m = self._logit(self.C).fit(Xn, y)
        self.coef = np.concatenate([m.intercept_, m.coef_[0]])
        return self

    @staticmethod
    def _logit(C: float) -> LogisticRegression:
        return LogisticRegression(C=C, max_iter=5000, tol=1e-8)

    def hazard(self, t: np.ndarray, gap: np.ndarray, k: np.ndarray, z: np.ndarray) -> np.ndarray:
        X = (self.features(t, gap, k, z) - self.mu) / self.sd
        return expit(self.coef[0] + X @ self.coef[1:])

    def simulate_last(
        self,
        z: np.ndarray,
        start: np.ndarray,
        last: np.ndarray,
        count: np.ndarray,
        rng: np.random.Generator,
        horizon: int = FINISH_HORIZON,
    ) -> np.ndarray:
        """Week of the latest order (within the horizon) for each path, simulating weeks
        ``start..horizon`` from state (``last`` order week, ``count`` orders); weeks before a
        path's ``start`` are already known."""
        start = np.asarray(start, dtype=int)
        last = np.asarray(last, dtype=int).copy()
        count = np.asarray(count, dtype=float).copy()
        lo = int(start.min()) if len(start) else horizon + 1
        for t in range(max(lo, 2), horizon + 1):
            active = start <= t
            if not active.any():
                continue
            idx = np.nonzero(active)[0]
            h = self.hazard(np.full(len(idx), t), t - last[idx], count[idx], z)
            hit = idx[rng.random(len(idx)) < h]
            last[hit] = t
            count[hit] += 1
        return last


# ------------------------------------------------------------------- calibration


def _num(values: Sequence[Any], missing: float) -> np.ndarray:
    return np.array([missing if v is None or pd.isna(v) else float(v) for v in values], float)


@dataclass
class WindowCalibrator:
    """Split-conformal (CQR, scale-normalised) widening of [p10, p90] for a horizon."""

    horizon: int
    q: float = 0.0
    n: int = 0
    raw_coverage: float | None = None

    def fit(self, t: Sequence[float], lo: Sequence[Any], hi: Sequence[Any]) -> WindowCalibrator:
        t = np.asarray(t, dtype=float)
        lo_a, hi_a = _num(lo, self.horizon + 1), _num(hi, self.horizon + 1)
        w = np.maximum(hi_a - lo_a, 1.0)
        sc = np.maximum((lo_a - t) / w, (t - hi_a) / w)
        self.n = len(sc)
        self.raw_coverage = float(np.mean(sc <= 0)) if len(sc) else None
        self.q = conformal_quantile(sc) if self.n >= MIN_CALIBRATION else 0.0
        return self

    @property
    def usable(self) -> bool:
        return self.n >= MIN_CALIBRATION

    def apply(self, lo: Any, hi: Any, floor: int, mid: Any = None) -> tuple[int | None, ...]:
        """[lo - q w, hi + q w] with the lower end >= ``floor`` (and <= mid, upper >= mid);
        an end beyond the horizon (or infinite q) is None."""
        if lo is None and hi is None:
            return None, None
        lo_v = self.horizon + 1 if lo is None else float(lo)
        hi_v = self.horizon + 1 if hi is None else float(hi)
        w = max(hi_v - lo_v, 1.0)
        if math.isinf(self.q):
            new_lo, new_hi = float(floor), math.inf
        else:
            new_lo, new_hi = math.floor(lo_v - self.q * w), math.ceil(hi_v + self.q * w)
        new_lo = max(new_lo, floor)
        if mid is not None:
            new_lo, new_hi = min(new_lo, mid), max(new_hi, mid)
        new_hi = max(new_hi, new_lo)
        return (
            None if new_lo > self.horizon else int(new_lo),
            None if new_hi > self.horizon else int(new_hi),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "n_calibration": self.n,
            "q": None if math.isinf(self.q) else self.q,
            "used": self.usable,
            "raw_coverage_on_calibration": self.raw_coverage,
        }


# --------------------------------------------------------------------- forecaster


def _quantiles(pmf: np.ndarray, start_week: int = 1) -> dict[str, int | None]:
    """Timing quantiles of a (normalised) pmf over weeks start_week..; None if not reached."""
    cum = np.cumsum(pmf)
    out: dict[str, int | None] = {}
    for q in QUANTILES:
        hit = np.nonzero(cum >= q - 1e-12)[0]
        out[tm.quantile_key(q)] = int(start_week + hit[0]) if len(hit) else None
    return out


def _cure_frame(
    data: CommencementData, acts: pd.DataFrame, Z: pd.DataFrame, cut: float
) -> pd.DataFrame:
    """Rows for ``tm.prepare``: signal = Royal Assent, 'bill' = first order seen by ``cut``."""
    firsts = []
    for bid in acts["bill_id"]:
        o = data.orders_until(bid, cut)
        firsts.append(pd.Timestamp(_date(o[0])) if len(o) else pd.NaT)
    df = pd.DataFrame(
        {
            "signal_id": acts["bill_id"].astype(str).to_numpy(),
            "title": acts["act_title"].astype(str).to_numpy(),
            "signal_date": acts["royal_assent_date"].to_numpy(),
            "bill_date": pd.to_datetime(pd.Series(firsts)).to_numpy(),
            "data_end": pd.Timestamp(_date(cut)),
        }
    )
    for c in Z.columns:
        df[c] = Z[c].to_numpy(float)
    return df


def _all_days(start: float, end: float) -> np.ndarray:
    """Every day in [start, end]: an all-ones 'sitting' mask (orders can come any day)."""
    return (np.arange(int(start), int(end) + 1).astype("datetime64[D]")).astype("datetime64[D]")


class CommencementForecaster:
    """Stage-3 models fitted on Acts with Royal Assent before ``cut`` (seen as of ``cut``)."""

    def __init__(self, data: CommencementData, cut: Any, calibrate: bool = True) -> None:
        self.data = data
        self.cut = pd.Timestamp(cut).normalize()
        cut_day = _day(self.cut)
        self.cut_day = cut_day
        train = data.acts[data.acts["ra_day"] < cut_day]
        if len(train) < 20:
            raise ValueError(f"too few Acts before {self.cut.date()} to fit stage 3")
        self.static = StaticFeatures().fit(train)
        Z = self.static.transform(train)
        self.zcols = self.static.columns
        df = _cure_frame(data, train, Z, cut_day)
        d = tm.prepare(
            df,
            _all_days(train["ra_day"].min(), cut_day),
            None,
            data_end=self.cut,
            calendar_end=self.cut,
        )
        self.n_train = len(train)
        self.n_first_events = int(d["delta"].sum())
        self.cure = tm.TakeUpModel().fit(d)
        val_from = cut_day - 365.25 * CALIBRATION_YEARS
        self.orders = OrderModel(self.zcols).fit(data, train, Z, cut_day, val_from=val_from)
        self.first_cal = CQRCalibrator()
        self.finish_cal = WindowCalibrator(FINISH_HORIZON)
        self.first_cal_years: int | None = None
        self.finish_cal_years: int | None = None
        if calibrate:
            self._calibrate()

    # ----------------------------------------------------------------- calibration

    def _calibration_set(self, years: int) -> dict[str, list[Any]] | None:
        """Raw windows and observed outcomes (seen by the cut) for Acts with Royal Assent in
        ``[cut - years, cut)``, predicted at Royal Assent by models fitted at ``cut - years``."""
        cal_cut = self.cut - pd.DateOffset(years=years)
        try:
            inner = CommencementForecaster(self.data, cal_cut, calibrate=False)
        except ValueError as exc:
            log.warning("stage 3 calibration model at %s failed: %s", cal_cut.date(), exc)
            return None
        acts = self.data.acts
        cal = acts[(acts["ra_day"] >= _day(cal_cut)) & (acts["ra_day"] < self.cut_day)]
        out: dict[str, list[Any]] = {k: [] for k in ("t1", "f_lo", "f_hi", "T", "g_lo", "g_hi")}
        for _, a in cal.iterrows():
            o = self.data.orders_until(a["bill_id"], self.cut_day)
            if not len(o):
                continue
            p = inner.predict(a, a["ra_day"])
            ra = float(a["ra_day"])
            t1 = _week(o[0], ra)
            if t1 <= FIRST_HORIZON:
                out["t1"].append(t1)
                out["f_lo"].append(p["first_order"]["raw"]["t_10"])
                out["f_hi"].append(p["first_order"]["raw"]["t_90"])
            if o[-1] <= self.cut_day - 7 * FINISH_QUIET_WEEKS:
                out["T"].append(min(_week(o[-1], ra), FINISH_HORIZON + 1))
                out["g_lo"].append(p["orders_finished"]["raw"]["t_10"])
                out["g_hi"].append(p["orders_finished"]["raw"]["t_90"])
        return out

    def _calibrate(self) -> None:
        """CQR calibrators; the window is widened through ``CALIBRATION_YEARS_TRY`` until each
        target has ``MIN_CALIBRATION`` observed outcomes (finished orders are rarer)."""
        first_done = finish_done = False
        for years in CALIBRATION_YEARS_TRY:
            cs = self._calibration_set(years)
            if cs is None:
                continue
            if not first_done and cs["t1"]:
                self.first_cal = CQRCalibrator().fit(cs["t1"], cs["f_lo"], cs["f_hi"])
                self.first_cal_years = years
                first_done = self.first_cal.n >= MIN_CALIBRATION
            if not finish_done and cs["T"]:
                self.finish_cal = WindowCalibrator(FINISH_HORIZON).fit(
                    cs["T"], cs["g_lo"], cs["g_hi"]
                )
                self.finish_cal_years = years
                finish_done = self.finish_cal.usable
            if first_done and finish_done:
                break

    # --------------------------------------------------------------------- predict

    def predict(self, act: pd.Series, as_of_day: float) -> dict[str, Any]:
        """Forecast for one Act (a row of ``data.acts``) with events <= ``as_of_day``."""
        ra = float(act["ra_day"])
        if as_of_day < ra:
            raise ValueError("as_of is before Royal Assent")
        z = self.static.transform(pd.DataFrame([act])).iloc[0].to_numpy(float)
        o = self.data.orders_until(act["bill_id"], as_of_day)
        days = as_of_day - ra
        s = int(min(np.floor(days / 7.0), FIRST_HORIZON))  # whole weeks seen
        w_now = int(np.floor(days / 7.0)) + 1  # current (partly seen) week
        rng = np.random.default_rng([int(act["bill_id"]), int(as_of_day)])
        out: dict[str, Any] = {"weeks_since_royal_assent": int(np.floor(days / 7.0))}
        if len(o):
            ow = np.array([_week(d, ra) for d in o], dtype=int)
            t1, L0 = int(ow[0]), int(ow[-1])
            out["needs_orders"] = {"p": 1.0, "p_at_royal_assent": None, "observed": True}
            out["first_order"] = {
                "observed": True,
                "week": t1,
                "raw": {"t_10": t1, "t_50": t1, "t_90": t1},
            }
            n = N_SIMS
            start = np.full(n, max(w_now, L0 + 1))
            T = self.orders.simulate_last(z, start, np.full(n, L0), np.full(n, len(o)), rng)
            out["orders_seen"] = len(o)
            floor_fin = L0
        else:
            xw = np.zeros(0)  # no x_* features: the timing curve is shared by all Acts
            pred = self.cure.predict(z, xw, weeks_elapsed=s)
            pi_ra = self.cure.predict(z, xw, weeks_elapsed=0)["pi"]
            out["needs_orders"] = {
                "p": float(pred["pi_now"]),
                "p_at_royal_assent": float(pi_ra),
                "p_lo_at_royal_assent": float(pred["pi_lo"]),
                "p_hi_at_royal_assent": float(pred["pi_hi"]),
                "p_first_order_52w": float(pred["p52"]),
                "observed": False,
            }
            out["first_order"] = {
                "observed": False,
                "week": None,
                "raw": {k: pred[k] for k in ("t_10", "t_50", "t_90")},
            }
            # first-order week drawn from the cure model's pmf for weeks s+1..FIRST_HORIZON,
            # given an order comes (renormalised), then the order process from there
            logS = np.concatenate([[0.0], self.cure.log_survival_u(xw)])
            S = np.exp(logS - logS[s])
            g = np.clip(S[s:-1] - S[s + 1 :], 0, None)
            T = np.zeros(0, dtype=int)
            if g.sum() > 0:
                t1s = s + 1 + rng.choice(len(g), size=N_SIMS, p=g / g.sum())
                T = self.orders.simulate_last(z, t1s + 1, t1s, np.ones(N_SIMS), rng)
            out["orders_seen"] = 0
            floor_fin = s + 1
            L0 = None
        if len(T):
            qs = np.quantile(T, QUANTILES, method="inverted_cdf")
            raw = {tm.quantile_key(q): int(v) for q, v in zip(QUANTILES, qs, strict=True)}
            by = {str(H): float(np.mean(T <= H)) for H in FINISH_BY_WEEKS}
        else:
            raw = {tm.quantile_key(q): None for q in QUANTILES}
            by = {str(H): None for H in FINISH_BY_WEEKS}
        out["orders_finished"] = {
            "observed": False,
            "latest_order_week": L0,
            "p_finished_already": float(np.mean(T == L0)) if L0 is not None and len(T) else 0.0,
            "p_by_week": by,
            "raw": raw,
        }
        # CQR windows
        fo = out["first_order"]
        if fo["observed"]:
            fo["calibrated"] = dict(fo["raw"])
            fo["is_calibrated"] = False
        else:
            raw = fo["raw"]
            ok = self.first_cal.n >= MIN_CALIBRATION and raw["t_50"] is not None
            lo, hi = raw["t_10"], raw["t_90"]
            if ok:
                (lo,), (hi,) = self.first_cal.apply([lo], [hi], s)
                lo = raw["t_50"] if lo is None else min(lo, raw["t_50"])
                hi = None if hi is None else max(hi, raw["t_50"])
            fo["calibrated"] = {"t_10": lo, "t_50": raw["t_50"], "t_90": hi}
            fo["is_calibrated"] = bool(ok)
        of = out["orders_finished"]
        raw = of["raw"]
        if self.finish_cal.usable and raw["t_50"] is not None:
            lo, hi = self.finish_cal.apply(raw["t_10"], raw["t_90"], floor_fin, raw["t_50"])
            of["calibrated"] = {"t_10": lo, "t_50": raw["t_50"], "t_90": hi}
            of["is_calibrated"] = True
        else:
            of["calibrated"] = dict(raw)
            of["is_calibrated"] = False
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "cut": str(self.cut.date()),
            "n_train_acts": self.n_train,
            "n_first_order_events": self.n_first_events,
            "z_features": list(self.zcols),
            "cure_parameters": [
                {"parameter": r.parameter, "estimate": float(r.estimate)}
                for r in self.cure.parameter_table().itertuples()
            ],
            "order_model_C": self.orders.C,
            "order_model_rows": self.orders.n_rows,
            "order_model_events": self.orders.n_events,
            "order_model_parameters": [
                {"parameter": n, "estimate": float(c)}
                for n, c in zip(
                    ["intercept", *self.orders.feature_names],
                    np.concatenate([[self.orders.coef[0]], self.orders.coef[1:] / self.orders.sd]),
                    strict=True,
                )
            ],
            "first_order_cqr": {**self.first_cal.summary(), "years": self.first_cal_years},
            "finish_cqr": {**self.finish_cal.summary(), "years": self.finish_cal_years},
        }


# ------------------------------------------------------------------------ public API

_FORECASTERS: dict[tuple[int, date], CommencementForecaster] = {}
_fc_lock = threading.Lock()


def _quarter_start(day: date) -> date:
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def forecaster_for(data: CommencementData, as_of: date) -> CommencementForecaster:
    """Models fitted on data visible at the start of ``as_of``'s quarter (capped at the data
    end), cached per quarter, so a forecast never uses outcomes after ``as_of``."""
    cut = min(_quarter_start(as_of), data.data_end.date())
    key = (id(data), cut)
    with _fc_lock:
        if key not in _FORECASTERS:
            _FORECASTERS[key] = CommencementForecaster(data, cut)
        return _FORECASTERS[key]


def _with_dates(block: dict[str, Any], ra: date) -> dict[str, Any]:
    for k in ("raw", "calibrated"):
        if k in block:
            block[f"{k}_dates"] = {
                q: None if t is None else (pd.Timestamp(ra) + pd.Timedelta(7 * t, unit="D")).date()
                for q, t in block[k].items()
            }
    return block


def forecast_act(
    bill_id: int,
    as_of: date | str | pd.Timestamp | None = None,
    data: CommencementData | None = None,
    forecaster: CommencementForecaster | None = None,
) -> dict[str, Any]:
    """Stage-3 forecast for the Act of ``bill_id`` at ``as_of`` (default today), JSON-friendly.

    Keys: ``bill_id``, ``act_title``, ``royal_assent_date``, ``as_of``, ``model_cut`` (data the
    models were fitted on ends there), ``weeks_since_royal_assent``, ``orders_seen``,
    ``needs_orders`` (``p``: P(the Act needs commencement orders | none by as_of), or 1 once one
    is seen), ``first_order`` and ``orders_finished`` (each with ``raw`` and ``calibrated``
    ``t_10/t_50/t_90`` in weeks from Royal Assent, their ``*_dates``, ``is_calibrated``;
    ``orders_finished.p_finished_already`` = P(no further order)), ``definitions``.
    Raises KeyError for an unknown Act and ValueError if ``as_of`` is before Royal Assent."""
    data = data or load_data()
    as_of_ts = pd.Timestamp(as_of or date.today()).normalize()
    act = data.act(bill_id)
    ra = pd.Timestamp(act["royal_assent_date"]).date()
    if as_of_ts.date() < ra:
        raise ValueError(f"as_of {as_of_ts.date()} is before Royal Assent ({ra})")
    fc = forecaster or forecaster_for(data, as_of_ts.date())
    pred = fc.predict(act, _day(as_of_ts))
    return {
        "bill_id": int(bill_id),
        "act_title": None if pd.isna(act["act_title"]) else str(act["act_title"]),
        "department": act["department"],
        "royal_assent_date": ra,
        "as_of": as_of_ts.date(),
        "model_cut": fc.cut.date(),
        "weeks_since_royal_assent": pred["weeks_since_royal_assent"],
        "orders_seen": pred["orders_seen"],
        "needs_orders": pred["needs_orders"],
        "first_order": _with_dates(pred["first_order"], ra),
        "orders_finished": _with_dates(pred["orders_finished"], ra),
        "definitions": {
            "weeks": "calendar weeks from Royal Assent (week 1 = days 0-6)",
            "orders_finished": (
                f"the latest commencement order, followed by {FINISH_QUIET_WEEKS} weeks "
                "without another; not 'fully in force' (the data cannot show that)"
            ),
            "window": "80% window t_10..t_90; None = beyond the model horizon",
        },
    }


# ------------------------------------------------------------------------- backtest

BACKTEST_OFFSETS = (0, 13, 26, 52, 104)  # as_of = Royal Assent + these weeks
REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "journey"


def _outcomes(data: CommencementData, act: pd.Series, end: float) -> dict[str, Any]:
    """What is known about an Act at ``end``: ever needs orders (None if not yet known),
    first-order week, finished week (None if not finished by the FINISH_QUIET_WEEKS rule)."""
    ra = float(act["ra_day"])
    o = data.orders_until(act["bill_id"], end)
    by: dict[int, int | None] = {}
    if len(o):
        finished = o[-1] <= end - 7 * FINISH_QUIET_WEEKS
        T = _week(o[-1], ra) if finished else None
        for H in FINISH_BY_WEEKS:  # 0 once an order falls after week H; 1 once finished by H
            by[H] = 0 if _week(o[-1], ra) > H else (1 if finished else None)
        return {"ever": 1, "t1": _week(o[0], ra), "T": T, "by": by}
    ever = 0 if ra <= end - 7 * NO_ORDER_KNOWN_WEEKS else None
    return {"ever": ever, "t1": None, "T": None, "by": {H: None for H in FINISH_BY_WEEKS}}


def _dept(fc: CommencementForecaster, act: pd.Series) -> str:
    slug = department_slug(act["department"])
    return slug if slug in fc.static.departments else "other"


def _baseline_pool(fc: CommencementForecaster) -> pd.DataFrame:
    """Training Acts' outcomes as seen at the cut (for the baselines)."""
    rows = []
    for _, a in fc.data.acts[fc.data.acts["ra_day"] < fc.cut_day].iterrows():
        o = _outcomes(fc.data, a, fc.cut_day)
        by = o.pop("by")
        rows.append({"dept": _dept(fc, a), **o, **{f"by_{H}": v for H, v in by.items()}})
    return pd.DataFrame(rows)


def _cond_by(pool: pd.DataFrame, H: int, s: int, first_seen: bool) -> float | None:
    """Share of training Acts with orders whose latest order is by week H (label known),
    among those with no order in their first s weeks when none is seen yet."""
    p = pool[pool["ever"].eq(1) & pool[f"by_{H}"].notna()]
    if not first_seen:
        p = p[p["t1"] > s]
    return float(p[f"by_{H}"].mean()) if len(p) else None


def _cond_share(pool: pd.DataFrame, s: int) -> float | None:
    """Share eventually needing orders among Acts with no order in their first s weeks."""
    p = pool[pool["ever"].notna() & (pool["t1"].isna() | (pool["t1"] > s))]
    return float(p["ever"].mean()) if len(p) else None


def _cond_median(values: pd.Series, floor: int) -> float | None:
    v = values.dropna()
    v = v[v >= floor]
    return float(np.median(v)) if len(v) else None


def _covered(t: float, lo: Any, hi: Any, horizon: int) -> bool:
    lo = horizon + 1 if lo is None or pd.isna(lo) else lo
    hi = math.inf if hi is None or pd.isna(hi) else hi
    return bool(lo <= t <= hi)


def backtest(
    data: CommencementData,
    cut: Any = TRAIN_CUT,
    offsets: Sequence[int] = BACKTEST_OFFSETS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on Acts with Royal Assent before ``cut`` (data visible at ``cut``); test on later
    Acts at Royal Assent + ``offsets`` weeks (<= the data end), each forecast using only that
    Act's events up to its as_of. Outcomes are as seen at the data end. Returns the long table
    (one row per Act, as_of and target) and the metrics."""
    fc = CommencementForecaster(data, cut)
    end = _day(data.data_end)
    pool = _baseline_pool(fc)
    test = data.acts[data.acts["ra_day"] >= fc.cut_day]
    rows: list[dict[str, Any]] = []
    for _, a in test.iterrows():
        truth = _outcomes(data, a, end)
        dept = _dept(fc, a)
        dpool = pool[pool["dept"] == dept]
        for off in offsets:
            as_of = float(a["ra_day"]) + 7 * off
            if as_of > end:
                continue
            p = fc.predict(a, as_of)
            s = p["weeks_since_royal_assent"]
            base = {
                "act": int(a["bill_id"]),
                "act_title": a["act_title"],
                "department": dept,
                "royal_assent_date": a["royal_assent_date"].date(),
                "as_of": _date(as_of),
                "offset_weeks": off,
            }
            if not p["first_order"]["observed"]:
                if truth["ever"] is not None:
                    rows.append(
                        {
                            **base,
                            "target": "needs_orders",
                            "predicted": p["needs_orders"]["p"],
                            "baseline_overall": _cond_share(pool, s),
                            "baseline_department": _cond_share(dpool, s)
                            if len(dpool) >= 5
                            else _cond_share(pool, s),
                            "actual": truth["ever"],
                        }
                    )
                if truth["t1"] is not None:
                    fo = p["first_order"]
                    rows.append(
                        {
                            **base,
                            "target": "first_order_week",
                            "predicted": fo["raw"]["t_50"],
                            "raw_t_10": fo["raw"]["t_10"],
                            "raw_t_90": fo["raw"]["t_90"],
                            "cqr_t_10": fo["calibrated"]["t_10"],
                            "cqr_t_90": fo["calibrated"]["t_90"],
                            "baseline_overall": _cond_median(pool["t1"], s + 1),
                            "baseline_department": _cond_median(dpool["t1"], s + 1)
                            if dpool["t1"].notna().sum() >= 5
                            else _cond_median(pool["t1"], s + 1),
                            "actual": truth["t1"],
                        }
                    )
            for H in FINISH_BY_WEEKS:
                if truth["by"][H] is None or s >= H:
                    continue
                seen = p["first_order"]["observed"]
                rows.append(
                    {
                        **base,
                        "target": f"orders_finished_by_{H}w",
                        "predicted": p["orders_finished"]["p_by_week"][str(H)],
                        "baseline_overall": _cond_by(pool, H, s, seen),
                        "baseline_department": _cond_by(dpool, H, s, seen)
                        if len(dpool) >= 5
                        else _cond_by(pool, H, s, seen),
                        "actual": truth["by"][H],
                    }
                )
            if truth["T"] is not None:
                of = p["orders_finished"]
                floor = of["latest_order_week"] or s + 1
                rows.append(
                    {
                        **base,
                        "target": "orders_finished_week",
                        "predicted": of["raw"]["t_50"],
                        "raw_t_10": of["raw"]["t_10"],
                        "raw_t_90": of["raw"]["t_90"],
                        "cqr_t_10": of["calibrated"]["t_10"],
                        "cqr_t_90": of["calibrated"]["t_90"],
                        "baseline_overall": _cond_median(pool["T"], floor),
                        "baseline_department": _cond_median(dpool["T"], floor)
                        if dpool["T"].notna().sum() >= 5
                        else _cond_median(pool["T"], floor),
                        "actual": truth["T"],
                    }
                )
    table = pd.DataFrame(rows)
    return table, _metrics(table, fc, test)


def _binary(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    ok = ~pd.isna(p)
    y, p = y[ok].astype(float), p[ok].astype(float)
    n = len(y)
    return {
        "n": n,
        "auc": float(roc_auc_score(y, p)) if n and 0 < y.sum() < n else None,
        "brier": float(np.mean((p - y) ** 2)) if n else None,
    }


def _mae(actual: np.ndarray, pred: Sequence[Any], horizon: int) -> float | None:
    pv = np.array([horizon + 1 if v is None or pd.isna(v) else float(v) for v in pred])
    return float(np.mean(np.abs(pv - actual))) if len(pv) else None


def _metrics(table: pd.DataFrame, fc: CommencementForecaster, test: pd.DataFrame) -> dict:
    out: dict[str, Any] = {
        "train_cut": str(fc.cut.date()),
        "data_end": str(fc.data.data_end.date()),
        "n_test_acts": len(test),
        "model": fc.summary(),
        "definitions": {
            "needs_orders": (
                "1 = at least one commencement order by the data end; 0 = none and Royal "
                f"Assent >= {NO_ORDER_KNOWN_WEEKS} weeks before the data end; else not scored"
            ),
            "orders_finished_week": (
                f"week of the latest order, scored only when {FINISH_QUIET_WEEKS} weeks with no"
                " further order followed before the data end (right-truncated: slow Acts in"
                " recent years are under-represented)"
            ),
            "orders_finished_by_Hw": (
                "1 = the latest order is in weeks 1..H and was followed by "
                f"{FINISH_QUIET_WEEKS} quiet weeks; 0 = an order came after week H; Acts with "
                "orders only (free of the right-truncation of orders_finished_week)"
            ),
            "baselines": "conditional on what is known at as_of (no order by then / T >= floor)",
        },
        "by_offset": {},
    }
    if table.empty:
        return out
    for off, g in table.groupby("offset_weeks"):
        block: dict[str, Any] = {}
        for target in ("needs_orders", *(f"orders_finished_by_{H}w" for H in FINISH_BY_WEEKS)):
            n = g[g["target"] == target]
            if not len(n):
                continue
            n = n[n["baseline_overall"].notna() & n["baseline_department"].notna()]
            if not len(n):
                continue
            y = n["actual"].to_numpy(float)
            block[target] = {
                "share_positive": float(y.mean()),
                "model": _binary(y, n["predicted"].to_numpy(float)),
                "baseline_overall": _binary(y, n["baseline_overall"].to_numpy(float)),
                "baseline_department": _binary(y, n["baseline_department"].to_numpy(float)),
            }
        for target, horizon in (
            ("first_order_week", FIRST_HORIZON),
            ("orders_finished_week", FINISH_HORIZON),
        ):
            t = g[g["target"] == target]
            if not len(t):
                continue
            act = t["actual"].to_numpy(float)
            block[target] = {
                "n": len(t),
                "median_actual": float(np.median(act)),
                "mae_model_p50": _mae(act, t["predicted"], horizon),
                "mae_baseline_overall": _mae(act, t["baseline_overall"], horizon),
                "mae_baseline_department": _mae(act, t["baseline_department"], horizon),
                "coverage_80_raw": float(
                    np.mean(
                        [
                            _covered(x, lo, hi, horizon)
                            for x, lo, hi in zip(act, t["raw_t_10"], t["raw_t_90"], strict=True)
                        ]
                    )
                ),
                "coverage_80_cqr": float(
                    np.mean(
                        [
                            _covered(x, lo, hi, horizon)
                            for x, lo, hi in zip(act, t["cqr_t_10"], t["cqr_t_90"], strict=True)
                        ]
                    )
                ),
            }
            if off == 0:
                by = {}
                for dept, gd in t.groupby("department"):
                    a2 = gd["actual"].to_numpy(float)
                    by[dept] = {
                        "n": len(gd),
                        "mae_model_p50": _mae(a2, gd["predicted"], horizon),
                        "mae_baseline_department": _mae(a2, gd["baseline_department"], horizon),
                    }
                block[target]["by_department"] = by
        out["by_offset"][str(int(off))] = block
    return out


def _json_default(x: Any) -> Any:
    if isinstance(x, (date, pd.Timestamp)):
        return str(x)[:10]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    raise TypeError(type(x))


def write_backtest(
    data: CommencementData | None = None, out_dir: Path = REPORT_DIR
) -> dict[str, Any]:
    """Run ``backtest`` and write ``stage3_backtest.csv`` and ``stage3_metrics.json``."""
    import json

    data = data or load_data()
    table, metrics = backtest(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "stage3_backtest.csv", index=False)
    (out_dir / "stage3_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default) + "\n"
    )
    return metrics


if __name__ == "__main__":  # uv run python -m westminster_weather.journey.commencement
    logging.basicConfig(level=logging.INFO)
    m = write_backtest()
    print(f"wrote {REPORT_DIR}/stage3_backtest.csv and stage3_metrics.json")
