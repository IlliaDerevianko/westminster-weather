"""FastAPI endpoints for the take-up model (Phase 1 -> 2; TAKEUP_PLAN task 10).

- ``GET /api/v1/takeup/ideas?as_of=``: every signal visible at ``as_of`` with its forecast.
- ``GET /api/v1/takeup/forecast?idea_id=&as_of=``: one idea's forecast, events and mentions.
- ``GET /api/v1/takeup/history?idea_id=&as_of=``: weekly ``pi_now`` / ``p52`` from the trigger.

Everything comes from ``TakeupData.get_takeup_signals(as_of)`` and the take-up tables filtered
to ``<= as_of`` (DATA_CONTRACT §10.9): nothing dated after ``as_of`` reaches a response.

Two models, chosen per request with ``?model=`` (default ``DEFAULT_MODEL``):

- ``hazard`` (default; ``takeup.hazard.HazardModel``): fitted once in memory on every idea
  visible at the last sitting day on record (``evaluate.default_end``). Its 80% timing window
  [t_10, t_90] is CQR-calibrated (``takeup.conformal``) for each request's ``as_of`` from
  ideas with a bill triggered in a held-out window ``[cut - years, cut)`` (cut = the start of
  ``as_of``'s quarter, capped at the fit end; years = 2, widened to 3 or 4 until there are
  ``MIN_CALIBRATION`` bills), predicted at their triggers by a hazard model fitted only on
  data visible at ``cut - years``. With fewer bills the raw window is served and
  ``timing_calibrated`` is false. ``/history`` re-predicts week by week with only the
  mentions and events dated on or before each week (a merged idea uses its anchor topic
  group's mentions, as it looked before the merge).
- ``cure`` (``takeup.model.TakeUpModel``): rebuilt from the latest ``takeup_runs`` row
  (``parameters`` JSON); without one it is fitted once (``evaluate.fit_all``) and kept in
  memory. Stored ``takeup_forecasts`` rows of that run for (signal, as_of) are preferred over
  computed values. Its windows are raw.

Both models' parameters use data up to their fit date, not ``as_of``. Nothing is written to
any database.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import date, timedelta
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from westminster_weather.takeup import model as tm
from westminster_weather.takeup.conformal import MIN_CALIBRATION, CQRCalibrator
from westminster_weather.takeup.hazard import HazardModel
from westminster_weather.takeup.signals import SOURCE_RANK, TakeupData

logger = logging.getLogger(__name__)

TABLE_TTL_SECONDS = 600  # reload take-up and reference tables after this long
AS_OF_CACHE_SIZE = 16  # prepared signals kept per service, by (as_of, model)
DEFAULT_MODEL = "hazard"  # served unless ?model=cure
CALIBRATION_YEARS_TRY = (2, 3, 4)  # held-out calibration windows, widened until enough bills
ModelName = Literal["hazard", "cure"]
STORED_KEYS = (
    "pi",
    "pi_lo",
    "pi_hi",
    "verdict",
    "weeks_elapsed",
    "pi_now",
    "p26",
    "p52",
    "t_10",
    "t_50",
    "t_90",
)

router = APIRouter(prefix="/api/v1/takeup", tags=["takeup"])

# ----------------------------------------------------------------------------- schemas


class LinkedBillSchema(BaseModel):
    bill_id: int
    title: str | None = None
    first_reading_date: date


class IdeaSummarySchema(BaseModel):
    signal_id: str
    idea_id: str
    topic: str | None = None
    trigger_date: date
    first_source: str | None = None
    taken_up: bool
    verdict: str
    pi: float
    pi_now: float
    p26: float
    p52: float
    weeks_elapsed: int
    t_50: int | None = None
    t_50_date: date | None = None
    linked_bill: LinkedBillSchema | None = None
    forecast_source: str  # 'stored' (takeup_forecasts) or 'computed'
    model: str  # 'hazard' or 'cure'


class IdeasResponse(BaseModel):
    as_of: date
    model: str
    run_id: int | None = None
    count: int
    ideas: list[IdeaSummarySchema]


class IdeaEventSchema(BaseModel):
    event_type: str
    event_date: date
    title: str | None = None
    source_url: str | None = None


class MentionSchema(BaseModel):
    mention_id: str
    group_id: str | None = None
    date: date
    source: str
    firmness: int | None = None
    sentence: str | None = None
    title: str | None = None


class ForecastResponse(BaseModel):
    as_of: date
    model: str
    run_id: int | None = None
    idea_id: str
    signal_id: str
    signal_ids: list[str]  # all signals of the idea at as_of (split groups before a merge)
    topic: str | None = None
    trigger_date: date
    first_source: str | None = None
    taken_up: bool
    pi: float
    pi_lo: float
    pi_hi: float
    verdict: str
    weeks_elapsed: int
    pi_now: float
    p26: float
    p52: float
    t_10: int | None = None
    t_50: int | None = None
    t_90: int | None = None
    t_10_date: date | None = None
    t_50_date: date | None = None
    t_90_date: date | None = None
    t_10_sitting: int | None = None
    t_50_sitting: int | None = None
    t_90_sitting: int | None = None
    # the model's own 80% window before CQR calibration (equal to t_10 / t_90 if uncalibrated)
    t_10_raw: int | None = None
    t_90_raw: int | None = None
    timing_calibrated: bool = False
    n_calibration: int | None = None  # bills the CQR adjustment was estimated from
    events: list[IdeaEventSchema]
    mentions: list[MentionSchema]
    linked_bill: LinkedBillSchema | None = None
    forecast_source: str


class HistoryPointSchema(BaseModel):
    week: int
    date: date
    pi_now: float
    p52: float


class HistoryMarkerSchema(BaseModel):
    week: int
    date: date
    kind: str  # 'mention', an idea_events event_type, or 'first_reading'
    label: str | None = None


class HistoryResponse(BaseModel):
    as_of: date
    model: str
    run_id: int | None = None
    idea_id: str
    signal_id: str
    trigger_date: date
    taken_up: bool
    points: list[HistoryPointSchema]
    markers: list[HistoryMarkerSchema]


# ----------------------------------------------------------------------------- helpers


def _d(value: Any) -> date | None:
    """A date, or None for missing values."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return pd.Timestamp(value).date()


def _int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _week_date(trigger: date, week: int | None) -> date | None:
    """End of calendar week ``week`` after the trigger (week t covers days 7(t-1)..7t)."""
    return None if week is None else trigger + timedelta(weeks=int(week))


def _flag(values: pd.Series) -> pd.Series:
    return values.astype("boolean").fillna(False).astype(bool)


def model_from_run(row: dict[str, Any]) -> tm.TakeUpModel:
    """Rebuild a ``TakeUpModel`` from a ``takeup_runs`` row (``features``, ``parameters``).

    ``parameters`` holds estimates and standard errors only, so the covariance is taken as
    diagonal: the pi CI ignores correlations between coefficients.
    """
    feats = [f for f in str(row["features"]).split(",") if f]
    zcols = [f for f in feats if f.startswith("z_")]
    xcols = [f for f in feats if f.startswith("x_")]
    params = row["parameters"]
    params = json.loads(params) if isinstance(params, str) else params
    names = ["intercept", *zcols, *(f"alpha_weeks_{e}+" for e in tm.BIN_EDGES), *xcols]
    theta = [float(params[n]["estimate"]) for n in names]
    se = [params[n].get("std_err") for n in names]
    sigma = None
    if all(s is not None and np.isfinite(float(s)) for s in se):
        sigma = np.diag(np.square(np.asarray(se, dtype=float)))
    return tm.TakeUpModel.from_params(theta, zcols, xcols, Sigma=sigma)


def _group(row: pd.Series) -> str | None:
    """The topic group of a split signal ('<idea_id>:<group_id>'); None for a merged one."""
    sid, idea = str(row["signal_id"]), str(row["idea_id"])
    return sid.removeprefix(f"{idea}:") if sid != idea else None


StoredLookup = Callable[[list[str], date], dict[str, dict[str, Any]]]
CalibratorFn = Callable[[date], CQRCalibrator | None]
_EMPTY_TIMELINE = pd.DataFrame(columns=["signal_id", "event_date", "firmness"])


def _quarter_start(day: date) -> date:
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def hazard_calibrator(
    data: Any, cut: Any, years_try: tuple[int, ...] = CALIBRATION_YEARS_TRY
) -> CQRCalibrator | None:
    """CQR calibrator for the hazard model's 80% window at ``cut``, using only data visible
    at ``cut``: ideas with a bill triggered in ``[cut - years, cut)`` predicted at their
    triggers by a hazard model fitted on ``get_takeup_signals(cut - years)``. ``years`` is
    widened through ``years_try`` until there are ``MIN_CALIBRATION`` bills; None if never."""
    from westminster_weather.takeup import evaluate as ev

    cut = pd.Timestamp(cut).normalize()
    for years in years_try:
        cal_cut = cut - pd.DateOffset(years=years)
        train = data.get_takeup_signals(cal_cut)
        train = train[train["signal_date"] < cal_cut].reset_index(drop=True)
        if train.empty:
            continue
        rows = data.get_takeup_signals(cut, features=ev.feature_columns(train))
        cal = rows[(rows["signal_date"] >= cal_cut) & (rows["signal_date"] < cut)]
        cal = cal.reset_index(drop=True)
        if cal.empty:
            continue
        sit = data.sitting_dates(cut)
        d_cal = tm.prepare(cal, sit, data.week_probability(cut), data_end=cut)
        bills = d_cal["delta"] == 1
        if bills.sum() < MIN_CALIBRATION:
            continue
        wp = data.week_probability(cal_cut)
        try:
            hz = ev.fit_hazard(data, train, cal_cut, wp)
        except ValueError as exc:
            logger.warning("hazard calibration fit at %s failed: %s", cal_cut.date(), exc)
            continue
        hp = hz.predict_frame(cal, ev._timeline(data, cut, cal), sit, wp)
        return CQRCalibrator().fit(d_cal["t"][bills], hp["t_10"][bills], hp["t_90"][bills])
    return None


def cached_calibrator(data: Any, fit_end: date | None) -> CalibratorFn:
    """``hazard_calibrator`` at the start of ``as_of``'s quarter (<= as_of, so as_of-safe),
    capped at ``fit_end``; one per quarter, cached for the process."""
    cache: dict[date, CQRCalibrator | None] = {}
    lock = threading.Lock()

    def get(as_of: date) -> CQRCalibrator | None:
        cut = _quarter_start(as_of)
        if fit_end is not None and cut > fit_end:
            cut = fit_end
        with lock:
            if cut not in cache:
                try:
                    cache[cut] = hazard_calibrator(data, cut)
                except Exception:
                    logger.exception("hazard CQR calibration at %s failed; raw windows", cut)
                    cache[cut] = None
            return cache[cut]

    return get


def calibrated_window(
    pred: dict[str, Any],
    cal: CQRCalibrator | None,
    mask: Callable[[], np.ndarray],
) -> dict[str, Any]:
    """``pred`` with t_10 / t_90 widened by ``cal`` (kept > weeks_elapsed and around t_50),
    their sitting-week counts recomputed from ``mask()``; raw values kept as ``*_raw``."""
    out = dict(pred)
    out["t_10_raw"], out["t_90_raw"] = pred.get("t_10"), pred.get("t_90")
    usable = cal is not None and cal.n >= MIN_CALIBRATION
    out["timing_calibrated"] = bool(usable)
    out["n_calibration"] = None if cal is None else int(cal.n)
    if not usable or pred.get("t_50") is None:
        out["timing_calibrated"] = False
        return out
    (lo,), (hi,) = cal.apply([pred.get("t_10")], [pred.get("t_90")], int(pred["weeks_elapsed"]))
    t50 = int(pred["t_50"])
    lo = t50 if lo is None else min(lo, t50)
    hi = None if hi is None else max(hi, t50)
    out["t_10"], out["t_90"] = lo, hi
    sitting = np.cumsum(mask())
    for key, t in (("t_10", lo), ("t_90", hi)):
        out[f"{key}_sitting"] = None if t is None else int(round(sitting[t - 1]))
    return out


class TakeupService:
    """Data, models and caches behind the endpoints (injectable for tests).

    ``model``: the cure model; ``hazard``: the hazard model (None = cure only);
    ``calibrator``: as_of -> CQR calibrator for the hazard windows (None = raw windows)."""

    def __init__(
        self,
        data: TakeupData,
        model: tm.TakeUpModel,
        run_id: int | None = None,
        stored_lookup: StoredLookup | None = None,
        hazard: HazardModel | None = None,
        calibrator: CalibratorFn | None = None,
    ) -> None:
        self.data = data
        self.model = model
        self.run_id = run_id
        self.stored_lookup = stored_lookup
        self.hazard = hazard
        self.calibrator = calibrator
        feats = [*model.zcols, *model.xcols]
        if hazard is not None:
            feats += [*hazard.zcols, *hazard.xflags]
        self.features = list(dict.fromkeys(feats))
        self._cache: OrderedDict[tuple[date, str], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def model_name(self, requested: str | None) -> str:
        """The model actually served: the request's (default DEFAULT_MODEL); cure if there
        is no hazard model."""
        name = requested or DEFAULT_MODEL
        return "cure" if name == "hazard" and self.hazard is None else name

    def run_id_for(self, model: str) -> int | None:
        """``takeup_runs`` id behind the forecasts (the hazard model is not stored: None)."""
        return self.run_id if model == "cure" else None

    # -------------------------------------------------------------- per-as_of state

    def state(self, as_of: date, model: str = "cure") -> dict[str, Any]:
        """Signals, prepared arrays and forecasts at ``as_of`` (LRU-cached)."""
        key = (as_of, model)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        st = self._build(as_of, model)
        with self._lock:
            self._cache[key] = st
            while len(self._cache) > AS_OF_CACHE_SIZE:
                self._cache.popitem(last=False)
        return st

    def _cure_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """``df`` with only the cure model's z / x columns (``prepare`` uses all of them)."""
        base = [c for c in df.columns if not c.startswith(("z_", "x_"))]
        return df[base + [c for c in [*self.model.zcols, *self.model.xcols] if c in df]]

    def _build(self, as_of: date, model: str) -> dict[str, Any]:
        ts = pd.Timestamp(as_of)
        df = self.data.get_takeup_signals(ts, features=self.features).reset_index(drop=True)
        st: dict[str, Any] = {
            "df": df,
            "df_as_of": as_of,
            "d": None,
            "preds": [],
            "stored": {},
            "model": model,
        }
        if df.empty:
            return st
        wp = self.data.week_probability(ts)
        sit = self.data.sitting_dates(ts)
        d = tm.prepare(self._cure_frame(df), sit, wp, data_end=ts, calendar_end=ts)
        st["d"] = d
        st["sit"], st["wp"] = sit, wp
        if model == "hazard":
            st["timeline"] = self._timelines(self.data.mention_timeline(ts))
            cal = self.calibrator(as_of) if self.calibrator is not None else None
            st["preds"] = [
                self._hazard_pred(sig, st, ts, bool(d["delta"][k] == 1), cal)
                for k, sig in enumerate(df.to_dict("records"))
            ]
        else:
            st["preds"] = [
                calibrated_window(
                    self.model.predict(**tm.row_inputs(d, k)), None, lambda: np.ones(0)
                )
                for k in range(len(df))
            ]
            if self.stored_lookup is not None:
                try:
                    st["stored"] = self.stored_lookup(df["signal_id"].astype(str).tolist(), as_of)
                except Exception:  # stored forecasts are optional; computed ones still work
                    logger.warning("takeup_forecasts lookup failed; using computed forecasts")
        st["first_source"] = self._first_sources(ts)
        return st

    @staticmethod
    def _timelines(tl: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if tl.empty:
            return {}
        tl = tl.assign(signal_id=tl["signal_id"].astype(str))
        return {str(sid): g for sid, g in tl.groupby("signal_id", sort=False)}

    def _hazard_pred(
        self,
        sig: dict[str, Any],
        st: dict[str, Any],
        as_of: pd.Timestamp,
        taken_up: bool,
        cal: CQRCalibrator | None,
    ) -> dict[str, Any]:
        """Hazard forecast for one signal row: at ``as_of`` if still silent, else (as the cure
        model does) from its signal date, with the bill hidden."""
        at = pd.Timestamp(sig["signal_date"]).normalize() if taken_up else as_of
        sid = str(sig["signal_id"])
        tl = st["timeline"].get(sid, _EMPTY_TIMELINE)
        pred = self.hazard.predict(sig, tl, at, st["sit"], st["wp"])

        def mask() -> np.ndarray:
            return tm.sitting_mask(
                [sig["signal_date"]],
                [pred["weeks_elapsed"]],
                st["sit"],
                st["wp"],
                calendar_end=at,
            )[0]

        return calibrated_window(pred, cal, mask)

    def _first_sources(self, as_of: pd.Timestamp) -> dict[str, str]:
        """signal key ('<idea_id>' and '<idea_id>:<group_id>') -> source of its first mention."""
        m = self.data.visible_mentions(as_of)
        if m.empty:
            return {}
        m = m.assign(rank=m["source"].map(SOURCE_RANK).fillna(len(SOURCE_RANK)))
        m = m.sort_values(["event_date", "rank", "mention_id"])
        out: dict[str, str] = {}
        for idea, src in m.drop_duplicates("idea_id")[["idea_id", "source"]].itertuples(False):
            out[str(idea)] = str(src)
        for idea, grp, src in m.drop_duplicates(["idea_id", "group_id"])[
            ["idea_id", "group_id", "source"]
        ].itertuples(False):
            out[f"{idea}:{grp}"] = str(src)
        return out

    # ---------------------------------------------------------------------- rows

    def forecast_row(self, st: dict[str, Any], k: int) -> tuple[dict[str, Any], str]:
        """Forecast for row ``k``: stored values of this run if present, else computed."""
        pred = dict(st["preds"][k])
        sid = str(st["df"].at[k, "signal_id"])
        stored = st["stored"].get(sid)
        if not stored:
            return pred, "computed"
        for key in STORED_KEYS:
            if key in stored:
                pred[key] = stored[key]
        if any(
            pred.get(f"{tm.quantile_key(q)}") != st["preds"][k].get(tm.quantile_key(q))
            for q in tm.QUANTILES
        ):  # sitting versions no longer match: drop them
            for q in tm.QUANTILES:
                pred[f"{tm.quantile_key(q)}_sitting"] = None
        return pred, "stored"

    def history_points(self, st: dict[str, Any], k: int) -> list[tuple[int, float, float]]:
        """(week, pi_now, p52) for weeks 0..last silent week of row ``k``.

        Cure: prepared once at ``as_of``; week w masks ``x`` events after week w (same ``z``
        and sitting mask). Hazard: re-predicted at each week's date (trigger + 7w; the last
        point of a still-silent idea at ``as_of`` itself) with only the mentions and events
        dated on or before it and the sitting probabilities known then. A merged idea (bill
        first read) uses its anchor topic group's mentions as they were before the merge, so
        the history never shows the merge."""
        d, row = st["d"], st["df"].iloc[k]
        taken_up = bool(d["delta"][k] == 1)
        last = int(d["t"][k]) - 1 if taken_up else int(d["t"][k])  # silent weeks only
        weeks = range(0, max(last, 0) + 1)
        if st["model"] != "hazard":
            z, xweek, mask = d["Z"][k], d["xweek"][k], d["M"][k]
            out = []
            for w in weeks:
                xw = np.where(xweek <= w, xweek, tm.T_MAX + 1)
                p = self.model.predict(z, xw, weeks_elapsed=w, mask=mask)
                out.append((w, float(p["pi_now"]), float(p["p52"])))
            return out
        sig = row.to_dict()
        trigger = pd.Timestamp(sig["signal_date"]).normalize()
        as_of = pd.Timestamp(st["df_as_of"]).normalize()
        tl = self._history_timeline(sig, taken_up, st)
        wp_by_year: dict[int, pd.Series] = {}  # week_probability depends on as_of's year only
        out = []
        for w in weeks:
            at = min(trigger + pd.Timedelta(7 * w, unit="D"), as_of)
            if w == last and not taken_up:
                at = as_of
            if at.year not in wp_by_year:
                wp_by_year[at.year] = self.data.week_probability(at)
            p = self.hazard.predict(sig, tl, at, st["sit"], wp_by_year[at.year])
            out.append((w, float(p["pi_now"]), float(p["p52"])))
        return out

    def _history_timeline(
        self, sig: dict[str, Any], taken_up: bool, st: dict[str, Any]
    ) -> pd.DataFrame:
        sid, idea = str(sig["signal_id"]), str(sig["idea_id"])
        own = st["timeline"].get(sid, _EMPTY_TIMELINE)
        if not (taken_up and sid == idea and pd.notna(sig.get("bill_date"))):
            return own
        before = pd.Timestamp(sig["bill_date"]).normalize() - pd.Timedelta(1, unit="D")
        pre = self.data.mention_timeline(before)
        if pre.empty:
            return own
        pre = pre[pre["signal_id"].astype(str).str.startswith(f"{idea}:")]
        if pre.empty:
            return own
        first = pre.sort_values(["event_date", "signal_id"]).iloc[0]["signal_id"]
        return pre[pre["signal_id"] == first]

    def linked_bill(self, row: pd.Series) -> LinkedBillSchema | None:
        if pd.isna(row["bill_id"]) or pd.isna(row["bill_date"]):
            return None
        bill_id = int(row["bill_id"])
        bills = self.data.tables["bills"]
        title = None
        for col in ("short_title", "title"):
            if col in bills:
                hit = bills.loc[bills["bill_id"] == bill_id, col]
                if len(hit) and pd.notna(hit.iloc[0]):
                    title = str(hit.iloc[0])
                    break
        return LinkedBillSchema(
            bill_id=bill_id, title=title, first_reading_date=pd.Timestamp(row["bill_date"]).date()
        )

    def summary(self, st: dict[str, Any], k: int) -> IdeaSummarySchema:
        row = st["df"].iloc[k]
        pred, source = self.forecast_row(st, k)
        trigger = pd.Timestamp(row["signal_date"]).date()
        sid = str(row["signal_id"])
        return IdeaSummarySchema(
            signal_id=sid,
            idea_id=str(row["idea_id"]),
            topic=None if pd.isna(row["title"]) else str(row["title"]),
            trigger_date=trigger,
            first_source=st.get("first_source", {}).get(sid),
            taken_up=bool(st["d"]["delta"][k] == 1),
            verdict=str(pred["verdict"]),
            pi=float(pred["pi"]),
            pi_now=float(pred["pi_now"]),
            p26=float(pred["p26"]),
            p52=float(pred["p52"]),
            weeks_elapsed=int(pred["weeks_elapsed"]),
            t_50=_int(pred.get("t_50")),
            t_50_date=_week_date(trigger, _int(pred.get("t_50"))),
            linked_bill=self.linked_bill(row),
            forecast_source=source,
            model=st["model"],
        )

    def resolve(self, st: dict[str, Any], idea_id: str) -> tuple[int, list[int]]:
        """(primary row, all rows) for an idea_id or signal_id at as_of; 404 if none.

        Before a merge an idea has one row per topic group; the primary row is the merged
        row, else the earliest-triggered group (the idea's anchor group)."""
        df = st["df"]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Idea '{idea_id}' not visible")
        exact = df.index[df["signal_id"].astype(str) == idea_id].tolist()
        if exact:
            return exact[0], exact
        rows = df.index[df["idea_id"].astype(str) == idea_id].tolist()
        if not rows:
            raise HTTPException(
                status_code=404, detail=f"Idea '{idea_id}' not found or not visible at as_of"
            )
        primary = min(rows, key=lambda k: (df.at[k, "signal_date"], str(df.at[k, "signal_id"])))
        return primary, rows

    def events(self, idea_id: str, as_of: date) -> list[IdeaEventSchema]:
        ev = self.data.tables["idea_events"]
        ev = ev[ev["idea_id"].astype(str) == idea_id]
        dates = pd.to_datetime(ev["event_date"], errors="coerce")
        ev = ev.assign(_d=dates)[dates <= pd.Timestamp(as_of)].sort_values("_d")
        return [
            IdeaEventSchema(
                event_type=str(r["event_type"]),
                event_date=r["_d"].date(),
                title=None if pd.isna(r.get("title")) else str(r.get("title")),
                source_url=None if pd.isna(r.get("source_url")) else str(r.get("source_url")),
            )
            for r in ev.to_dict("records")
        ]

    def mentions(
        self, idea_id: str, as_of: date, group_id: str | None = None
    ) -> list[MentionSchema]:
        """Visible mentions of the idea; of one topic group only if ``group_id`` is given
        (a split signal must not reveal the later merge)."""
        im = self.data.tables["idea_mentions"]
        im = im[im["idea_id"].astype(str) == idea_id][["mention_id", "group_id"]]
        if group_id is not None:
            im = im[im["group_id"].astype(str) == group_id]
        sm = self.data.tables["signal_mentions"]
        sm = sm.assign(_d=pd.to_datetime(sm["event_date"], errors="coerce"))
        visible = sm["_d"] <= pd.Timestamp(as_of)
        sm = sm[_flag(sm["is_commitment"]) & _flag(sm["is_minister"]) & visible]
        im = im.assign(mention_id=im["mention_id"].astype(str))
        sm = sm.assign(mention_id=sm["mention_id"].astype(str))
        m = im.merge(sm, on="mention_id", how="inner").sort_values(["_d", "mention_id"])
        return [
            MentionSchema(
                mention_id=str(r["mention_id"]),
                group_id=None if pd.isna(r["group_id"]) else str(r["group_id"]),
                date=r["_d"].date(),
                source=str(r["source"]),
                firmness=_int(r.get("firmness")),
                sentence=None if pd.isna(r.get("sentence")) else str(r.get("sentence")),
                title=None if pd.isna(r.get("title")) else str(r.get("title")),
            )
            for r in m.to_dict("records")
        ]


# ----------------------------------------------------------------- default service


def _latest_run(engine: Any) -> dict[str, Any] | None:
    from sqlalchemy import select

    from westminster_weather.takeup.tables import takeup_runs

    with engine.connect() as conn:
        row = (
            conn.execute(select(takeup_runs).order_by(takeup_runs.c.run_id.desc()).limit(1))
            .mappings()
            .first()
        )
    return dict(row) if row else None


def _stored_lookup(engine: Any, run_id: int) -> StoredLookup:
    def lookup(signal_ids: list[str], as_of: date) -> dict[str, dict[str, Any]]:
        from sqlalchemy import select

        from westminster_weather.takeup.tables import takeup_forecasts as tf

        if not signal_ids:
            return {}
        q = select(tf).where(tf.c.run_id == run_id, tf.c.as_of == as_of)
        if len(signal_ids) <= 500:
            q = q.where(tf.c.signal_id.in_(signal_ids))
        with engine.connect() as conn:
            rows = conn.execute(q).mappings().all()
        return {str(r["signal_id"]): dict(r) for r in rows}

    return lookup


_default: dict[str, Any] = {
    "service": None,
    "loaded_at": 0.0,
    "fitted": None,
    "hazard": None,
    "calibrator": None,
}
_default_lock = threading.Lock()


def _default_hazard(data: TakeupData) -> tuple[HazardModel | None, CalibratorFn | None]:
    """The hazard model fitted in memory on every idea visible at ``default_end`` (once per
    process; not saved) and its per-quarter CQR calibrator; (None, None) if the fit fails."""
    if _default["hazard"] is None:
        from westminster_weather.takeup import evaluate

        end = evaluate.default_end(data)
        train = data.get_takeup_signals(end).reset_index(drop=True)
        try:
            logger.info("fitting the take-up hazard model in memory (not saved)")
            _default["hazard"] = evaluate.fit_hazard(data, train, end)
        except ValueError as exc:
            logger.warning("hazard model fit failed (%s); serving the cure model only", exc)
            return None, None
        _default["calibrator"] = cached_calibrator(data, end.date())
    return _default["hazard"], _default["calibrator"]


def _load_default_service() -> TakeupService:
    from westminster_weather.takeup import db

    data = TakeupData()
    _ = data.tables  # load now (take-up DB + production reference tables, read only)
    hazard, calibrator = _default_hazard(data)
    engine = db.get_takeup_engine()
    run = None
    try:
        run = _latest_run(engine)
    except Exception:
        logger.warning("could not read takeup_runs; falling back to an in-memory fit")
    if run is not None:
        model = model_from_run(run)
        run_id = int(run["run_id"])
        return TakeupService(
            data, model, run_id, _stored_lookup(engine, run_id), hazard, calibrator
        )
    if _default["fitted"] is None:
        from westminster_weather.takeup import evaluate

        logger.warning("no takeup_runs row: fitting the take-up model in memory (not saved)")
        _default["fitted"] = evaluate.fit_all(data, evaluate.default_end(data))["model"]
    return TakeupService(data, _default["fitted"], None, None, hazard, calibrator)


def get_takeup_service() -> TakeupService:
    """The process-wide service; tables and the model are reloaded every TABLE_TTL_SECONDS."""
    with _default_lock:
        stale = time.monotonic() - _default["loaded_at"] > TABLE_TTL_SECONDS
        if _default["service"] is None or stale:
            try:
                _default["service"] = _load_default_service()
            except Exception as exc:
                from westminster_weather.takeup.db import TakeupDatabaseNotConfigured

                if isinstance(exc, TakeupDatabaseNotConfigured):
                    raise HTTPException(
                        status_code=503, detail="Take-up database is not configured"
                    ) from exc
                raise
            _default["loaded_at"] = time.monotonic()
        return _default["service"]


Service = Annotated[TakeupService, Depends(get_takeup_service)]
AsOf = Annotated[date | None, Query(description="Date (YYYY-MM-DD) to evaluate at; default today.")]
IdeaId = Annotated[str, Query(description="idea_id, or a signal_id '<idea_id>:<group_id>'")]
Model = Annotated[
    ModelName | None,
    Query(description="'hazard' (default: recency hazard, CQR windows) or 'cure'."),
]


# ----------------------------------------------------------------------- endpoints


@router.get("/ideas", response_model=IdeasResponse)
def list_ideas(service: Service, as_of: AsOf = None, model: Model = None) -> IdeasResponse:
    """Every idea (signal) visible at ``as_of``, with its take-up forecast."""
    as_of = as_of or date.today()
    name = service.model_name(model)
    st = service.state(as_of, name)
    ideas = [service.summary(st, k) for k in range(len(st["df"]))]
    return IdeasResponse(
        as_of=as_of,
        model=name,
        run_id=service.run_id_for(name),
        count=len(ideas),
        ideas=ideas,
    )


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    service: Service, idea_id: IdeaId, as_of: AsOf = None, model: Model = None
) -> ForecastResponse:
    """One idea's forecast at ``as_of``, with its events and mentions dated ``<= as_of``."""
    as_of = as_of or date.today()
    name = service.model_name(model)
    st = service.state(as_of, name)
    k, rows = service.resolve(st, idea_id)
    row = st["df"].iloc[k]
    pred, source = service.forecast_row(st, k)
    trigger = pd.Timestamp(row["signal_date"]).date()
    real_id = str(row["idea_id"])
    timing: dict[str, Any] = {}
    for q in tm.QUANTILES:
        key = tm.quantile_key(q)
        t = _int(pred.get(key))
        timing[key] = t
        timing[f"{key}_date"] = _week_date(trigger, t)
        timing[f"{key}_sitting"] = _int(pred.get(f"{key}_sitting"))
    return ForecastResponse(
        as_of=as_of,
        model=name,
        run_id=service.run_id_for(name),
        idea_id=real_id,
        signal_id=str(row["signal_id"]),
        signal_ids=[str(st["df"].at[r, "signal_id"]) for r in rows],
        topic=None if pd.isna(row["title"]) else str(row["title"]),
        trigger_date=trigger,
        first_source=st.get("first_source", {}).get(str(row["signal_id"])),
        taken_up=bool(st["d"]["delta"][k] == 1),
        pi=float(pred["pi"]),
        pi_lo=float(pred["pi_lo"]),
        pi_hi=float(pred["pi_hi"]),
        verdict=str(pred["verdict"]),
        weeks_elapsed=int(pred["weeks_elapsed"]),
        pi_now=float(pred["pi_now"]),
        p26=float(pred["p26"]),
        p52=float(pred["p52"]),
        **timing,
        t_10_raw=_int(pred.get("t_10_raw")),
        t_90_raw=_int(pred.get("t_90_raw")),
        timing_calibrated=bool(pred.get("timing_calibrated", False)),
        n_calibration=_int(pred.get("n_calibration")),
        events=service.events(real_id, as_of),
        mentions=service.mentions(real_id, as_of, _group(row)),
        linked_bill=service.linked_bill(row),
        forecast_source=source,
    )


@router.get("/history", response_model=HistoryResponse)
def get_history(
    service: Service, idea_id: IdeaId, as_of: AsOf = None, model: Model = None
) -> HistoryResponse:
    """Weekly ``pi_now`` and ``p52`` from the trigger to ``as_of`` (or to the bill's week),
    each point using only what was known at that week (``TakeupService.history_points``)."""
    as_of = as_of or date.today()
    name = service.model_name(model)
    st = service.state(as_of, name)
    k, _ = service.resolve(st, idea_id)
    d, row = st["d"], st["df"].iloc[k]
    trigger = pd.Timestamp(row["signal_date"]).date()
    taken_up = bool(d["delta"][k] == 1)
    points = [
        HistoryPointSchema(week=w, date=_week_date(trigger, w), pi_now=pi_now, p52=p52)
        for w, pi_now, p52 in service.history_points(st, k)
    ]

    def week_of(day: date) -> int:
        return max(int(np.ceil((day - trigger).days / 7)), 0)

    real_id = str(row["idea_id"])
    markers = [
        HistoryMarkerSchema(
            week=week_of(e.event_date), date=e.event_date, kind=e.event_type, label=e.title
        )
        for e in service.events(real_id, as_of)
    ]
    markers += [
        HistoryMarkerSchema(week=week_of(m.date), date=m.date, kind="mention", label=m.title)
        for m in service.mentions(real_id, as_of, _group(row))
    ]
    bill = service.linked_bill(row)
    if bill is not None:
        markers.append(
            HistoryMarkerSchema(
                week=week_of(bill.first_reading_date),
                date=bill.first_reading_date,
                kind="first_reading",
                label=bill.title,
            )
        )
    markers.sort(key=lambda m: (m.date, m.kind))
    return HistoryResponse(
        as_of=as_of,
        model=name,
        run_id=service.run_id_for(name),
        idea_id=real_id,
        signal_id=str(row["signal_id"]),
        trigger_date=trigger,
        taken_up=taken_up,
        points=points,
        markers=markers,
    )
