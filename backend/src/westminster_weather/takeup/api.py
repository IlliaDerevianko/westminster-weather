"""FastAPI endpoints for the take-up model (Phase 1 -> 2; TAKEUP_PLAN task 10).

- ``GET /api/v1/takeup/ideas?as_of=``: every signal visible at ``as_of`` with its forecast.
- ``GET /api/v1/takeup/forecast?idea_id=&as_of=``: one idea's forecast, events and mentions.
- ``GET /api/v1/takeup/history?idea_id=&as_of=``: weekly ``pi_now`` / ``p52`` from the trigger.

Everything comes from ``TakeupData.get_takeup_signals(as_of)`` and the take-up tables filtered
to ``<= as_of`` (DATA_CONTRACT §10.9): nothing dated after ``as_of`` reaches a response.

The model is rebuilt from the latest ``takeup_runs`` row (``parameters`` JSON); without one it
is fitted once (``evaluate.fit_all``) on the first request and kept in memory. Stored
``takeup_forecasts`` rows of that run for (signal, as_of) are preferred over computed values.
Nothing is written to any database.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import date, timedelta
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from westminster_weather.takeup import model as tm
from westminster_weather.takeup.signals import SOURCE_RANK, TakeupData

logger = logging.getLogger(__name__)

TABLE_TTL_SECONDS = 600  # reload take-up and reference tables after this long
AS_OF_CACHE_SIZE = 16  # prepared signals kept per service, by as_of
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


class IdeasResponse(BaseModel):
    as_of: date
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


class TakeupService:
    """Data, model and caches behind the endpoints (injectable for tests)."""

    def __init__(
        self,
        data: TakeupData,
        model: tm.TakeUpModel,
        run_id: int | None = None,
        stored_lookup: StoredLookup | None = None,
    ) -> None:
        self.data = data
        self.model = model
        self.run_id = run_id
        self.stored_lookup = stored_lookup
        self.features = [*model.zcols, *model.xcols]
        self._cache: OrderedDict[date, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    # -------------------------------------------------------------- per-as_of state

    def state(self, as_of: date) -> dict[str, Any]:
        """Signals, prepared arrays and forecasts at ``as_of`` (LRU-cached)."""
        with self._lock:
            if as_of in self._cache:
                self._cache.move_to_end(as_of)
                return self._cache[as_of]
        st = self._build(as_of)
        with self._lock:
            self._cache[as_of] = st
            while len(self._cache) > AS_OF_CACHE_SIZE:
                self._cache.popitem(last=False)
        return st

    def _build(self, as_of: date) -> dict[str, Any]:
        ts = pd.Timestamp(as_of)
        df = self.data.get_takeup_signals(ts, features=self.features).reset_index(drop=True)
        st: dict[str, Any] = {"df": df, "d": None, "preds": [], "stored": {}}
        if df.empty:
            return st
        wp = self.data.week_probability(ts)
        d = tm.prepare(df, self.data.sitting_dates(ts), wp, data_end=ts, calendar_end=ts)
        st["d"] = d
        st["preds"] = [self.model.predict(**tm.row_inputs(d, k)) for k in range(len(df))]
        if self.stored_lookup is not None:
            try:
                st["stored"] = self.stored_lookup(df["signal_id"].astype(str).tolist(), as_of)
            except Exception:  # stored forecasts are optional; computed ones still work
                logger.warning("takeup_forecasts lookup failed; using computed forecasts")
        st["first_source"] = self._first_sources(ts)
        return st

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


_default: dict[str, Any] = {"service": None, "loaded_at": 0.0, "fitted": None}
_default_lock = threading.Lock()


def _load_default_service() -> TakeupService:
    from westminster_weather.takeup import db

    data = TakeupData()
    _ = data.tables  # load now (take-up DB + production reference tables, read only)
    engine = db.get_takeup_engine()
    run = None
    try:
        run = _latest_run(engine)
    except Exception:
        logger.warning("could not read takeup_runs; falling back to an in-memory fit")
    if run is not None:
        model = model_from_run(run)
        run_id = int(run["run_id"])
        return TakeupService(data, model, run_id, _stored_lookup(engine, run_id))
    if _default["fitted"] is None:
        from westminster_weather.takeup import evaluate

        logger.warning("no takeup_runs row: fitting the take-up model in memory (not saved)")
        _default["fitted"] = evaluate.fit_all(data, evaluate.default_end(data))["model"]
    return TakeupService(data, _default["fitted"], None, None)


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


# ----------------------------------------------------------------------- endpoints


@router.get("/ideas", response_model=IdeasResponse)
def list_ideas(service: Service, as_of: AsOf = None) -> IdeasResponse:
    """Every idea (signal) visible at ``as_of``, with its take-up forecast."""
    as_of = as_of or date.today()
    st = service.state(as_of)
    ideas = [service.summary(st, k) for k in range(len(st["df"]))]
    return IdeasResponse(as_of=as_of, run_id=service.run_id, count=len(ideas), ideas=ideas)


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(service: Service, idea_id: IdeaId, as_of: AsOf = None) -> ForecastResponse:
    """One idea's forecast at ``as_of``, with its events and mentions dated ``<= as_of``."""
    as_of = as_of or date.today()
    st = service.state(as_of)
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
        run_id=service.run_id,
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
        events=service.events(real_id, as_of),
        mentions=service.mentions(real_id, as_of, _group(row)),
        linked_bill=service.linked_bill(row),
        forecast_source=source,
    )


@router.get("/history", response_model=HistoryResponse)
def get_history(service: Service, idea_id: IdeaId, as_of: AsOf = None) -> HistoryResponse:
    """Weekly ``pi_now`` and ``p52`` from the trigger to ``as_of`` (or to the bill's week).

    Prepared once at ``as_of``; the point for week w uses only the events known by week w
    (later ``x`` events are masked), the same ``z`` and the idea's sitting mask."""
    as_of = as_of or date.today()
    st = service.state(as_of)
    k, _ = service.resolve(st, idea_id)
    d, row = st["d"], st["df"].iloc[k]
    trigger = pd.Timestamp(row["signal_date"]).date()
    taken_up = bool(d["delta"][k] == 1)
    last = int(d["t"][k]) - 1 if taken_up else int(d["t"][k])  # silent weeks only
    z, xweek, mask = d["Z"][k], d["xweek"][k], d["M"][k]
    points = []
    for w in range(0, max(last, 0) + 1):
        xw = np.where(xweek <= w, xweek, tm.T_MAX + 1)
        p = service.model.predict(z, xw, weeks_elapsed=w, mask=mask)
        points.append(
            HistoryPointSchema(
                week=w, date=_week_date(trigger, w), pi_now=p["pi_now"], p52=p["p52"]
            )
        )

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
        run_id=service.run_id,
        idea_id=real_id,
        signal_id=str(row["signal_id"]),
        trigger_date=trigger,
        taken_up=taken_up,
        points=points,
        markers=markers,
    )
