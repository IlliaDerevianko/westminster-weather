"""Take-up model A: discrete-time hazard with recency features (alternative to the cure model).

One training row per (signal, calendar week t >= 1) from the trigger until the bill week
(y = 1) or censoring (the row's ``data_end``, which carries change-of-government censoring
from ``get_takeup_signals``, or ``T_MAX`` weeks). Weeks and the sitting mask come from
``model.prepare`` so both models see identical outcomes: a week in which neither House sat has
no row (hazard 0); a week the sitting calendar does not cover gets the sitting probability as
its row weight; the bill week is always a sitting week.

  h_i(t) = M_i(t) * sigma(beta_0 + beta . x_i(t))

``x_i(t)`` uses only information dated on or before the START of week t (day
``signal_date + 7(t-1)``): the trigger-day mention(s) define the signal and are always known;
every later mention, event and calendar point counts only once its date is <= the week start.
Features: weeks-since-trigger bins (as the cure model: 1-4, 5-12, 13-26, 27-52, 53-104, 105+)
and log t; weeks since the last mention (0-4, 5-12, 13-26, 27-52, 53+); mentions in the last 26
weeks (capped); log1p cumulative mentions; max firmness so far; any firm mention so far; first
source; years since the last election and months into the session at the week start; the new
Parliament flag; other ``x_*`` events; department one-hots. Features are standardised and fitted
by L2-regularised logistic regression (sklearn, no class weights, so probabilities stay
calibrated); C is chosen on a time-ordered validation split inside the training period (rows
whose week starts in the last ``VALIDATION_YEARS`` years before ``as_of``).

Forecast at ``as_of`` after s silent weeks: future weeks assume NO new mentions (weeks since the
last mention keeps growing, the 26-week count decays as mentions leave the window, no new
events) and weight each future week by the probability that it is a sitting week. With
S(t) = prod_{u<=t} (1 - h(u)):

  pi     = 1 - S(T_MAX)                    bill within 5 years of the trigger (from week 1)
  pi_now = 1 - S(T_MAX) / S(s)             bill within the rest of the horizon, given none by s
  pH     = 1 - S(s+H) / S(s)               bill in the next H weeks
  t_q    = min{t > s : (1 - S(t)/S(s)) / pi_now >= q}   timing if a bill comes within T_MAX

95% intervals for pi and pi_now by the delta method on the logistic coefficients (logit
scale); verdict 'likely' / 'unlikely' / 'uncertain' when the pi_now interval is above / below /
across 0.5. Output keys match ``TakeUpModel.predict``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from westminster_weather.takeup import model as tm

log = logging.getLogger(__name__)

T_MAX = tm.T_MAX
SINCE_EDGES = tm.BIN_EDGES  # weeks since the trigger: 1-4, 5-12, 13-26, 27-52, 53-104, 105+
GAP_EDGES = (0, 5, 13, 27, 53)  # weeks since the last mention: 0-4, 5-12, 13-26, 27-52, 53+
RECENT_WEEKS = 26
RECENT_CAP = 5
YEARS_CAP = 5.5
MONTHS_CAP = 24.0
DAYS_PER_MONTH = 30.4375
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_C = 0.1
VALIDATION_YEARS = 2
HORIZONS = tm.HORIZONS
QUANTILES = tm.QUANTILES
# x_* columns turned into mention features instead of flags (they are derived from mentions)
MENTION_X = ("x_firm_commitment", "x_repeat_mention")
DAY = np.timedelta64(1, "D")
_EPOCH = pd.Timestamp("1970-01-01")


def _day(value: Any) -> float:
    """A date as float days since 1970-01-01 (NaN if missing)."""
    if value is None or pd.isna(value):
        return np.nan
    return float((pd.Timestamp(value).normalize() - _EPOCH).days)


def _days(values: Iterable[Any]) -> np.ndarray:
    return np.sort(np.array([_day(v) for v in values], dtype=float))


def timeline_from_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback mention timeline when the data source has no ``mention_timeline``: the trigger
    (firmness ``z_initial_firmness``), the first repeat mention and the first firm mention
    (``x_repeat_mention`` / ``x_firm_commitment``) of each signal row."""
    rows = []
    for r in df.itertuples(index=False):
        r = r._asdict()
        sid = str(r["signal_id"])
        firm0 = r.get("z_initial_firmness", 1.0)
        rows.append((sid, r["signal_date"], int(0 if pd.isna(firm0) else firm0)))
        if pd.notna(r.get("x_repeat_mention", pd.NaT)):
            rows.append((sid, r["x_repeat_mention"], 1))
        if pd.notna(r.get("x_firm_commitment", pd.NaT)):
            rows.append((sid, r["x_firm_commitment"], 2))
    out = pd.DataFrame(rows, columns=["signal_id", "event_date", "firmness"])
    out["event_date"] = pd.to_datetime(out["event_date"])
    return out.sort_values(["signal_id", "event_date"]).reset_index(drop=True)


class _Mentions:
    """Mention days and their max firmness by signal_id (sorted by date)."""

    def __init__(self, timeline: pd.DataFrame | None) -> None:
        self.by_sid: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if timeline is None or timeline.empty:
            return
        tl = timeline.assign(
            signal_id=timeline["signal_id"].astype(str),
            day=(pd.to_datetime(timeline["event_date"]).dt.normalize() - _EPOCH).dt.days,
            firmness=pd.to_numeric(timeline["firmness"]).fillna(0),
        ).sort_values(["signal_id", "day"])
        # one point per mention DAY (max firmness that day): several contributions in one
        # debate are one mention, as for the cure model's x_repeat_mention
        tl = tl.groupby(["signal_id", "day"], as_index=False, sort=True)["firmness"].max()
        for sid, g in tl.groupby("signal_id", sort=False):
            self.by_sid[sid] = (g["day"].to_numpy(float), g["firmness"].to_numpy(float))

    def get(self, sid: str, until: float) -> tuple[np.ndarray, np.ndarray]:
        days, firm = self.by_sid.get(str(sid), (np.array([]), np.array([])))
        keep = days <= until
        return days[keep], firm[keep]


class HazardModel:
    """Discrete-time hazard model with recency features (module docstring)."""

    def __init__(self, c_grid: Sequence[float] = C_GRID) -> None:
        self.c_grid = tuple(c_grid)
        self.C: float = DEFAULT_C
        self.feature_names: list[str] = []
        self.zcols: list[str] = []  # static z_* columns used as features
        self.xflags: list[str] = []  # x_* event columns used as 0/1 flags
        self.mu = self.sd = None
        self.coef: np.ndarray | None = None  # [intercept, standardised coefficients]
        self.Sigma: np.ndarray | None = None
        self.elections: np.ndarray = np.array([])
        self.session_starts: np.ndarray = np.array([])
        self.validation: list[dict[str, Any]] = []
        self.n_rows = self.n_events = 0

    # ------------------------------------------------------------------ features

    def _set_columns(self, signals: pd.DataFrame) -> None:
        self.zcols = [
            c
            for c in signals.columns
            if c.startswith("z_dept_") or c == "z_src_hansard"  # static: source, department
        ]
        self.xflags = [c for c in signals.columns if c.startswith("x_") and c not in MENTION_X]
        self.feature_names = [
            *(f"since_{e}+" for e in SINCE_EDGES[1:]),
            "log_weeks_since_trigger",
            *(f"gap_{e}+" for e in GAP_EDGES[1:]),
            "recent_mentions_26w",
            "log1p_mentions",
            "max_firmness",
            "any_firm",
            "years_since_election",
            "months_into_session",
            *self.xflags,
            *self.zcols,
        ]

    def week_features(
        self,
        sig: dict[str, Any],
        m_days: np.ndarray,
        m_firm: np.ndarray,
        weeks: np.ndarray,
        known_until: float,
    ) -> np.ndarray:
        """Feature matrix for weeks ``weeks`` of one signal.

        Row for week t uses only points dated <= its week start ``d0 + 7(t-1)`` (and
        <= ``known_until``, the date of the information set); the trigger-day mentions count as
        known from the signal date.
        """
        d0 = _day(sig["signal_date"])
        ws = d0 + 7.0 * (np.asarray(weeks, dtype=float) - 1)
        info = np.minimum(ws, known_until)
        days = np.asarray(m_days, dtype=float)
        firm = np.asarray(m_firm, dtype=float)
        if len(days):  # the trigger-day mentions define the signal: known at week 1
            days = np.where(days == days[0], np.minimum(days, d0), days)
        cum = np.searchsorted(days, info, side="right")
        has = cum > 0
        last = np.where(has, days[np.maximum(cum - 1, 0)] if len(days) else d0, d0)
        gap = np.floor(np.maximum(info - last, 0) / 7.0)
        recent = cum - np.searchsorted(days, info - 7.0 * RECENT_WEEKS, side="right")
        cmax = np.maximum.accumulate(firm) if len(firm) else np.array([0.0])
        maxfirm = np.where(has, cmax[np.maximum(cum - 1, 0)], 0.0)

        t = np.asarray(weeks, dtype=float)
        since = np.searchsorted(SINCE_EDGES, t, side="right") - 1
        gapb = np.searchsorted(GAP_EDGES, gap, side="right") - 1
        cols = [
            *((since == k).astype(float) for k in range(1, len(SINCE_EDGES))),
            np.log(t),
            *((gapb == k).astype(float) for k in range(1, len(GAP_EDGES))),
            np.minimum(recent, RECENT_CAP).astype(float),
            np.log1p(cum.astype(float)),
            maxfirm,
            (maxfirm >= 2).astype(float),
            self._since_points(self.elections, info, sig.get("z_years_since_election"), d0, ws)
            / 365.25,
            self._since_points(
                self.session_starts,
                info,
                None
                if sig.get("z_months_into_session") is None
                else float(sig["z_months_into_session"]) * DAYS_PER_MONTH,
                d0,
                ws,
                in_years=False,
            )
            / DAYS_PER_MONTH,
        ]
        cols[-2] = np.minimum(cols[-2], YEARS_CAP)
        cols[-1] = np.minimum(cols[-1], MONTHS_CAP)
        for c in self.xflags:
            x = _day(sig.get(c))
            cols.append((~np.isnan([x] * len(info)) & (x <= info)).astype(float))
        for c in self.zcols:
            v = sig.get(c, 0.0)
            cols.append(np.full(len(info), 0.0 if v is None or pd.isna(v) else float(v)))
        return np.column_stack(cols) if cols else np.zeros((len(info), 0))

    @staticmethod
    def _since_points(
        points: np.ndarray,
        info: np.ndarray,
        at_trigger: Any,
        d0: float,
        ws: np.ndarray,
        in_years: bool = True,
    ) -> np.ndarray:
        """Days from the last point <= info to the week start. Without a known point: the value
        at the trigger (``at_trigger``, years or days) plus the days elapsed since then."""
        if at_trigger is None or pd.isna(at_trigger):
            base = np.zeros(len(ws))
        else:
            base = float(at_trigger) * (365.25 if in_years else 1.0) + (ws - d0)
        if not len(points):
            return np.maximum(base, 0.0)
        idx = np.searchsorted(points, info, side="right")
        last = np.where(idx > 0, points[np.maximum(idx - 1, 0)], np.nan)
        return np.maximum(np.where(np.isnan(last), base, ws - last), 0.0)

    # ---------------------------------------------------------------------- rows

    def build_rows(
        self,
        signals: pd.DataFrame,
        timeline: pd.DataFrame | None,
        sitting_dates: Any,
        as_of: Any,
        week_probability: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Person-week rows: X, y, weight (sitting mask; 0-weight weeks dropped), week start
        day, signal_id and week number. Outcomes and masks come from ``model.prepare``."""
        as_of = pd.Timestamp(as_of).normalize()
        df = signals.reset_index(drop=True)
        if not self.feature_names:
            self._set_columns(df)
        d = tm.prepare(df, sitting_dates, week_probability, data_end=as_of, calendar_end=as_of)
        tl = timeline if timeline is not None else timeline_from_signals(df)
        mentions = _Mentions(tl)
        known = _day(as_of)
        Xs, ys, ws, starts, sids, wk = [], [], [], [], [], []
        recs = df.to_dict("records")
        for k, sig in enumerate(recs):
            T = int(d["t"][k])
            if T < 1:
                continue
            weeks = np.arange(1, T + 1)
            w = d["M"][k, :T]
            keep = w > 0
            if not keep.any():
                continue
            # the row's own data_end bounds what it may see (change-of-government censoring)
            end_k = min(known, _day(sig.get("data_end", as_of)) if "data_end" in sig else known)
            md, mf = mentions.get(sig["signal_id"], end_k)
            X = self.week_features(sig, md, mf, weeks[keep], known_until=end_k)
            y = np.zeros(T)
            if d["delta"][k] == 1:
                y[T - 1] = 1.0
            Xs.append(X)
            ys.append(y[keep])
            ws.append(w[keep])
            starts.append(_day(sig["signal_date"]) + 7.0 * (weeks[keep] - 1))
            sids.append(np.full(int(keep.sum()), str(sig["signal_id"]), dtype=object))
            wk.append(weeks[keep])
        p = len(self.feature_names)
        cat = np.concatenate
        return {
            "X": cat(Xs) if Xs else np.zeros((0, p)),
            "y": cat(ys) if ys else np.zeros(0),
            "w": cat(ws) if ws else np.zeros(0),
            "week_start": cat(starts) if starts else np.zeros(0),
            "signal_id": cat(sids) if sids else np.zeros(0, dtype=object),
            "week": cat(wk) if wk else np.zeros(0, dtype=int),
            "prepared": d,
        }

    # ------------------------------------------------------------------------ fit

    def fit(
        self,
        signals: pd.DataFrame,
        timeline: pd.DataFrame | None,
        sitting_dates: Any,
        as_of: Any,
        week_probability: pd.Series | None = None,
        elections: Iterable[Any] = (),
        session_starts: Iterable[Any] = (),
    ) -> HazardModel:
        """Fit on ``signals`` (``get_takeup_signals(as_of)``) and ``timeline``
        (``mention_timeline(as_of)``; None = derived from the signal rows)."""
        as_of = pd.Timestamp(as_of).normalize()
        self.elections = _days(elections)
        self.session_starts = _days(session_starts)
        self.feature_names = []
        rows = self.build_rows(signals, timeline, sitting_dates, as_of, week_probability)
        X, y, w = rows["X"], rows["y"], rows["w"]
        if not len(y) or y.sum() == 0:
            raise ValueError("no take-up events to fit the hazard model on")
        self.n_rows, self.n_events = len(y), int(y.sum())
        self.mu = np.average(X, axis=0, weights=w)
        sd = np.sqrt(np.average((X - self.mu) ** 2, axis=0, weights=w))
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        Xs = (X - self.mu) / self.sd

        val_start = _day(as_of - pd.DateOffset(years=VALIDATION_YEARS))
        tr, va = rows["week_start"] < val_start, rows["week_start"] >= val_start
        self.validation = []
        if len(self.c_grid) > 1 and y[tr].sum() >= 5 and y[va].sum() >= 3:
            for C in self.c_grid:
                m = self._logit(C).fit(Xs[tr], y[tr], sample_weight=w[tr])
                p = np.clip(m.predict_proba(Xs[va])[:, 1], 1e-12, 1 - 1e-12)
                ll = -np.sum(w[va] * (y[va] * np.log(p) + (1 - y[va]) * np.log1p(-p)))
                self.validation.append({"C": C, "val_logloss": float(ll / w[va].sum())})
            self.C = min(self.validation, key=lambda r: r["val_logloss"])["C"]
        else:
            self.C = self.c_grid[0] if len(self.c_grid) == 1 else DEFAULT_C
        m = self._logit(self.C).fit(Xs, y, sample_weight=w)
        self.coef = np.concatenate([m.intercept_, m.coef_[0]])
        # covariance: inverse Hessian of the penalised log-likelihood (intercept unpenalised)
        X1 = np.column_stack([np.ones(len(Xs)), Xs])
        p = expit(X1 @ self.coef)
        H = X1.T @ (X1 * (w * p * (1 - p))[:, None])
        H[np.arange(1, X1.shape[1]), np.arange(1, X1.shape[1])] += 1.0 / self.C
        self.Sigma = np.linalg.pinv(H)
        return self

    @staticmethod
    def _logit(C: float) -> LogisticRegression:
        return LogisticRegression(C=C, max_iter=5000, tol=1e-8)

    def parameter_table(self) -> pd.DataFrame:
        """Coefficients on the ORIGINAL feature scale (per unit of the feature), with
        delta-method standard errors; the intercept is for the standardised design."""
        se = np.sqrt(np.clip(np.diag(self.Sigma), 0, None))
        est = np.concatenate([[self.coef[0]], self.coef[1:] / self.sd])
        se = np.concatenate([[se[0]], se[1:] / self.sd])
        return pd.DataFrame(
            {"parameter": ["intercept", *self.feature_names], "estimate": est, "std_err": se}
        )

    # -------------------------------------------------------------------- predict

    def _design(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])

    def predict(
        self,
        sig: dict[str, Any] | pd.Series,
        mentions: pd.DataFrame | None,
        as_of: Any,
        sitting_dates: Any,
        week_probability: pd.Series | None = None,
        horizons: Iterable[int] = HORIZONS,
    ) -> dict[str, Any]:
        """Forecast for one signal with the information at ``as_of`` (no bill by then).

        ``mentions``: its timeline rows (``event_date``, ``firmness``); rows after ``as_of`` and
        ``x_*`` dates after ``as_of`` are ignored. None = derived from the signal row.
        """
        if self.coef is None:
            raise ValueError("fit the model first")
        sig = dict(sig)
        as_of = pd.Timestamp(as_of).normalize()
        known = _day(as_of)
        for c in self.xflags:
            if c in sig and pd.notna(sig[c]) and _day(sig[c]) > known:
                sig[c] = pd.NaT
        if mentions is None:
            mentions = timeline_from_signals(pd.DataFrame([sig]))
        ml = _Mentions(mentions.assign(signal_id=str(sig["signal_id"])))
        md, mf = ml.get(str(sig["signal_id"]), known)
        d0 = _day(sig["signal_date"])
        s = int(np.clip(np.floor((known - d0) / 7.0), 0, T_MAX))
        mask = tm.sitting_mask(
            [sig["signal_date"]],
            [s],
            sitting_dates,
            week_probability,
            calendar_end=as_of,
        )[0]
        weeks = np.arange(1, T_MAX + 1)
        X1 = self._design(self.week_features(sig, md, mf, weeks, known_until=known))
        sig_p = expit(X1 @ self.coef)
        h = mask * sig_p
        log1m = np.log1p(-np.clip(h, 0, 1 - 1e-15))
        logS = np.concatenate([[0.0], np.cumsum(log1m)])  # logS[t] = log S(t)

        pi, pi_lo, pi_hi = self._prob_ci(logS, mask, sig_p, X1, 0)
        pi_now, now_lo, now_hi = self._prob_ci(logS, mask, sig_p, X1, s)
        verdict = "likely" if now_lo > 0.5 else "unlikely" if now_hi < 0.5 else "uncertain"
        out: dict[str, Any] = {
            "pi": pi,
            "pi_lo": pi_lo,
            "pi_hi": pi_hi,
            "verdict": verdict,
            "weeks_elapsed": s,
            "pi_now": pi_now,
            "pi_now_lo": now_lo,
            "pi_now_hi": now_hi,
        }
        cond = -np.expm1(logS[s + 1 :] - logS[s])  # P(bill in weeks s+1..t | none by s)
        for H in horizons:
            Hs = min(s + H, T_MAX)
            out[f"p{H}"] = float(cond[Hs - s - 1]) if Hs > s else 0.0
        sitting = np.cumsum(mask)
        for q in QUANTILES:
            key = tm.quantile_key(q)
            hit = np.nonzero(cond / pi_now >= q - 1e-12)[0] if pi_now > 0 else []
            tq = s + 1 + int(hit[0]) if len(hit) else None
            out[key] = tq
            out[f"{key}_sitting"] = int(round(sitting[tq - 1])) if tq else None
        return out

    def _prob_ci(
        self, logS: np.ndarray, mask: np.ndarray, p: np.ndarray, X1: np.ndarray, s: int
    ) -> tuple[float, float, float]:
        """P(bill in weeks s+1..T_MAX) and its delta-method 95% CI on the logit scale."""
        L = logS[T_MAX] - logS[s]
        prob = max(float(-np.expm1(L)), 0.0)
        if prob <= 0 or s >= T_MAX or self.Sigma is None:
            return prob, prob, prob
        m, ps = mask[s:], p[s:]
        dL = -(m * ps * (1 - ps) / np.clip(1 - m * ps, 1e-15, None)) @ X1[s:]
        g = -dL / prob  # d logit(prob) / d coef = -1/(1 - e^L) dL
        se = float(np.sqrt(max(g @ self.Sigma @ g, 0.0)))
        lg = np.log(prob) - np.log1p(-min(prob, 1 - 1e-15))
        return prob, float(expit(lg - 1.96 * se)), float(expit(lg + 1.96 * se))

    def predict_frame(
        self,
        signals: pd.DataFrame,
        timeline: pd.DataFrame | None,
        sitting_dates: Any,
        week_probability: pd.Series | None = None,
        as_of: Any = None,
        horizons: Iterable[int] = HORIZONS,
    ) -> pd.DataFrame:
        """``predict`` for every row; ``as_of`` None = each row at its own signal date (with
        its bill hidden), else all at ``as_of``. Columns: signal_id, as_of and the outputs."""
        if signals.empty:
            return pd.DataFrame()
        tl = timeline if timeline is not None else timeline_from_signals(signals)
        groups = {
            str(sid): g
            for sid, g in tl.assign(signal_id=tl["signal_id"].astype(str)).groupby("signal_id")
        }
        empty = pd.DataFrame(columns=["signal_id", "event_date", "firmness"])
        rows = []
        for sig in signals.to_dict("records"):
            at = pd.Timestamp(sig["signal_date"] if as_of is None else as_of).normalize()
            p = self.predict(
                sig,
                groups.get(str(sig["signal_id"]), empty),
                at,
                sitting_dates,
                week_probability,
                horizons,
            )
            rows.append(
                {
                    "signal_id": str(sig["signal_id"]),
                    "as_of": at.date(),
                    **{k: v for k, v in p.items() if not k.endswith("_sitting")},
                }
            )
        return pd.DataFrame(rows)
