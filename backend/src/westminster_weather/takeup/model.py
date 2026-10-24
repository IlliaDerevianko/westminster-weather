"""Take-up model: will a government commitment to legislate become a bill, and when?

Mixture cure model in discrete weeks (TAKEUP_PLAN §2.2), ported from the prototype
``takeup_model.py`` with the sitting calendar added (change 1):

  pi_i        = sigma(gamma0 + gamma . z_i)                    P(ever taken up)
  lambda_i(t) = M_i(t) * sigma(alpha_bin(t) + beta . x_i(t))   weekly hazard, if taken up
  S_u(t)      = prod_{s<=t} (1 - lambda_i(s))                  not yet arrived, if taken up
  f_u(t)      = lambda_i(t) S_u(t-1)                           arrives exactly in week t

Week 1 = days 0-6 after the signal date. ``M_i(t)`` is 1 if either House sat on any day of
idea i's week t and 0 if neither did, so weeks without sittings add nothing to the
likelihood and do not lower ``pi_now``. Weeks after the idea's ``data_end``, and weeks the
sitting calendar does not cover, have not been observed: there ``M_i(t)`` is the historical
probability that the week is a sitting week.

A bill first read on or before the row's ``data_end`` is an event, also in the last partial
week. A first reading up to a few days before the signal (the signals layer allows 7) counts
as a week-1 bill.

Log-likelihood:
  bill seen in week t_i : log pi_i + log f_u(t_i)
  no bill after c_i     : log(1 - pi_i + pi_i S_u(c_i))

Input table, one row per idea (``get_takeup_signals(as_of)``, DATA_CONTRACT §10.9):
  signal_id, title, signal_date, bill_date (NaN if none), data_end (optional, per row)
  z_*  numeric features known at the signal
  x_*  dates when a time-varying feature switched on (NaN = has not happened); the feature
       is 1 from that week onwards
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

log = logging.getLogger(__name__)

T_MAX = 260  # follow ideas for at most 5 years
BIN_EDGES = (1, 5, 13, 27, 53, 105)  # baseline alpha_t is constant within these week ranges
RIDGE = 0.05  # small penalty on gamma and beta keeps fits stable with little data
# Penalty on differences of neighbouring baseline bins: keeps a bin with exposure but no
# bills finite (its MLE is -inf) by pulling it towards the bin before it.
ALPHA_SMOOTH = 0.05
ALPHA_START = -3.0  # starting value of the baseline hazard logits
HORIZONS = (13, 26, 52)
QUANTILES = (0.1, 0.5, 0.9)
HESSIAN_EPS = 1e-4

_EPOCH = pd.Timestamp("1970-01-01")


# ---------------------------------------------------------------------------- dates


def _to_days(values: Any) -> np.ndarray:
    """Dates as float days since 1970-01-01, NaN where missing."""
    if values is None:
        return np.array([], dtype=float)
    if isinstance(values, pd.DataFrame):
        values = values["date"]
    s = pd.to_datetime(pd.Series(np.asarray(values, dtype=object)), errors="raise")
    return (s.dt.normalize() - _EPOCH).dt.days.to_numpy(dtype=float)


def _week_of(day: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Week number (1 = first 7 days after the signal) of each date, NaN if missing."""
    return np.floor((day - start) / 7) + 1


def bin_index(n_weeks: int) -> np.ndarray:
    """Baseline bin of each week 1..n_weeks."""
    weeks = np.arange(1, n_weeks + 1)
    return np.searchsorted(BIN_EDGES, weeks, side="right") - 1


def _expected_sitting(days: np.ndarray, week_probability: pd.Series | None) -> np.ndarray:
    """P(a week is a sitting week), looked up by the ISO week of ``days`` (the week's middle).

    ``week_probability`` is indexed by ISO week number (``calendar.week_sitting_probability``);
    ``None`` means every week sits. ISO weeks missing from it get the mean probability.
    """
    if week_probability is None:
        return np.ones(len(days))
    uniq, inverse = np.unique(days, return_inverse=True)
    iso = pd.DatetimeIndex(pd.to_datetime(uniq, unit="D")).isocalendar()["week"]
    prob = pd.Series(week_probability, dtype=float)
    looked_up = prob.reindex(iso.to_numpy(dtype=int)).fillna(prob.mean()).to_numpy()
    return np.clip(looked_up, 0.0, 1.0)[inverse]


def _calendar_span(sit: np.ndarray, calendar_start: Any, calendar_end: Any) -> tuple[float, float]:
    """First and last day the sitting calendar covers (default: first and last sitting)."""
    start = _to_days([calendar_start])[0] if calendar_start is not None else np.nan
    end = _to_days([calendar_end])[0] if calendar_end is not None else np.nan
    if np.isnan(start):
        start = sit[0] if len(sit) else np.inf
    if np.isnan(end):
        end = sit[-1] if len(sit) else -np.inf
    return float(start), float(end)


def sitting_mask(
    signal_dates: Any,
    observed_weeks: Any,
    sitting_dates: Any,
    week_probability: pd.Series | None = None,
    n_weeks: int = T_MAX,
    calendar_start: Any = None,
    calendar_end: Any = None,
) -> np.ndarray:
    """Mask ``M[i, t-1]`` for weeks t = 1..n_weeks of each idea.

    Week t of idea i covers days ``7(t-1) .. 7t-1`` after its signal date. For observed
    weeks (``t <= observed_weeks[i]``) M is 1 if any of ``sitting_dates`` (either House) falls
    in the week, else 0. Later weeks are not observed: M is the probability that the week is
    a sitting week. So is a week with no known sitting that is not wholly inside
    ``[calendar_start, calendar_end]``, the span the calendar is complete for (default: its
    first and last sitting dates; pass ``calendar_end=as_of`` when the calendar is complete up
    to ``as_of``, so that a recess just before ``as_of`` counts as known non-sitting weeks).
    """
    sit = _to_days(sitting_dates)
    sit = np.unique(sit[~np.isnan(sit)])
    return _mask_from_days(
        _to_days(signal_dates),
        np.asarray(observed_weeks, dtype=float).reshape(-1),
        sit,
        week_probability,
        n_weeks,
        *_calendar_span(sit, calendar_start, calendar_end),
    )


def _mask_from_days(
    start: np.ndarray,
    observed: np.ndarray,
    sit: np.ndarray,
    week_probability: pd.Series | None,
    n_weeks: int,
    cal_start: float,
    cal_end: float,
) -> np.ndarray:
    """``sitting_mask`` on day numbers (``sit`` sorted and unique)."""

    bounds = start[:, None] + 7.0 * np.arange(n_weeks + 1)[None, :]
    counts = np.searchsorted(sit, bounds, side="left")
    mask = (np.diff(counts, axis=1) > 0).astype(float)

    first_day = bounds[:, :-1]
    weeks = np.arange(1, n_weeks + 1)
    outside = (mask == 0) & ((first_day < cal_start) | (first_day + 6 > cal_end))
    unobserved = (weeks[None, :] > observed[:, None]) | outside
    if unobserved.any():
        mask[unobserved] = _expected_sitting(first_day[unobserved] + 3, week_probability)
    return mask


# ----------------------------------------------------------------------------- data


def prepare(
    df: pd.DataFrame,
    sitting_dates: Any,
    week_probability: pd.Series | None = None,
    data_end: Any = None,
    calendar_start: Any = None,
    calendar_end: Any = None,
) -> dict[str, Any]:
    """Turn the signals table into the arrays the likelihood needs.

    ``sitting_dates``: dates on which either House sat (``sitting_days.date``).
    ``week_probability``: P(sitting) by ISO week, for weeks after each row's ``data_end``
    (default: every week sits). ``data_end``: global cut; rows never look past it.
    A row's own ``data_end`` (e.g. a dissolution before a change of government) ends its
    follow-up: a bill after it is censored, and events after it are ignored.
    ``calendar_start`` / ``calendar_end``: the span ``sitting_dates`` is complete for (default:
    its first and last sitting); see ``sitting_mask``. Pass ``calendar_end=as_of`` with a
    calendar cut at ``as_of``.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    signal = _to_days(df["signal_date"])
    if np.isnan(signal).any():
        raise ValueError("every row needs a signal_date")

    default_end = pd.Timestamp(data_end) if data_end is not None else pd.Timestamp.today()
    default_day = float((default_end.normalize() - _EPOCH).days)
    end = _to_days(df["data_end"]) if "data_end" in df else np.full(n, np.nan)
    end = np.where(np.isnan(end), default_day, end)
    if data_end is not None:  # never look past the cut (no leakage)
        end = np.minimum(end, default_day)

    zcols = [c for c in df.columns if c.startswith("z_")]
    xcols = [c for c in df.columns if c.startswith("x_")]

    bill_day = _to_days(df["bill_date"]) if "bill_date" in df else np.full(n, np.nan)
    bill_week = _week_of(bill_day, signal)
    obs_week = np.floor((end - signal) / 7)  # whole weeks observed without a bill
    # A bill read on or before data_end is an event, also in the last partial week.
    seen = ~np.isnan(bill_day) & (bill_day <= end)
    # A first reading shortly before the signal (the signals layer allows 7 days) is a bill
    # for this idea: count it in week 1, not as a silent idea.
    early = seen & (bill_week < 1)
    if early.any():
        log.warning("%d bill(s) read before their signal date counted as week 1", early.sum())
    bill_week = np.where(early, 1.0, bill_week)
    delta = (seen & (bill_week <= T_MAX)).astype(int)  # later: censored at T_MAX
    t = np.clip(np.where(delta == 1, bill_week, obs_week), 0, T_MAX).astype(int)

    xweek = np.zeros((n, len(xcols)))
    for j, c in enumerate(xcols):
        xd = _to_days(df[c])
        xd = np.where(xd > end, np.nan, xd)  # not yet known at the row's data_end
        xweek[:, j] = np.nan_to_num(np.maximum(_week_of(xd, signal), 1), nan=T_MAX + 1)

    # an idea with a bill is observed up to its bill week (which may be a partial week)
    observed = np.clip(np.where(delta == 1, np.maximum(obs_week, t), obs_week), 0, T_MAX)
    sit = _to_days(sitting_dates)
    sit = np.unique(sit[~np.isnan(sit)])
    span = _calendar_span(sit, calendar_start, calendar_end)
    mask = _mask_from_days(signal, observed, sit, week_probability, T_MAX, *span)

    # A first reading proves a House sat; a 0 here is a gap in the calendar data.
    ev = np.nonzero(delta == 1)[0]
    gap = mask[ev, t[ev] - 1] == 0
    forced = np.zeros(n, dtype=bool)
    forced[ev[gap]] = True
    mask[ev, t[ev] - 1] = 1.0
    if gap.any():
        log.warning("%d bill week(s) had no sitting in the calendar; set to sitting", gap.sum())

    d = {
        "Z": df[zcols].astype(float).fillna(0).to_numpy(),
        "zcols": zcols,
        "xweek": xweek,
        "xcols": xcols,
        "M": mask,
        "t": t,
        "delta": delta,
        "weeks_observed": observed.astype(int),
        "ids": df["signal_id"].astype(str).to_numpy(),
        "titles": df["title"].astype(str).to_numpy() if "title" in df else None,
        "signal_date": signal.astype("datetime64[D]"),
        "data_end": end.astype("datetime64[D]"),
        "forced": forced,
        "bill_before_signal": early,
    }
    return _with_summaries(d)


_ROW_KEYS = (
    "Z",
    "xweek",
    "M",
    "t",
    "delta",
    "weeks_observed",
    "ids",
    "titles",
    "signal_date",
    "data_end",
    "forced",
    "bill_before_signal",
)


def _with_summaries(d: dict[str, Any]) -> dict[str, Any]:
    """Add counts derived from the row arrays.

    ``n_forced_sitting`` / ``forced_ids``: bill weeks the calendar had as non-sitting.
    ``n_bill_before_signal``: bills read before their signal date, counted as week 1.
    ``bin_events`` / ``bin_exposure``: bills, and sitting weeks at risk (sum of M up to each
    row's t), per baseline bin. A bin with exposure but no bills is held up only by the
    smoothing penalty.
    """
    t, delta = d["t"], d["delta"]
    bins = bin_index(T_MAX)
    at_risk = (np.arange(1, T_MAX + 1)[None, :] <= t[:, None]) * d["M"]
    d["n_forced_sitting"] = int(d["forced"].sum())
    d["forced_ids"] = d["ids"][d["forced"]]
    d["n_bill_before_signal"] = int(d["bill_before_signal"].sum())
    d["bin_events"] = np.bincount(bins[t[delta == 1] - 1], minlength=len(BIN_EDGES))
    d["bin_exposure"] = np.bincount(bins, weights=at_risk.sum(axis=0), minlength=len(BIN_EDGES))
    return d


def resample(d: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    """Rows ``idx`` of prepared data (for the bootstrap); the summary counts are recomputed."""
    out = dict(d)
    for key in _ROW_KEYS:
        if d.get(key) is not None:
            out[key] = d[key][idx]
    return _with_summaries(out)


def row_inputs(d: dict[str, Any], k: int) -> dict[str, Any]:
    """Keyword arguments for ``TakeUpModel.predict`` for row ``k`` of prepared data.

    An idea with a bill is predicted from its signal (``weeks_elapsed = 0``); one without is
    predicted after its observed silent weeks.
    """
    return {
        "z": d["Z"][k],
        "xweek_row": d["xweek"][k],
        "weeks_elapsed": int(d["t"][k]) if d["delta"][k] == 0 else 0,
        "mask": d["M"][k],
    }


# ------------------------------------------------------------------------ the model


class TakeUpModel:
    """Mixture cure model with a sitting-week mask on the hazard."""

    def __init__(self) -> None:
        self.zcols: list[str] = []
        self.xcols: list[str] = []
        self.pz = self.px = 0
        self.theta: np.ndarray | None = None
        self.converged = False
        self.Sigma: np.ndarray | None = None

    @classmethod
    def from_params(
        cls,
        theta: Sequence[float],
        zcols: Sequence[str],
        xcols: Sequence[str],
        Sigma: np.ndarray | None = None,
    ) -> TakeUpModel:
        """A model with given parameters (e.g. loaded from ``takeup_runs.parameters``)."""
        m = cls()
        m.zcols, m.xcols = list(zcols), list(xcols)
        m.pz, m.px = len(m.zcols), len(m.xcols)
        m.theta = np.asarray(theta, dtype=float)
        if len(m.theta) != m.n_params:
            raise ValueError(f"expected {m.n_params} parameters, got {len(m.theta)}")
        m.Sigma = None if Sigma is None else np.asarray(Sigma, dtype=float)
        return m

    @property
    def n_params(self) -> int:
        return 1 + self.pz + len(BIN_EDGES) + self.px

    @property
    def param_names(self) -> list[str]:
        return ["intercept", *self.zcols, *(f"alpha_weeks_{e}+" for e in BIN_EDGES), *self.xcols]

    def _unpack(self, theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        pz, nb, px = self.pz, len(BIN_EDGES), self.px
        g0, g = theta[0], theta[1 : 1 + pz]
        a = theta[1 + pz : 1 + pz + nb]
        b = theta[1 + pz + nb : 1 + pz + nb + px]
        return g0, g, a, b

    def _hazard_logits(self, xweek: np.ndarray, W: int, a: np.ndarray, b: np.ndarray):
        """h[i, t-1] = alpha_bin(t) + beta . x_i(t), for weeks 1..W."""
        weeks = np.arange(1, W + 1)
        h = np.broadcast_to(a[bin_index(W)], (len(xweek), W)).copy()
        for j in range(xweek.shape[1]):
            h += b[j] * (weeks[None, :] >= xweek[:, j : j + 1])
        return h

    @staticmethod
    def _log_one_minus_hazard(h: np.ndarray, m: np.ndarray) -> np.ndarray:
        """log(1 - m * sigma(h)), stable when m = 1."""
        full = m >= 1
        partial = np.log1p(-np.where(full, 0.0, m) * expit(h))
        return np.where(full, log_expit(-h), partial)

    def _loglik(
        self, theta: np.ndarray, d: dict[str, Any], grad: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Per-row log-likelihood and (optionally) the gradient of its sum."""
        g0, g, a, b = self._unpack(theta)
        Z, xweek, t, delta = d["Z"], d["xweek"], d["t"], d["delta"]
        n = len(t)
        eta = g0 + Z @ g
        log_pi, log_1mpi = log_expit(eta), log_expit(-eta)
        W = max(int(t.max()), 1) if n else 1
        h = self._hazard_logits(xweek, W, a, b)
        m = d["M"][:, :W]
        log_surv = self._log_one_minus_hazard(h, m)
        # logS[:, t] = log S_u(t)
        logS = np.concatenate([np.zeros((n, 1)), np.cumsum(log_surv, axis=1)], axis=1)

        rows = np.arange(n)
        ev = delta == 1
        re, te = rows[ev], t[ev]
        rc, tc = rows[~ev], t[~ev]
        ll = np.empty(n)
        with np.errstate(divide="ignore"):
            log_m_event = np.log(m[re, te - 1])
        ll[ev] = log_pi[ev] + log_m_event + log_expit(h[re, te - 1]) + logS[re, te - 1]
        ll[~ev] = np.logaddexp(log_1mpi[~ev], log_pi[~ev] + logS[rc, tc])
        if not grad:
            return ll, None

        # d ll / d h[i, s]
        p = expit(h)
        full = m >= 1
        q = np.where(full, -p, -m * p * (1 - p) / (1 - np.where(full, 0.0, m) * p))
        weeks = np.arange(1, W + 1)
        G = np.zeros((n, W))
        G[ev] = np.where(weeks[None, :] < te[:, None], q[ev], 0.0)
        G[re, te - 1] = 1 - p[re, te - 1]
        w = np.exp(log_pi[~ev] + logS[rc, tc] - ll[~ev])  # P(susceptible | no bill by c)
        G[~ev] = np.where(weeks[None, :] <= tc[:, None], q[~ev], 0.0) * w[:, None]

        pi = expit(eta)
        deta = np.empty(n)
        deta[ev] = 1 - pi[ev]
        deta[~ev] = w - pi[~ev]
        grad_a = np.bincount(bin_index(W), weights=G.sum(axis=0), minlength=len(BIN_EDGES))
        grad_b = np.array(
            [np.sum(G * (weeks[None, :] >= xweek[:, j : j + 1])) for j in range(self.px)]
        )
        gradient = np.concatenate([[deta.sum()], Z.T @ deta, grad_a, grad_b])
        return ll, gradient

    def _penalty(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        _, g, a, b = self._unpack(theta)
        step = np.diff(a)
        pen = 0.5 * RIDGE * (np.sum(g**2) + np.sum(b**2)) + 0.5 * ALPHA_SMOOTH * np.sum(step**2)
        dpen = np.zeros_like(theta)
        dpen[1 : 1 + self.pz] = RIDGE * g
        da = np.zeros(len(a))
        da[1:] += ALPHA_SMOOTH * step
        da[:-1] -= ALPHA_SMOOTH * step
        dpen[1 + self.pz : 1 + self.pz + len(BIN_EDGES)] = da
        dpen[1 + self.pz + len(BIN_EDGES) :] = RIDGE * b
        return pen, dpen

    def _negloglik(self, theta: np.ndarray, d: dict[str, Any]) -> float:
        ll, _ = self._loglik(theta, d)
        return -ll.sum() + self._penalty(theta)[0]

    def _negloglik_and_grad(self, theta: np.ndarray, d: dict[str, Any]):
        ll, gradient = self._loglik(theta, d, grad=True)
        pen, dpen = self._penalty(theta)
        return -ll.sum() + pen, -gradient + dpen

    def loglik_contributions(self, d: dict[str, Any], theta: np.ndarray | None = None):
        """Each row's log-likelihood (without the ridge penalty)."""
        return self._loglik(self.theta if theta is None else np.asarray(theta), d)[0]

    def fit(
        self, d: dict[str, Any], hessian: bool = True, theta0: np.ndarray | None = None
    ) -> TakeUpModel:
        """Penalised maximum likelihood by L-BFGS; the covariance from the numeric Hessian."""
        self.zcols, self.xcols = list(d["zcols"]), list(d["xcols"])
        self.pz, self.px = len(self.zcols), len(self.xcols)
        if "bin_events" in d:
            empty = (d["bin_events"] == 0) & (d["bin_exposure"] > 0)
            if empty.any():
                names = [f"weeks_{e}+" for e, x in zip(BIN_EDGES, empty, strict=True) if x]
                log.warning("baseline bin(s) with exposure but no bills: %s", ", ".join(names))
        if theta0 is None:
            theta0 = np.zeros(self.n_params)
            theta0[1 + self.pz : 1 + self.pz + len(BIN_EDGES)] = ALPHA_START
        res = minimize(
            self._negloglik_and_grad,
            np.asarray(theta0, dtype=float),
            args=(d,),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7},
        )
        self.theta, self.converged = res.x, bool(res.success)
        if not self.converged:
            log.warning("take-up fit did not fully converge: %s", res.message)
        self.Sigma = np.linalg.pinv(self._num_hessian(d)) if hessian else None
        return self

    def _num_hessian(self, d: dict[str, Any], eps: float = HESSIAN_EPS) -> np.ndarray:
        """Hessian of the penalised negative log-likelihood by central differences of its
        gradient; its inverse estimates the covariance."""
        k = len(self.theta)
        H = np.zeros((k, k))
        for i in range(k):
            e = np.zeros(k)
            e[i] = eps
            g_plus = self._negloglik_and_grad(self.theta + e, d)[1]
            g_minus = self._negloglik_and_grad(self.theta - e, d)[1]
            H[:, i] = (g_plus - g_minus) / (2 * eps)
        return (H + H.T) / 2

    def standard_errors(self) -> np.ndarray:
        """Square roots of the covariance diagonal; NaN where it is not positive (the
        parameter is not identified by the data, so it has no usable standard error)."""
        if self.Sigma is None:
            raise ValueError("fitted without the Hessian")
        var = np.diag(self.Sigma)
        bad = ~(var > 0)
        if bad.any():
            names = [n for n, x in zip(self.param_names, bad, strict=True) if x]
            log.warning("non-positive variance (standard error NaN): %s", ", ".join(names))
        with np.errstate(invalid="ignore"):
            return np.where(bad, np.nan, np.sqrt(var))

    def parameter_table(self) -> pd.DataFrame:
        """Estimates and standard errors, one row per parameter."""
        se = self.standard_errors() if self.Sigma is not None else np.full(self.n_params, np.nan)
        return pd.DataFrame({"parameter": self.param_names, "estimate": self.theta, "std_err": se})

    # ------------------------------------------------------------------ predictions

    def pi_with_ci(self, z: np.ndarray) -> tuple[float, float, float, str]:
        """pi, its 95% CI by the delta method, and the verdict."""
        g0, g, _, _ = self._unpack(self.theta)
        z = np.asarray(z, dtype=float)
        eta = g0 + z @ g
        zf = np.concatenate([[1.0], z])
        if self.Sigma is None:  # bootstrap refits skip the Hessian
            se = 0.0
        else:
            S = self.Sigma[: 1 + self.pz, : 1 + self.pz]
            se = float(np.sqrt(max(zf @ S @ zf, 0)))
        lo, hi = expit(eta - 1.96 * se), expit(eta + 1.96 * se)
        verdict = "likely" if lo > 0.5 else "unlikely" if hi < 0.5 else "uncertain"
        return float(expit(eta)), float(lo), float(hi), verdict

    def survival_u(
        self, xweek_row: np.ndarray, mask: np.ndarray | None = None, W: int = T_MAX
    ) -> np.ndarray:
        """S_u(1..W) for one idea; ``mask`` is its sitting mask (all ones if None)."""
        _, _, a, b = self._unpack(self.theta)
        h = self._hazard_logits(np.asarray(xweek_row, dtype=float)[None, :], W, a, b)[0]
        m = np.ones(W) if mask is None else np.asarray(mask, dtype=float)[:W]
        return np.exp(np.cumsum(self._log_one_minus_hazard(h, m)))

    def predict(
        self,
        z: np.ndarray,
        xweek_row: np.ndarray,
        weeks_elapsed: int = 0,
        horizons: Iterable[int] = HORIZONS,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Forecast for one idea after ``weeks_elapsed`` silent calendar weeks.

        ``mask`` is the idea's sitting mask from ``prepare`` (observed weeks 0/1, later weeks
        the sitting probability); all ones if None. Timing quantiles ``t_q`` are calendar
        weeks from the signal, if taken up; ``t_q_sitting`` is the (expected) number of
        sitting weeks up to and including week ``t_q``.
        """
        pi, lo, hi, verdict = self.pi_with_ci(z)
        m = np.ones(T_MAX) if mask is None else np.asarray(mask, dtype=float)[:T_MAX]
        Su = self.survival_u(xweek_row, m)
        Su0 = np.concatenate([[1.0], Su])  # Su0[t] = S_u(t)
        s = int(min(max(weeks_elapsed, 0), T_MAX))
        pi_now = pi * Su0[s] / (1 - pi + pi * Su0[s])
        out: dict[str, Any] = {
            "pi": pi,
            "pi_lo": lo,
            "pi_hi": hi,
            "verdict": verdict,
            "weeks_elapsed": s,
            "pi_now": float(pi_now),
        }
        for H in horizons:  # chance of a bill by week s+H, from now
            Hs = min(s + H, T_MAX)
            p = pi_now * (1 - Su0[Hs] / Su0[s]) if Su0[s] > 0 else 0.0
            out[f"p_bill_next_{H}w"] = float(p)
        sitting_weeks = np.cumsum(m)
        for q in QUANTILES:  # t_q = first week with 1 - S_u(t) >= q
            hit = np.nonzero(1 - Su >= q)[0]
            tq = int(hit[0] + 1) if len(hit) else None
            out[f"t_{q}"] = tq
            out[f"t_{q}_sitting"] = int(round(sitting_weeks[tq - 1])) if tq else None
        return out


# ------------------------------------------------------------------------ utilities


def bootstrap(
    d: dict[str, Any], B: int, seed: int = 0, base: TakeUpModel | None = None
) -> list[TakeUpModel]:
    """Refit on ideas resampled with replacement (no Hessian, for speed).

    If ``base`` is given, each refit starts from its estimates.
    """
    rng = np.random.default_rng(seed)
    n = len(d["t"])
    theta0 = None if base is None else base.theta
    return [
        TakeUpModel().fit(resample(d, rng.integers(0, n, n)), hessian=False, theta0=theta0)
        for _ in range(B)
    ]


def bootstrap_intervals(
    models: Sequence[TakeUpModel],
    z: np.ndarray,
    xweek_row: np.ndarray,
    weeks_elapsed: int = 0,
    mask: np.ndarray | None = None,
    level: float = 0.90,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of ``pi_now`` and the 26 / 52-week probabilities."""
    keys = ("pi_now", "p_bill_next_26w", "p_bill_next_52w")
    preds = [m.predict(z, xweek_row, weeks_elapsed, horizons=(26, 52), mask=mask) for m in models]
    tail = 100 * (1 - level) / 2
    out = {}
    for key in keys:
        lo, hi = np.percentile([p[key] for p in preds], [tail, 100 - tail])
        out[key] = (float(lo), float(hi))
    return out


def describe(
    sid: str, title: str, p: dict[str, Any], extra: tuple[float, float] | None = None
) -> str:
    """Plain-English summary of one forecast; ``extra`` is a bootstrap interval of pi_now."""

    def when(cal: int | None, sit: int | None) -> str:
        return f"week {cal} ({sit} sitting weeks)" if cal else "beyond 5 years"

    lines = [
        f"{sid}  {title}",
        f"  Chance it is ever taken up: {p['pi']:.0%} "
        f"(95% CI {p['pi_lo']:.0%}–{p['pi_hi']:.0%}) -> {p['verdict'].upper()}",
    ]
    if p["weeks_elapsed"]:
        interval = f" (90% bootstrap {extra[0]:.0%}–{extra[1]:.0%})" if extra is not None else ""
        lines.append(
            f"  After {p['weeks_elapsed']} weeks with no bill, chance now: "
            f"{p['pi_now']:.0%}{interval}"
        )
    lines.append(
        "  Chance of a bill in the next 13 / 26 / 52 weeks: "
        f"{p['p_bill_next_13w']:.0%} / {p['p_bill_next_26w']:.0%} / {p['p_bill_next_52w']:.0%}"
    )
    lines.append(
        f"  If taken up: median {when(p['t_0.5'], p['t_0.5_sitting'])}, 80% window "
        f"{when(p['t_0.1'], p['t_0.1_sitting'])} to {when(p['t_0.9'], p['t_0.9_sitting'])}"
    )
    return "\n".join(lines)
