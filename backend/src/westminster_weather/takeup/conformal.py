"""Conformalised timing windows (CQR-style) for the take-up models.

Both models give an 80% timing window [t_10, t_90] (calendar weeks from the trigger, given a
bill comes). On real data the raw windows cover far fewer than 80% of bills, so they are
recalibrated by split conformal quantile regression (Romano, Patterson and Candes 2019) with a
scale-normalised score:

  w_i = max(t_90 - t_10, 1)
  E_i = max((t_10 - t_i) / w_i, (t_i - t_90) / w_i)     (> 0: bill week outside the window)
  q   = the ceil((n + 1)(1 - alpha))-th smallest E_i     (finite-sample conformal quantile)
  window -> [t_10 - q w, t_90 + q w], with the lower end kept > s (weeks already passed)

Scores come from a CALIBRATION set that the model never trained on: ideas with a bill whose
triggers fall in a held-out window just before the test cut, predicted by a model fitted only
on data before that window (``evaluate.backtest``). A missing t_10 / t_90 (the quantile is
beyond ``T_MAX``) is treated as ``T_MAX + 1``. A negative q narrows the window.

The coverage guarantee (>= 1 - alpha on average) holds only if calibration and test ideas are
exchangeable. They are not exactly: the calibration window is earlier in time (drift in how
fast governments legislate), and its bills are right-truncated (only bills first read before
the test cut are seen, so slow bills are under-represented in the scores). Treat the
calibrated coverage as approximate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from westminster_weather.takeup.model import T_MAX

ALPHA = 0.2  # 80% windows
MIN_CALIBRATION = 10  # fewer calibration bills -> no adjustment (q = 0)


def _num(values: Sequence[Any], missing: float = T_MAX + 1) -> np.ndarray:
    return np.array([missing if v is None or pd.isna(v) else float(v) for v in values], dtype=float)


def conformity_scores(t: Sequence[float], lo: Sequence[Any], hi: Sequence[Any]) -> np.ndarray:
    """Scale-normalised CQR scores of actual bill weeks ``t`` against [lo, hi]."""
    t, lo, hi = np.asarray(t, dtype=float), _num(lo), _num(hi)
    w = np.maximum(hi - lo, 1.0)
    return np.maximum((lo - t) / w, (t - hi) / w)


def conformal_quantile(scores: Sequence[float], alpha: float = ALPHA) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score; +inf if that exceeds n."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(s[k - 1])


def widen(
    lo: Sequence[Any], hi: Sequence[Any], q: float, s: Sequence[int] | int = 0
) -> tuple[list[int | None], list[int | None]]:
    """[lo - q w, hi + q w] with w = max(hi - lo, 1), the lower end kept > s and >= 1. An
    upper end above ``T_MAX`` (or infinite q) becomes None ('beyond 5 years')."""
    lo_a, hi_a = _num(lo), _num(hi)
    s_a = np.broadcast_to(np.asarray(s, dtype=float), lo_a.shape)
    w = np.maximum(hi_a - lo_a, 1.0)
    if math.isinf(q):
        new_lo, new_hi = s_a + 1, np.full(len(lo_a), np.inf)
    else:
        new_lo = np.floor(lo_a - q * w)
        new_hi = np.ceil(hi_a + q * w)
    new_lo = np.maximum(new_lo, np.maximum(s_a + 1, 1))
    new_hi = np.maximum(new_hi, new_lo)
    out_lo = [None if x > T_MAX else int(x) for x in new_lo]
    out_hi = [None if x > T_MAX else int(x) for x in new_hi]
    return out_lo, out_hi


@dataclass
class CQRCalibrator:
    """Split-conformal calibration of [t_10, t_90] windows."""

    alpha: float = ALPHA
    q: float = 0.0
    n: int = 0
    scores: list[float] = field(default_factory=list)

    def fit(self, t: Sequence[float], lo: Sequence[Any], hi: Sequence[Any]) -> CQRCalibrator:
        sc = conformity_scores(t, lo, hi)
        self.n, self.scores = len(sc), [float(x) for x in sc]
        self.q = conformal_quantile(sc, self.alpha) if self.n >= MIN_CALIBRATION else 0.0
        return self

    def apply(
        self, lo: Sequence[Any], hi: Sequence[Any], s: Sequence[int] | int = 0
    ) -> tuple[list[int | None], list[int | None]]:
        return widen(lo, hi, self.q, s)

    def summary(self) -> dict[str, Any]:
        raw = float(np.mean(np.asarray(self.scores) <= 0)) if self.scores else None
        return {
            "alpha": self.alpha,
            "n_calibration": self.n,
            "q": None if math.isinf(self.q) else self.q,
            "raw_coverage_on_calibration": raw,
        }
