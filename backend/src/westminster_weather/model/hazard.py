"""Advance hazard: the daily chance a bill moves to its next state (BUILD_PLAN task 6,
MODEL_SPEC §4.1).

    logit P(advance on day t | still in k) =
        α_k + f_k(τ)            one effect per (state, time-in-state bucket) cell
      + β · x_i + γ · z_i(t)    Tier 1 features, shared across states
      + δ · g(d_i(t))           end-of-session buckets (41+ sitting days left = reference)

One pooled L2-regularised logistic regression on the bill × sitting-day rows from
``features.build_training_rows``. ``C`` is chosen by time-ordered validation: fit on
training sessions up to ``VALIDATION_SPLIT``, score log-loss on the later training sessions,
then refit on all training sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from westminster_weather.model import features
from westminster_weather.model.splits import TRAIN_CUTOFF, TRAIN_SESSIONS

# Time-ordered validation inside the training sessions (BUILD_PLAN task 6).
VALIDATION_SPLIT = 26  # fit on sessions <= 26, validate on 27-30
C_GRID: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MAX_ITER = 2000

CONTINUOUS = ("introduced_months_into_session", "n_stage_sittings_cum", "parliament_age_years")
END_LEVELS = tuple(label for _, _, label in features.END_BUCKETS if label != "41+")
TAU_LEVELS = tuple(label for _, _, label in features.TAU_BUCKETS)
STATE_LEVELS = tuple(s.value for s in features.HAZARD_STATES)


@dataclass
class DesignSpec:
    """How rows become model inputs; fitted on training rows and stored with the model."""

    shared: list[str]  # Tier 1 feature columns (binary as-is, continuous standardised)
    means: dict[str, float] = field(default_factory=dict)
    sds: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, rows: pd.DataFrame, feature_columns: list[str]) -> DesignSpec:
        spec = cls(shared=list(feature_columns))
        for col in CONTINUOUS:
            spec.means[col] = float(rows[col].mean())
            spec.sds[col] = float(rows[col].std()) or 1.0
        return spec

    @property
    def columns(self) -> list[str]:
        cells = [f"cell:{s}|{t}" for s in STATE_LEVELS for t in TAU_LEVELS]
        ends = [f"end:{e}" for e in END_LEVELS]
        return [*cells, *ends, *self.shared]

    def matrix(self, rows: pd.DataFrame) -> np.ndarray:
        out = pd.DataFrame(0.0, index=rows.index, columns=self.columns)
        cell = "cell:" + rows["state"].astype(str) + "|" + rows["tau_bucket"].astype(str)
        for col in out.columns[: len(STATE_LEVELS) * len(TAU_LEVELS)]:
            out[col] = (cell == col).astype(float)
        for e in END_LEVELS:
            out[f"end:{e}"] = (rows["days_to_session_end_bucket"] == e).astype(float)
        for col in self.shared:
            values = rows[col].astype(float) if col in rows else 0.0
            if col in self.means:
                values = (values - self.means[col]) / self.sds[col]
            out[col] = values
        return out.to_numpy()


@dataclass
class HazardModel:
    spec: DesignSpec
    model: LogisticRegression
    C: float
    departments: list[str]
    validation: dict[str, Any]
    pace_medians: dict[tuple[str, str], float] = field(default_factory=dict)
    # Spread of the per-path speed multiplier exp(u), u ~ Normal(0, speed_sd), used by the
    # simulator (tuned on training-era sessions only).
    speed_sd: float = 0.0

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """P(advance on that sitting day) for each row."""
        return self.model.predict_proba(self.spec.matrix(rows))[:, 1]

    def coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_[0], index=self.spec.columns)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> HazardModel:
        return joblib.load(path)


def _fit(rows: pd.DataFrame, spec: DesignSpec, C: float) -> LogisticRegression:
    model = LogisticRegression(C=C, max_iter=MAX_ITER)
    model.fit(spec.matrix(rows), rows["y"].to_numpy())
    return model


def training_rows(cutoff: date = TRAIN_CUTOFF, data: Any = None) -> features.TrainingRows:
    """Hazard rows for training-session bills, built from data as of ``cutoff``."""
    built = features.build_training_rows(cutoff, data)
    built.rows = built.rows[built.rows["introduced_session_id"].isin(TRAIN_SESSIONS)]
    return built


def choose_c(rows: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Validation log-loss for each C (fit ≤ VALIDATION_SPLIT, score the later sessions),
    plus two reference points: a constant rate and a model with state × time cells only."""
    fit_rows = rows[rows["introduced_session_id"] <= VALIDATION_SPLIT]
    val_rows = rows[rows["introduced_session_id"] > VALIDATION_SPLIT]
    spec = DesignSpec.fit(fit_rows, feature_columns)
    y_val = val_rows["y"].to_numpy()

    scores = {}
    for C in C_GRID:
        model = _fit(fit_rows, spec, C)
        scores[C] = log_loss(y_val, model.predict_proba(spec.matrix(val_rows))[:, 1])
    best = min(scores, key=scores.get)

    constant = np.full(len(y_val), fit_rows["y"].mean())
    cells_only = DesignSpec(shared=[])
    cells_model = _fit(fit_rows, cells_only, best)
    return {
        "fit_sessions": sorted(int(s) for s in fit_rows["introduced_session_id"].unique()),
        "validation_sessions": sorted(int(s) for s in val_rows["introduced_session_id"].unique()),
        "fit_rows": len(fit_rows),
        "validation_rows": len(val_rows),
        "log_loss_by_C": {str(c): round(v, 5) for c, v in scores.items()},
        "best_C": best,
        "log_loss_best": round(scores[best], 5),
        "log_loss_constant_rate": round(log_loss(y_val, constant), 5),
        "log_loss_state_and_time_only": round(
            log_loss(y_val, cells_model.predict_proba(cells_only.matrix(val_rows))[:, 1]), 5
        ),
    }


def fit_hazard(built: features.TrainingRows) -> HazardModel:
    """Choose C by time-ordered validation, then refit on all training sessions."""
    validation = choose_c(built.rows, built.feature_columns)
    spec = DesignSpec.fit(built.rows, built.feature_columns)
    model = _fit(built.rows, spec, validation["best_C"])
    return HazardModel(
        spec, model, validation["best_C"], built.departments, validation, built.pace_medians
    )


def fit_hazard_with_c(
    built: features.TrainingRows, C: float, validation: dict[str, Any]
) -> HazardModel:
    """Refit with a C already chosen by validation (used for the live model on all data)."""
    spec = DesignSpec.fit(built.rows, built.feature_columns)
    return HazardModel(
        spec, _fit(built.rows, spec, C), C, built.departments, validation, built.pace_medians
    )


def lightgbm_validation(built: features.TrainingRows) -> dict[str, Any]:
    """Optional upgrade (MODEL_SPEC §4.1): the same rows and split with LightGBM."""
    import lightgbm as lgb

    rows = built.rows
    cols = ["state", "tau", "days_to_session_end_bucket", *built.feature_columns]

    def frame(df: pd.DataFrame) -> pd.DataFrame:
        out = df[cols].copy()
        out["state"] = pd.Categorical(out["state"], categories=STATE_LEVELS)
        out["days_to_session_end_bucket"] = pd.Categorical(
            out["days_to_session_end_bucket"], categories=[*END_LEVELS, "41+"]
        )
        return out

    fit_rows = rows[rows["introduced_session_id"] <= VALIDATION_SPLIT]
    val_rows = rows[rows["introduced_session_id"] > VALIDATION_SPLIT]
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0,
    )  # fmt: skip
    model.fit(frame(fit_rows), fit_rows["y"])
    pred = model.predict_proba(frame(val_rows))[:, 1]
    return {"log_loss": round(log_loss(val_rows["y"], pred), 5)}


# Plain-English names for the coefficient chart.
LABELS = {
    "end:0-5": "0-5 sitting days left in session",
    "end:6-15": "6-15 sitting days left in session",
    "end:16-40": "16-40 sitting days left in session",
    "originating_house_lords": "Started in the Lords",
    "introduced_months_into_session": "Introduced later in the session (+1 SD)",
    "n_stage_sittings_cum": "More stage sittings so far (+1 SD)",
    "committee_whole_house": "Committee of the whole House",
    "same_day_2r_3r": "2nd and 3rd reading on the same day",
    "carry_over": "Carried over from previous session",
    "parliament_age_years": "Older Parliament (+1 SD)",
    "log_pace": "Slower so far than typical (log pace +1)",
}


def label(column: str, departments: list[str] | None = None) -> str:
    if column in LABELS:
        return LABELS[column]
    if column.startswith("dept_"):
        names = {features.department_column(d): d for d in departments or []}
        return names.get(column, column[5:].replace("_", " "))
    return column


def plot_coefficients(hazard: HazardModel, path: Path) -> Path:
    """MODEL_SPEC §7 chart 4: shared drivers of speed, as odds ratios."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coefs = hazard.coefficients()
    shared = coefs[[c for c in coefs.index if not c.startswith("cell:")]]
    shared = shared.sort_values()
    odds = np.exp(shared)

    fig, ax = plt.subplots(figsize=(11, 0.38 * len(shared) + 1.8))
    colors = ["#c44e52" if v < 1 else "#55a868" for v in odds]
    ax.barh([label(c, hazard.departments) for c in shared.index], odds - 1, left=1, color=colors)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_xscale("log")
    ticks = [0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.5, 3.6)
    ax.set_xlabel("Effect on the daily chance of moving on (>1× faster, <1× slower)")
    ax.set_title(
        "What speeds bills up or slows them down\n"
        "(logistic advance hazard, trained on 2006-07 to 2017-19)"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
