"""Baselines the passage model has to beat (BUILD_PLAN task 5, MODEL_SPEC §7).

- **B0:** median calendar days from 1st reading to Royal Assent, by originating House.
  Forecast Royal Assent date = 1st reading + that median (never before ``as_of``).
- **B1:** for a bill in state *k*, the median calendar days from entering *k* to Royal Assent
  among training bills, by originating House and state (the chain order differs by House).
  Forecast = ``as_of`` + that median.

Both are fitted on training sessions as of ``TRAIN_CUTOFF`` and scored on test bills that
reached Royal Assent, with one forecast per state the bill reached (``as_of`` = the state's
entry date), as in the MODEL_SPEC §7 backtest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from westminster_weather.domain.stages import BillStage
from westminster_weather.model import states
from westminster_weather.model.splits import TEST_SESSIONS, TRAIN_CUTOFF, TRAIN_SESSIONS
from westminster_weather.model.states import DataSource

# Fall back to the state's median over both Houses when a (House, state) cell has fewer
# training bills than this.
MIN_BILLS_PER_CELL = 5

RA = BillStage.ROYAL_ASSENT.value


def _acts_with_entries(trans: pd.DataFrame) -> pd.DataFrame:
    """Transition rows of bills that reached Royal Assent, with their Royal Assent date."""
    ra = trans[trans["model_state"] == RA][["bill_id", "entered_date"]].rename(
        columns={"entered_date": "ra_date"}
    )
    return trans.merge(ra, on="bill_id")


@dataclass
class Baselines:
    b0_days: dict[str, float]  # originating House -> median days 1st reading to RA
    b1_days: dict[tuple[str, str], float]  # (House, state) -> median days entry to RA
    b1_state_days: dict[str, float]  # state -> median over both Houses (fallback)
    n_train_acts: int

    def predict_b0(self, house: str, introduced: date, as_of: date) -> date:
        return max(as_of, introduced + timedelta(days=round(self.b0_days[house])))

    def predict_b1(self, house: str, state: str, as_of: date) -> date:
        days = self.b1_days.get((house, state), self.b1_state_days[state])
        return as_of + timedelta(days=round(days))


def fit(data: DataSource | None = None, cutoff: date = TRAIN_CUTOFF) -> Baselines:
    """Fit B0 and B1 on bills from training sessions, as known on ``cutoff``."""
    trans = states.build_transitions(cutoff, data)
    trans = trans[trans["introduced_session_id"].isin(TRAIN_SESSIONS)]
    acts = _acts_with_entries(trans)
    acts["days_to_ra"] = (
        pd.to_datetime(acts["ra_date"]) - pd.to_datetime(acts["entered_date"])
    ).dt.days

    first = acts.sort_values("state_order").groupby("bill_id").head(1)
    b0 = first.groupby("originating_house")["days_to_ra"].median().to_dict()

    origins = acts[acts["model_state"] != RA]
    cells = origins.groupby(["originating_house", "model_state"])["days_to_ra"].agg(
        ["median", "size"]
    )
    b1 = {k: float(v["median"]) for k, v in cells.iterrows() if v["size"] >= MIN_BILLS_PER_CELL}
    b1_state = origins.groupby("model_state")["days_to_ra"].median().to_dict()
    return Baselines(b0, b1, b1_state, int(first["bill_id"].nunique()))


def forecast_points(as_of: date, data: DataSource | None = None) -> pd.DataFrame:
    """One forecast point per test bill per state it reached before Royal Assent, for bills
    that reached Royal Assent by ``as_of`` (the ones MAE is scored on)."""
    src = states._source(data)
    trans = states.build_transitions(as_of, src)
    trans = trans[trans["introduced_session_id"].isin(TEST_SESSIONS)]
    acts = _acts_with_entries(trans)
    points = acts[acts["model_state"] != RA]
    bills = src.get_bills(as_of).set_index("bill_id")["introduced_date"]
    return points.assign(
        introduced_date=points["bill_id"].map(bills),
        forecast_as_of=points["entered_date"],
    )[
        [
            "bill_id",
            "originating_house",
            "introduced_session_id",
            "introduced_date",
            "model_state",
            "forecast_as_of",
            "ra_date",
        ]
    ].reset_index(drop=True)


def evaluate(
    baselines: Baselines, as_of: date, data: DataSource | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Absolute errors (calendar days) of B0 and B1 on every test forecast point."""
    points = forecast_points(as_of, data)
    points["b0_pred"] = [
        baselines.predict_b0(h, i, a)
        for h, i, a in zip(
            points["originating_house"], points["introduced_date"], points["forecast_as_of"],
            strict=True,
        )
    ]  # fmt: skip
    points["b1_pred"] = [
        baselines.predict_b1(h, s, a)
        for h, s, a in zip(
            points["originating_house"], points["model_state"], points["forecast_as_of"],
            strict=True,
        )
    ]  # fmt: skip
    for b in ("b0", "b1"):
        points[f"{b}_abs_error"] = (
            pd.to_datetime(points[f"{b}_pred"]) - pd.to_datetime(points["ra_date"])
        ).dt.days.abs()

    order = [s.value for s in states.chain("Commons") if s.value != RA]
    by_state = (
        points.groupby("model_state")[["b0_abs_error", "b1_abs_error"]]
        .mean()
        .reindex(order)
        .dropna(how="all")
    )
    metrics = {
        "train_sessions": list(TRAIN_SESSIONS),
        "test_sessions": list(TEST_SESSIONS),
        "train_cutoff": TRAIN_CUTOFF.isoformat(),
        "evaluated_as_of": as_of.isoformat(),
        "n_train_acts": baselines.n_train_acts,
        "n_test_bills": int(points["bill_id"].nunique()),
        "n_forecasts": len(points),
        "B0": {
            "description": "median days 1st reading -> Royal Assent, by originating House",
            "median_days": {k: float(v) for k, v in baselines.b0_days.items()},
            "mae_days": round(float(points["b0_abs_error"].mean()), 1),
            "mae_days_by_state": by_state["b0_abs_error"].round(1).to_dict(),
        },
        "B1": {
            "description": "median days from entering the current state -> Royal Assent, "
            "by originating House and state",
            "mae_days": round(float(points["b1_abs_error"].mean()), 1),
            "mae_days_by_state": by_state["b1_abs_error"].round(1).to_dict(),
        },
    }
    return points, metrics


def save_metrics(metrics: dict[str, Any], path: Path) -> Path:
    """Write ``metrics`` under ``baselines`` in ``reports/metrics.json``, keeping other keys."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing["baselines"] = metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n")
    return path


def plot_mae_by_state(metrics: dict[str, Any], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    b0 = metrics["B0"]["mae_days_by_state"]
    b1 = metrics["B1"]["mae_days_by_state"]
    labels = list(b0)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - 0.2,
        [b0[s] for s in labels],
        0.4,
        label=f"B0 (overall {metrics['B0']['mae_days']} days)",
    )
    ax.bar(
        x + 0.2,
        [b1[s] for s in labels],
        0.4,
        label=f"B1 (overall {metrics['B1']['mae_days']} days)",
    )
    ax.set_xticks(x, [s.replace("_", " ") for s in labels], rotation=30, ha="right")
    ax.set_ylabel("Average error in Royal Assent date (calendar days)")
    ax.set_title(
        f"Baseline errors on test bills (sessions {metrics['test_sessions'][0]}-"
        f"{metrics['test_sessions'][-1]}), by stage the forecast was made from"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
