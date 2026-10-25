"""Calibration of the outlook's deadline probabilities on the rolling backtest.

For every backtest forecast point (bill, date it entered a stage), the paths are re-simulated
with models refitted on data before that session (as in ``evaluate.run_backtest``), and the
chance of Royal Assent before three deadlines is counted: the next Christmas recess, the next
summer recess (Commons rising dates from the recorded calendar) and the end of the session.
Outcome: the bill really got Royal Assent on or before that date.

- Training era (sessions 27-30, ``TUNING_SESSIONS``): used only to fit the recalibration.
- Test era (sessions 34-39): calibration table, Brier score and the before/after comparison.
  Nothing is tuned on it.

Isotonic recalibration is fitted on training-era forecasts and saved to
``reports/deadline_calibration.json`` only if a well-populated test bin is off by more than 10
points; ``outlook.calibrate`` then applies it. One curve for the session-end deadline and one
for date deadlines (recesses, quarter ends, custom dates): chosen over a single pooled curve by
cross-validation inside the training era (fit sessions 27-28, check 29-30 and back; Brier
0.110 vs 0.124 pooled vs 0.137 none), not on test data.

Run from ``backend/``: uv run python -m westminster_weather.model.outlook_calibration
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from westminster_weather.model import states
from westminster_weather.model.evaluate import backtest_points, fit_models_as_of
from westminster_weather.model.splits import TEST_SESSIONS, TUNING_SESSIONS

REPORTS = Path(__file__).resolve().parents[3] / "reports"
POINTS_CSV = REPORTS / "deadline_calibration_points.csv"
METRICS_JSON = REPORTS / "deadline_calibration.json"
CHART = REPORTS / "deadline_calibration.png"
N_PATHS = 4000
WORKERS = 6
BINS = np.linspace(0, 1, 11)
WELL_POPULATED = 30  # forecasts in a bin before it counts towards "poorly calibrated"
MAX_GAP = 0.10
DEADLINES = ("christmas", "summer", "session_end")


@cache
def _commons_sitting() -> pd.DatetimeIndex:
    from westminster_weather.model.outlook import _sitting_dates

    return _sitting_dates(states._source(None).get_calendar(date.today())["sitting_days"])


def _next_actual(kind: str, as_of: date) -> date | None:
    from westminster_weather.model.outlook import recess_start

    for year in (as_of.year, as_of.year + 1):
        d = recess_start(kind, year, _commons_sitting())
        if d is not None and d > as_of:
            return d
    return None


def _one(models: Any, job: tuple[int, date, date | None]) -> dict[str, Any] | None:
    from westminster_weather.model.forecast import run_paths
    from westminster_weather.model.outlook import Distribution

    bill_id, as_of, session_end = job
    run = run_paths(bill_id, as_of, models=models, variant="rolling", n=N_PATHS)
    if run.start is None:
        return None
    dist = Distribution.from_result(run.result)
    row: dict[str, Any] = {"bill_id": bill_id, "as_of": as_of}
    for kind, d in (
        ("christmas", _next_actual("christmas", as_of)),
        ("summer", _next_actual("summer", as_of)),
        ("session_end", session_end),
    ):
        row[f"{kind}_date"] = d
        row[f"p_{kind}"] = dist.p_by(d) if d is not None else np.nan
    return row


def _chunk(args: tuple[Any, list[tuple[int, date, date | None]]]) -> list[dict[str, Any]]:
    models, jobs = args
    return [r for job in jobs if (r := _one(models, job)) is not None]


def simulate_points(sessions: tuple[int, ...], C: float, speed_sd: float) -> pd.DataFrame:
    today = date.today()
    pts = backtest_points(today, sessions=sessions)
    cal = states._source(None).get_calendar(today)["sessions"].sort_values("start_date")
    starts = cal["start_date"].to_numpy()
    idx = np.searchsorted(starts, pts["as_of"].to_numpy(), side="right") - 1
    pts["refit"] = [s - timedelta(days=1) for s in starts[idx]]
    tasks = []
    for refit, grp in pts.groupby("refit"):
        models = fit_models_as_of(refit, C, speed_sd=speed_sd)
        jobs = [
            (int(b), a, (None if pd.isna(e) else pd.Timestamp(e).date()))
            for b, a, e in zip(grp["bill_id"], grp["as_of"], grp["session_end"], strict=True)
        ]
        tasks += [(models, jobs[i : i + 20]) for i in range(0, len(jobs), 20)]
        print(f"  refit {refit}: {len(jobs)} points", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = [r for chunk in pool.map(_chunk, tasks) for r in chunk]
    out = pts.merge(pd.DataFrame(rows), on=["bill_id", "as_of"])
    for kind in DEADLINES:
        d = pd.to_datetime(out[f"{kind}_date"])
        out[f"y_{kind}"] = (out["outcome"] == "royal_assent") & (
            pd.to_datetime(out["outcome_date"]) <= d
        )
    out["era"] = "train" if sessions == TUNING_SESSIONS else "test"
    return out


def long_form(points: pd.DataFrame) -> pd.DataFrame:
    """One row per (forecast, deadline) with a known deadline date."""
    parts = []
    for kind in DEADLINES:
        p = points[[f"p_{kind}", f"y_{kind}", "era", "bill_id", "as_of"]].dropna(
            subset=[f"p_{kind}"]
        )
        parts.append(p.rename(columns={f"p_{kind}": "p", f"y_{kind}": "y"}).assign(deadline=kind))
    df = pd.concat(parts, ignore_index=True)
    df["y"] = df["y"].astype(int)
    return df


def group_of(deadline: pd.Series) -> pd.Series:
    """Recalibration curve per deadline: ``session_end`` or ``dates``."""
    return deadline.where(deadline == "session_end", "dates")


def table(df: pd.DataFrame, col: str = "p") -> pd.DataFrame:
    b = pd.cut(
        df[col],
        BINS,
        include_lowest=True,
        right=False,
        labels=[
            f"{int(lo * 100)}-{int(hi * 100)}%" for lo, hi in zip(BINS[:-1], BINS[1:], strict=True)
        ],
    )
    t = df.groupby(b, observed=True).agg(
        n=("y", "size"), predicted=(col, "mean"), actual=("y", "mean")
    )
    t["gap"] = t["actual"] - t["predicted"]
    return t


def brier(p: pd.Series, y: pd.Series) -> float:
    return float(np.mean((p - y) ** 2))


def run() -> dict[str, Any]:
    from westminster_weather.model import hazard as hz
    from westminster_weather.model.forecast import MODELS_DIR

    haz = hz.HazardModel.load(MODELS_DIR / "hazard_train.joblib")
    speed_sd = float(getattr(haz, "speed_sd", 0.0) or 0.0)
    if POINTS_CSV.exists():
        points = pd.read_csv(POINTS_CSV)
    else:
        print("training era (sessions 27-30)", flush=True)
        train = simulate_points(TUNING_SESSIONS, haz.C, speed_sd)
        print("test era (sessions 34-39)", flush=True)
        test = simulate_points(TEST_SESSIONS, haz.C, speed_sd)
        points = pd.concat([train, test], ignore_index=True)
        REPORTS.mkdir(parents=True, exist_ok=True)
        points.to_csv(POINTS_CSV, index=False)
    df = long_form(points)
    tr, te = df[df["era"] == "train"], df[df["era"] == "test"]

    base_rate = tr.groupby("deadline")["y"].mean()  # historical average, training era only
    te = te.assign(base=te["deadline"].map(base_rate))
    t_before = table(te)
    poor = bool(((t_before["n"] >= WELL_POPULATED) & (t_before["gap"].abs() > MAX_GAP)).any())

    metrics: dict[str, Any] = {
        "n_forecasts": {"train": int(len(tr)), "test": int(len(te))},
        "n_bills": {"train": int(tr["bill_id"].nunique()), "test": int(te["bill_id"].nunique())},
        "base_rate_train": {k: round(float(v), 3) for k, v in base_rate.items()},
        "brier_test": {
            "model": round(brier(te["p"], te["y"]), 4),
            "baseline_historical_average": round(brier(te["base"], te["y"]), 4),
            **{f"model_{k}": round(brier(g["p"], g["y"]), 4) for k, g in te.groupby("deadline")},
        },
        "table_test": t_before.round(3).reset_index(names="bin").to_dict("records"),
        "poorly_calibrated": poor,
        "recalibrated": False,
    }
    if poor:
        from sklearn.isotonic import IsotonicRegression

        curves, p_cal = {}, np.empty(len(te))
        for grp in ("session_end", "dates"):
            fit = tr[group_of(tr["deadline"]) == grp]
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(fit["p"], fit["y"])
            curves[grp] = {
                "x": [float(v) for v in iso.X_thresholds_],
                "y": [float(v) for v in iso.y_thresholds_],
            }
            m = (group_of(te["deadline"]) == grp).to_numpy()
            p_cal[m] = iso.predict(te["p"].to_numpy()[m])
        te = te.assign(p_cal=p_cal)
        t_after = table(te, "p_cal")
        metrics.update(
            recalibrated=True,
            isotonic=curves,
            table_test_after=t_after.round(3).reset_index(names="bin").to_dict("records"),
        )
        metrics["brier_test"]["model_recalibrated"] = round(brier(te["p_cal"], te["y"]), 4)
    else:
        t_after = None
    metrics["headline"] = headline(t_after if t_after is not None else t_before)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=str))
    chart(t_before, t_after, metrics)
    return metrics


def headline(t: pd.DataFrame) -> str:
    row = t.loc["70-80%"] if "70-80%" in t.index else None
    if row is None or row["n"] == 0:
        return "Too few forecasts near 70% to state a headline."
    return (
        f"When we said about 70% (70-80%, {int(row['n'])} forecasts), it happened "
        f"{round(100 * row['actual'])}% of the time."
    )


def chart(before: pd.DataFrame, after: pd.DataFrame | None, metrics: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from westminster_weather.analysis.stakeholder_chart import (
        BASELINE_COLOUR,
        GRID,
        INK,
        INK_SOFT,
        MODEL_COLOUR,
        SURFACE,
        V1_COLOUR,
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.5, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=BASELINE_COLOUR, lw=1.2, ls="--", label="perfectly calibrated")
    series = [
        (
            before,
            V1_COLOUR if after is not None else MODEL_COLOUR,
            "model" if after is None else "model, before recalibration",
        )
    ]
    if after is not None:
        series.append(
            (after, MODEL_COLOUR, "after isotonic recalibration (fitted on 2014-19 only)")
        )
    for t, colour, label in series:
        ax.plot(
            t["predicted"],
            t["actual"],
            color=colour,
            lw=2.5,
            marker="o",
            ms=7,
            markeredgecolor=SURFACE,
            label=label,
        )
    for _, r in before.iterrows():
        ax.annotate(
            f"n={int(r['n'])}",
            (r["predicted"], r["actual"]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=8,
            color=INK_SOFT,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xlabel("Predicted chance of Royal Assent before the deadline", color=INK_SOFT)
    ax.set_ylabel("Share that really did", color=INK_SOFT)
    ax.grid(color=GRID, lw=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK_SOFT)
    b = metrics["brier_test"]
    fig.text(
        0.06,
        0.95,
        "Deadline forecasts: predicted vs what happened",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.915,
        f"Test bills 2019-2026, {metrics['n_forecasts']['test']} deadline forecasts "
        "(next Christmas recess, next summer recess, session end).",
        fontsize=9.5,
        color=INK_SOFT,
    )
    extra = f"; after recalibration {b['model_recalibrated']}" if "model_recalibrated" in b else ""
    fig.text(
        0.06,
        0.02,
        f"Brier score: model {b['model']}{extra}; always-the-historical-"
        f"average baseline {b['baseline_historical_average']} (lower is better).",
        fontsize=9,
        color=INK_SOFT,
    )
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12)
    fig.savefig(CHART, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return CHART


def main() -> None:
    m = run()
    print(json.dumps({k: v for k, v in m.items() if k != "isotonic"}, indent=2, default=str))
    print(f"\nchart: {CHART}")


if __name__ == "__main__":
    main()
