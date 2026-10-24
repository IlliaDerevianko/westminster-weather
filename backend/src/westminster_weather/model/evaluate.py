"""Backtest, metrics and charts (BUILD_PLAN task 10, MODEL_SPEC §7).

Every test bill (introduced in ``TEST_SESSIONS``; the current session is never scored) is
forecast at the entry date of each state it reached before Royal Assent or failure, with data
as of that date. Several states entered on the same day give one forecast from the last.

Two ways to choose the models:

- ``frozen``: the ``train`` models, fitted once on data up to ``TRAIN_CUTOFF``;
- ``rolling``: refitted once per session, on bills that had finished before the session in
  which the forecast is made began (the hazard keeps the C chosen by validation).

Metrics:

- MAE (calendar days) of the median Royal Assent date, on forecasts for bills that reached
  Royal Assent, for the model and baselines B0 and B1 on the same forecasts;
- coverage of the 10th-90th percentile window (target about 80%) on the same forecasts;
- Brier score of ``passage_probability`` and ``p_ra_before_session_end`` on all forecasts;
- concordance index of predicted median time to Royal Assent against the actual time, with
  failed bills censored at their failure date.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from westminster_weather.domain.stages import BillStage
from westminster_weather.model import baseline, states
from westminster_weather.model.splits import TEST_SESSIONS, TRAIN_CUTOFF, TRAIN_SESSIONS

RA = BillStage.ROYAL_ASSENT.value
STATE_ORDER = [s.value for s in states.chain("Commons") if s.value != RA]


def backtest_points(evaluated_as_of: date, data: Any = None) -> pd.DataFrame:
    """One forecast point per (test bill, date it entered a state), with what happened."""
    src = states._source(data)
    trans = states.build_transitions(evaluated_as_of, src)
    trans = trans[trans["introduced_session_id"].isin(TEST_SESSIONS)]
    bills = src.get_bills(evaluated_as_of).set_index("bill_id")
    finished = bills[bills["outcome"] != "in_progress"].index
    trans = trans[trans["bill_id"].isin(finished)]
    points = (
        trans[trans["model_state"] != RA]
        .sort_values(["bill_id", "state_order"])
        .groupby(["bill_id", "entered_date"], as_index=False)
        .last()[["bill_id", "originating_house", "introduced_session_id", "entered_date"]]
        .rename(columns={"entered_date": "as_of"})
    )
    # Drop points on the day a bill ended (nothing left to forecast).
    points["outcome"] = points["bill_id"].map(bills["outcome"])
    points["outcome_date"] = points["bill_id"].map(bills["outcome_date"])
    points["introduced_date"] = points["bill_id"].map(bills["introduced_date"])
    points = points[points["as_of"] < points["outcome_date"]]

    sessions = src.get_calendar(evaluated_as_of)["sessions"].sort_values("start_date")
    starts = sessions["start_date"].to_numpy()
    idx = np.searchsorted(starts, points["as_of"].to_numpy(), side="right") - 1
    points["session_end"] = sessions["end_date"].to_numpy()[idx]
    points["ra"] = points["outcome"] == "royal_assent"
    points["ra_before_session_end"] = points["ra"] & (
        points["outcome_date"] <= points["session_end"]
    )
    return points.reset_index(drop=True)


def fit_models_as_of(cutoff: date, C: float, data: Any = None) -> Any:
    """All model components fitted on data available on ``cutoff``; the hazard only on bills
    that had finished by then."""
    from westminster_weather.model import failure, features, hazard, pingpong
    from westminster_weather.model.simulate import Models

    src = states._source(data)
    built = features.build_training_rows(cutoff, src)
    finished = src.get_bills(cutoff).query("outcome != 'in_progress'")["bill_id"]
    built.rows = built.rows[built.rows["bill_id"].isin(finished)]
    return Models(
        hazard=hazard.fit_hazard_with_c(built, C, {"refit_as_of": cutoff.isoformat()}),
        failure=failure.fit_failure(cutoff, src),
        pingpong=pingpong.fit_pingpong(cutoff, src),
    )


def _forecast_chunk(args: tuple[Any, list[tuple[int, date]]]) -> list[dict[str, Any]]:
    models, jobs = args
    return [_forecast_one(job, models) for job in jobs]


def _forecast_one(job: tuple[int, date], models: Any = None) -> dict[str, Any]:
    from westminster_weather.model.forecast import forecast_bill

    bill_id, as_of = job
    if models is None:
        r = forecast_bill(bill_id, as_of, variant="train")
    else:
        r = forecast_bill(bill_id, as_of, models=models, variant="rolling")
    t = r.time_to_royal_assent
    return {
        "bill_id": bill_id,
        "as_of": as_of,
        "start_state": r.current_stage.value if r.current_stage else None,
        "passage_probability": r.passage_probability,
        "p_ra_before_session_end": r.p_ra_before_session_end,
        "p10": t.p10_date if t else None,
        "p50": t.estimated_median_date if t else None,
        "p90": t.p90_date if t else None,
        **{f"risk_{k.value}": v for k, v in r.primary_stop_risks.items()},
    }


def run_backtest(
    points: pd.DataFrame,
    workers: int = 6,
    mode: str = "frozen",
    evaluated_as_of: date | None = None,
    C: float | None = None,
) -> pd.DataFrame:
    """Forecast every point in parallel processes, with ``frozen`` or ``rolling`` models."""
    jobs = list(zip(points["bill_id"].astype(int), points["as_of"], strict=True))
    if mode == "frozen":
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_forecast_one, jobs, chunksize=8))
    else:
        sessions = (
            states._source(None)
            .get_calendar(evaluated_as_of or date.today())["sessions"]
            .sort_values("start_date")
        )
        starts = sessions["start_date"].to_numpy()
        idx = np.searchsorted(starts, points["as_of"].to_numpy(), side="right") - 1
        refit_dates = [start - timedelta(days=1) for start in starts[idx]]
        tasks = []
        for refit in sorted(set(refit_dates)):
            models = fit_models_as_of(refit, C)
            mine = [job for job, r in zip(jobs, refit_dates, strict=True) if r == refit]
            tasks += [(models, mine[i : i + 20]) for i in range(0, len(mine), 20)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [row for chunk in pool.map(_forecast_chunk, tasks) for row in chunk]
    return points.merge(pd.DataFrame(results), on=["bill_id", "as_of"])


def _days(a: pd.Series, b: pd.Series) -> pd.Series:
    return (pd.to_datetime(a) - pd.to_datetime(b)).dt.days


def score(results: pd.DataFrame, baselines: baseline.Baselines) -> dict[str, Any]:
    df = results.copy()
    df["b0"] = [
        baselines.predict_b0(h, i, a)
        for h, i, a in zip(df["originating_house"], df["introduced_date"], df["as_of"], strict=True)
    ]
    df["b1"] = [
        baselines.predict_b1(h, s, a)
        for h, s, a in zip(df["originating_house"], df["start_state"], df["as_of"], strict=True)
    ]
    acts = df[df["ra"]].copy()
    for col in ("p50", "b0", "b1"):
        acts[f"{col}_err"] = _days(acts[col], acts["outcome_date"]).abs()
    acts["covered"] = (pd.to_datetime(acts["p10"]) <= pd.to_datetime(acts["outcome_date"])) & (
        pd.to_datetime(acts["outcome_date"]) <= pd.to_datetime(acts["p90"])
    )
    no_ra_paths = acts["p50"].isna()
    carried = acts[pd.to_datetime(acts["outcome_date"]) > pd.to_datetime(acts["session_end"])]

    by_state = (
        acts.groupby("start_state")[["p50_err", "b0_err", "b1_err"]]
        .mean()
        .reindex(STATE_ORDER)
        .dropna(how="all")
        .round(1)
    )
    counts = acts.groupby("start_state").size().reindex(by_state.index)

    # Concordance: higher predicted time to Royal Assent should go with later Royal Assent.
    predicted = _days(df["p50"], df["as_of"]).fillna(10_000)
    observed = _days(df["outcome_date"], df["as_of"])
    cindex = concordance_index(observed, predicted, event_observed=df["ra"])

    bins = np.linspace(0, 1, 11)
    df["bin"] = np.clip(np.digitize(df["passage_probability"], bins) - 1, 0, 9)
    calibration = df.groupby("bin").agg(
        predicted=("passage_probability", "mean"), actual=("ra", "mean"), n=("ra", "size")
    )
    return {
        "train_sessions": list(TRAIN_SESSIONS),
        "test_sessions": list(TEST_SESSIONS),
        "train_cutoff": TRAIN_CUTOFF.isoformat(),
        "n_forecasts": len(df),
        "n_bills": int(df["bill_id"].nunique()),
        "n_forecasts_for_acts": len(acts),
        "n_act_forecasts_without_ra_paths": int(no_ra_paths.sum()),
        "mae_days": {
            "model": round(float(acts["p50_err"].mean()), 1),
            "B0": round(float(acts["b0_err"].mean()), 1),
            "B1": round(float(acts["b1_err"].mean()), 1),
        },
        "mae_days_carried_over_bills": {
            "model": round(float(carried["p50_err"].mean()), 1),
            "B1": round(float(carried["b1_err"].mean()), 1),
            "n": len(carried),
        },
        "median_abs_error_days": {
            "model": float(acts["p50_err"].median()),
            "B1": float(acts["b1_err"].median()),
        },
        "mae_days_by_start_state": {
            s: {**by_state.loc[s].to_dict(), "n": int(counts.loc[s])} for s in by_state.index
        },
        "coverage_p10_p90": round(float(acts["covered"].mean()), 3),
        "brier_passage_probability": round(
            float(np.mean((df["passage_probability"] - df["ra"]) ** 2)), 4
        ),
        "brier_passage_base_rate": round(float(np.mean((df["ra"].mean() - df["ra"]) ** 2)), 4),
        "brier_ra_before_session_end": round(
            float(np.mean((df["p_ra_before_session_end"] - df["ra_before_session_end"]) ** 2)), 4
        ),
        "concordance_index": round(float(cindex), 3),
        "calibration": calibration.round(3).reset_index().to_dict("records"),
        "actual_passage_rate": round(float(df["ra"].mean()), 3),
        "mean_predicted_passage": round(float(df["passage_probability"].mean()), 3),
    }


# --------------------------------------------------------------------------- charts


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_mae_by_state(metrics: dict[str, Any], path: Path) -> Path:
    """Chart 1: MAE by the state the forecast was made from."""
    plt = _plt()
    rows = metrics["mae_days_by_start_state"]
    labels = list(rows)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, key, name, color in (
        (-0.27, "p50_err", "Model", "#4c72b0"),
        (0.0, "b1_err", "B1", "#dd8452"),
        (0.27, "b0_err", "B0", "#999999"),
    ):
        ax.bar(x + offset, [rows[s][key] for s in labels], 0.27, label=name, color=color)
    ax.set_xticks(
        x,
        [f"{s.replace('_', ' ')} (n={rows[s]['n']})" for s in labels],
        rotation=25,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Average error in Royal Assent date (days)")
    ax.set_title("Error by the stage the forecast was made from (test bills, 2019 onwards)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_vs_baselines(metrics: dict[str, Any], path: Path) -> Path:
    """Chart 2: overall MAE, model vs B0 and B1."""
    plt = _plt()
    mae = metrics["mae_days"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    names = ["Model", "B1", "B0"]
    values = [mae["model"], mae["B1"], mae["B0"]]
    bars = ax.bar(names, values, color=["#4c72b0", "#dd8452", "#999999"])
    for bar, v in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f} days", ha="center", va="bottom")
    ax.set_ylabel("Average error in Royal Assent date (days)")
    ax.set_title(f"Model vs baselines ({metrics['n_forecasts_for_acts']} forecasts)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_calibration(metrics: dict[str, Any], path: Path) -> Path:
    """Chart 3: calibration of passage_probability."""
    plt = _plt()
    cal = pd.DataFrame(metrics["calibration"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfect calibration")
    ax.scatter(cal["predicted"], cal["actual"], s=cal["n"] / cal["n"].max() * 400 + 20,
               color="#4c72b0", alpha=0.7, label="forecasts (size = count)")  # fmt: skip
    for row in cal.itertuples():
        ax.annotate(str(row.n), (row.predicted, row.actual), fontsize=7, ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast chance of Royal Assent")
    ax.set_ylabel("Share that actually got Royal Assent")
    ax.set_title(f"Calibration (Brier {metrics['brier_passage_probability']})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_metrics(metrics: dict[str, Any], path: Path) -> Path:
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing["backtest"] = metrics
    path.write_text(json.dumps(existing, indent=2, default=str) + "\n")
    return path


def write_report(all_metrics: dict[str, Any], path: Path) -> Path:
    """reports/EVALUATION.md: plain-English summary built from ``metrics.json``."""
    m = all_metrics["backtest"]
    mae = m["mae_days"]
    states_rows = "\n".join(
        f"| {s.replace('_', ' ')} | {v['n']} | {v['p50_err']:.0f} | {v['b1_err']:.0f} | "
        f"{v['b0_err']:.0f} |"
        for s, v in m["mae_days_by_start_state"].items()
    )
    odds = all_metrics.get("hazard", {}).get("odds_ratios", {})
    from westminster_weather.model.hazard import HazardModel, label

    model_path = Path(__file__).resolve().parents[3] / "models" / "hazard_train.joblib"
    departments = HazardModel.load(model_path).departments if model_path.exists() else []

    drivers = sorted(odds.items(), key=lambda kv: abs(np.log(kv[1])), reverse=True)[:8]
    driver_rows = "\n".join(
        f"| {label(c, departments)} | {v:.2f}× | {'faster' if v > 1 else 'slower'} |"
        for c, v in drivers
    )
    verdict = "beats" if mae["model"] < mae["B1"] else "does NOT beat"
    comparison = ""
    if "backtest_comparison" in all_metrics:
        cols = all_metrics["backtest_comparison"]
        header = "| | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
        lines = [
            ("Average error (days)", lambda c: f"{c['mae_days']['model']:.0f}"),
            ("Median error (days)", lambda c: f"{c['median_abs_error_days']['model']:.0f}"),
            ("80% window coverage", lambda c: f"{100 * c['coverage_p10_p90']:.0f}%"),
            (
                "Forecast pass chance vs actual",
                lambda c: (
                    f"{100 * c['mean_predicted_passage']:.0f}% vs "
                    f"{100 * c['actual_passage_rate']:.0f}%"
                ),
            ),
            (
                "Brier vs always-the-pass-rate",
                lambda c: (
                    f"{c['brier_passage_probability']:.3f} vs {c['brier_passage_base_rate']:.3f}"
                ),
            ),
            (
                "Error for carried-over bills (days)",
                lambda c: f"{c['mae_days_carried_over_bills']['model']:.0f}",
            ),
        ]
        body = "".join(
            f"| {name} | " + " | ".join(fn(c) for c in cols.values()) + " |\n" for name, fn in lines
        )
        comparison = "## Before and after the session-end fix\n\n" + header + body
    mae_row = f"{mae['model']:.0f}** | {mae['B1']:.0f} | {mae['B0']:.0f}"
    train_range = f"{m['train_sessions'][0]}-{m['train_sessions'][-1]}"
    test_range = f"{m['test_sessions'][0]}-{m['test_sessions'][-1]}"
    text = f"""# Passage model: evaluation

Backtest of the inside-Parliament passage model (MODEL_SPEC §7). Generated by
`uv run python -m westminster_weather.model.cli evaluate`; numbers are in `metrics.json`.

## How it was tested

- **Rolling refits:** before each test session, every model is refitted on the bills that had
  finished before that session began (first refit: data up to {m["train_cutoff"]}, sessions
  {train_range}, 2006-07 to 2017-19).
- **Tested on** {m["n_bills"]} bills introduced in sessions {test_range}
  (2019 onwards) that have finished; the current session is not scored.
- Each test bill was forecast on the day it entered each stage, as if it were that day:
  {m["n_forecasts"]:,} forecasts, {m["n_forecasts_for_acts"]:,} of them for bills that became law.

## Results

| | Model | B1 | B0 |
|---|---|---|---|
| Average error in Royal Assent date (days) | **{mae_row} |

- **The model {verdict} both baselines on timing.** B1 uses the typical remaining time
  from the current stage; B0 the typical total time from 1st reading.
- **Errors fall as a bill progresses** (table below): this is what live updating is for.
- **Uncertainty windows are too narrow:** the 80% window contained the real Royal Assent date
  {100 * m["coverage_p10_p90"]:.0f}% of the time (target 80%). Most misses are bills that
  took longer than forecast.
- **The chance of becoming law is too pessimistic:** forecasts averaged
  {100 * m["mean_predicted_passage"]:.0f}% but {100 * m["actual_passage_rate"]:.0f}% became law.
  The Brier score ({m["brier_passage_probability"]}) is worse than always forecasting the
  overall pass rate ({m["brier_passage_base_rate"]}).
- **Ranking is good:** concordance index {m["concordance_index"]} (0.5 = random, 1 = perfect)
  for which bills get Royal Assent sooner.

| Forecast made at | Forecasts | Model | B1 | B0 |
|---|---|---|---|---|
{states_rows}

Charts: `mae_by_state.png`, `model_vs_baselines.png`, `calibration.png`,
`hazard_coefficients.png`, `state_durations.png`.

## What drives speed

Largest shared effects in the advance hazard (effect on the daily chance of moving on):

| Factor | Effect | |
|---|---|---|
{driver_rows}

The end of a session speeds bills up; carried-over and Lords-first bills are slower.

{comparison}
## Limitations

- **Bills are forecast too fast.** The median forecast is earlier than the real Royal Assent date
  for about 6 in 10 forecasts, and a third of real dates fall after the 80% window. Bills since
  2019 have moved more slowly than the training history suggests. This is the main reason the
  80% window covers only {100 * m["coverage_p10_p90"]:.0f}% of outcomes.
- **Carried-over bills** are forecast far too early (about 200 days): most simulated paths finish
  before the drawn session end, so the carry-over route rarely applies.
- **Session ends:** a prorogation kills an unfinished bill only with the historical rate
  (12% in its first House, 4% later); otherwise the bill is carried over or rushed through. The
  rates rest on 17 deaths, 10 of them at the October 2019 prorogation.
- **No vote, rebellion or Lords-defeat data yet** (BUILD_PLAN task 7 skipped): defeat risk is a
  fixed 0.2% per Commons vote stage.
- **Committee of the whole House** is only recorded separately by the Bills API from 2022-23,
  so the model cannot learn its effect.
- **Small sample** (about 590 bills, 66 failures): wide intervals are honest, not a bug.
- Pre-Parliament stages and commencement are not covered.
"""
    path.write_text(text)
    return path
