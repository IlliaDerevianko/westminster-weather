"""Public forecast API (BUILD_PLAN task 9, MODEL_SPEC §5.1).

``forecast_bill(bill_id, as_of)`` reads the bill's position on ``as_of`` through the
DATA_CONTRACT §5 interface, runs the simulator and returns a ``BillForecastResponse`` with
per-state results.

Model sets never see data from after ``as_of``:

- ``live`` (fitted on all data up to its fit date) is used only for ``as_of`` on or after
  that date;
- ``train`` (fitted on data up to ``TRAIN_CUTOFF``) is used for earlier dates, e.g. for
  historical forecasts and backtests;
- dates before ``TRAIN_CUTOFF`` cannot be forecast without leakage and are refused.

``commencement_probability`` and ``time_to_commencement`` stay null until a commencement
model exists.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from westminster_weather.data import calendar
from westminster_weather.domain.stages import BillStage, LifecyclePhase, StopReason
from westminster_weather.model import states
from westminster_weather.model.departments import current_department
from westminster_weather.model.failure import FailureModel, carried_over_in_session
from westminster_weather.model.features import (
    COVARIATE_FEATURES,
    DAYS_PER_MONTH,
    days_since_first_reading,
    department_column,
    log_pace,
)
from westminster_weather.model.hazard import HazardModel
from westminster_weather.model.simulate import (
    N_PATHS,
    OUTCOME_REASON,
    ROYAL_ASSENT,
    Models,
    SimResult,
    StartState,
    simulate,
)
from westminster_weather.model.splits import TRAIN_CUTOFF
from westminster_weather.model.states import DataSource
from westminster_weather.schemas.bills import (
    BillForecastResponse,
    StageForecastSchema,
    TimelineQuantiles,
)

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
RA = BillStage.ROYAL_ASSENT.value
FAIL_FIELD = {
    StopReason.DEFEATED: "p_fail_defeated",
    StopReason.PAUSED_OR_WITHDRAWN: "p_fail_withdrawn",
    StopReason.SESSION_RUNS_OUT: "p_fail_session_end",
    StopReason.ELECTION_CALLED: "p_fail_dissolution",
}
FAIL_CAUSE_REASON = {
    "session_end": StopReason.SESSION_RUNS_OUT,
    "dissolution": StopReason.ELECTION_CALLED,
    "withdrawn": StopReason.PAUSED_OR_WITHDRAWN,
    "defeated": StopReason.DEFEATED,
}


class ForecastUnavailable(ValueError):
    """The bill cannot be forecast on that date (unknown bill, or no leakage-free model)."""


@cache
def load_models(variant: str, models_dir: Path = MODELS_DIR) -> Models:
    quantile_path = models_dir / f"quantile_{variant}.joblib"
    try:
        return Models(
            hazard=HazardModel.load(models_dir / f"hazard_{variant}.joblib"),
            failure=FailureModel.load(models_dir / f"failure_{variant}.joblib"),
            pingpong=joblib.load(models_dir / f"pingpong_{variant}.joblib"),
            quantile=joblib.load(quantile_path) if quantile_path.exists() else None,
        )
    except FileNotFoundError as exc:
        raise ForecastUnavailable(
            f"model set '{variant}' not found in {models_dir}; run `cli train` first"
        ) from exc


def choose_variant(as_of: date, models_dir: Path = MODELS_DIR) -> str:
    """The model set that uses no data from after ``as_of``."""
    live_fitted = load_models("live", models_dir).failure.fitted_as_of
    if live_fitted is not None and as_of >= live_fitted:
        return "live"
    if as_of >= TRAIN_CUTOFF:
        return "train"
    raise ForecastUnavailable(
        f"no model was trained only on data before {as_of}; the earliest forecast date is "
        f"{TRAIN_CUTOFF}"
    )


# --------------------------------------------------------------------------- start state


def shared_features(
    bill: pd.Series, as_of: date, src: Any, departments: list[str]
) -> dict[str, float]:
    """The bill's Tier 1 feature values on ``as_of`` (information up to and including
    ``as_of``), as the hazard model expects them."""
    cal = src.get_calendar(as_of)
    sessions = cal["sessions"].set_index("session_id")
    cov = src.get_covariates(as_of)
    cov = cov[(cov["bill_id"] == bill["bill_id"]) & (cov["valid_from"] <= as_of)]
    latest = cov.sort_values("valid_from").groupby("feature")["value"].last()
    parls = cal["parliaments"]
    first_meeting = parls.loc[parls["first_meeting_date"] <= as_of, "first_meeting_date"].max()
    dept = current_department(bill["department"])
    values = {
        "originating_house_lords": float(bill["originating_house"] == "Lords"),
        "introduced_months_into_session": (
            bill["introduced_date"] - sessions.loc[bill["introduced_session_id"], "start_date"]
        ).days
        / DAYS_PER_MONTH,
        "parliament_age_years": (as_of - first_meeting).days / 365.25,
        **{name: float(latest.get(name, 0.0)) for name in COVARIATE_FEATURES},
    }
    for d in departments:
        values[department_column(d)] = float(dept == d)
    return values


def current_log_pace(
    trans: pd.DataFrame, dates: np.ndarray, house: str, medians: dict[tuple[str, str], float]
) -> float:
    """Pace of the bill to its current state (frozen for the rest of the simulation)."""
    if not medians:
        return 0.0
    days = days_since_first_reading(trans, dates)
    last = trans.iloc[-1]
    return log_pace(days.iloc[-1], medians.get((house, last["model_state"])))


def _current_session(sessions: pd.DataFrame, as_of: date) -> pd.Series:
    return sessions[sessions["start_date"] <= as_of].sort_values("start_date").iloc[-1]


def _announced(cal: dict[str, pd.DataFrame], kinds: tuple[str, ...], as_of: date) -> date | None:
    announced = cal["announced"]
    if announced.empty:
        return None
    hits = announced[announced["date_type"].isin(kinds) & (announced["target_date"] >= as_of)]
    return None if hits.empty else hits["target_date"].min()


def start_state(
    bill_id: int, as_of: date, src: Any, hazard: HazardModel
) -> tuple[StartState | None, pd.Series, pd.DataFrame]:
    """The bill's position on ``as_of``. Returns ``None`` as the start state when the bill had
    already reached Royal Assent or failed by then."""
    bills = src.get_bills(as_of)
    match = bills[bills["bill_id"] == bill_id]
    if match.empty:
        raise ForecastUnavailable(f"bill {bill_id} was not introduced by {as_of}")
    bill = match.iloc[0]
    trans = states.build_transitions(as_of, src)
    trans = trans[trans["bill_id"] == bill_id].sort_values("state_order")
    if trans.empty:
        raise ForecastUnavailable(f"bill {bill_id} has no model states by {as_of}")
    last = trans.iloc[-1]
    if last["model_state"] == RA or last["exit_type"] == "fail":
        return None, bill, trans

    cal = src.get_calendar(as_of)
    dates = calendar.sitting_dates(cal["sitting_days"])
    session = _current_session(cal["sessions"], as_of)
    between_sessions = pd.notna(session["end_date"]) and session["end_date"] <= as_of
    session_days = (
        0
        if between_sessions
        else calendar.count_sitting_days(dates, session["start_date"] - timedelta(days=1), as_of)
    )
    parls = cal["parliaments"].sort_values("first_meeting_date")
    current = parls[parls["first_meeting_date"] <= as_of].iloc[-1]
    previous = parls[parls["first_meeting_date"] < current["first_meeting_date"]].iloc[-1]
    chain = [s.value for s in states.chain(bill["originating_house"])]
    start = StartState(
        bill_id=bill_id,
        as_of=as_of,
        chain=chain,
        k0=chain.index(last["model_state"]),
        tau0=calendar.count_sitting_days(dates, last["entered_date"], as_of),
        shared_features={
            **shared_features(bill, as_of, src, hazard.departments),
            "log_pace": current_log_pace(
                trans, dates, bill["originating_house"], getattr(hazard, "pace_medians", {})
            ),
        },
        carry_over_active=(not between_sessions)
        and carried_over_in_session(bill["carry_over_date"], session["start_date"]),
        session_days_so_far=session_days,
        announced_session_end=_announced(cal, ("prorogation", "dissolution"), as_of),
        announced_dissolution=_announced(cal, ("dissolution",), as_of),
        election_date=previous["election_date"],
        parliament_first_meeting=current["first_meeting_date"],
        parliaments=cal["parliaments"],
    )
    return start, bill, trans


# --------------------------------------------------------------------------- summaries


def _quantiles(as_of: date, dates: np.ndarray) -> TimelineQuantiles | None:
    if len(dates) == 0:
        return None
    offsets = (dates - np.datetime64(as_of, "D")).astype(int)
    p10, p50, p90 = (int(np.percentile(offsets, q, method="lower")) for q in (10, 50, 90))
    return TimelineQuantiles(
        p10_days=p10,
        p50_days=p50,
        p90_days=p90,
        estimated_median_date=as_of + timedelta(days=p50),
        p10_date=as_of + timedelta(days=p10),
        p90_date=as_of + timedelta(days=p90),
    )


def summarise(result: SimResult, entered_current: date) -> dict[str, Any]:
    """Every MODEL_SPEC §5.1 output from the simulated paths."""
    start = result.start
    out = result.outcome
    ra = out == ROYAL_ASSENT
    risks = {reason: float(np.mean(out == code)) for code, reason in OUTCOME_REASON.items()}
    stages = []
    for k in range(start.k0, len(start.chain)):
        reached = result.entry_day[:, k] >= 0
        if k == start.k0:
            entry = entered_current
        elif reached.any():
            entry = pd.Timestamp(
                result.date_of(np.percentile(result.entry_day[reached, k], 50, method="lower"))
            ).date()
        else:
            entry = None
        fields = {f: 0.0 for f in FAIL_FIELD.values()}
        for code, reason in OUTCOME_REASON.items():
            fields[FAIL_FIELD[reason]] = float(np.mean((out == code) & (result.fail_state == k)))
        stages.append(
            StageForecastSchema(
                model_state=BillStage(start.chain[k]),
                p_reach=float(reached.mean()),
                entry_date_p50=entry,
                **fields,
            )
        )
    return {
        "passage_probability": float(ra.mean()),
        # Royal Assent in the current session (an extended, rushed-through session counts).
        "p_ra_before_session_end": float(np.mean(ra & ~result.carried)),
        "time_to_royal_assent": _quantiles(start.as_of, result.date_of(result.end_day[ra])),
        "primary_stop_risks": risks,
        "stages": stages,
    }


def forecast_bill(
    bill_id: int,
    as_of: date,
    data: DataSource | None = None,
    variant: str | None = None,
    n: int = N_PATHS,
    models_dir: Path = MODELS_DIR,
    models: Models | None = None,
) -> BillForecastResponse:
    """Forecast one bill as of ``as_of`` (MODEL_SPEC §5). ``models`` overrides the saved model
    sets (the rolling backtest passes models fitted on data before ``as_of``)."""
    src = states._source(data)
    if models is None:
        variant = variant or choose_variant(as_of, models_dir)
        models = load_models(variant, models_dir)
    else:
        variant = variant or "custom"
    start, bill, trans = start_state(bill_id, as_of, src, models.hazard)
    common = {
        "bill_id": bill_id,
        "short_title": bill["short_title"],
        "as_of": as_of,
        "model_variant": variant,
    }
    if start is None:
        return _resolved(trans.iloc[-1], as_of, common)

    if variant == "live":
        start.scheduled_entry = scheduled_entry(start, bill_id, as_of, src)
    result = simulate(start, models, src.get_calendar(as_of)["sitting_days"], n=n)
    summary = summarise(result, trans.iloc[-1]["entered_date"])
    if models.quantile is not None and summary["time_to_royal_assent"] is not None:
        summary["time_to_royal_assent"] = _average_with_quantile_model(
            summary["time_to_royal_assent"], models.quantile, start, bill
        )
    return BillForecastResponse(
        **common,
        current_phase=LifecyclePhase.INSIDE_PARLIAMENT,
        current_stage=BillStage(start.chain[start.k0]),
        n_paths=n,
        **summary,
    )


def scheduled_entry(
    start: StartState, bill_id: int, as_of: date, src: Any
) -> tuple[int, date] | None:
    """The first later stage of this bill with a future-dated scheduled sitting (live only)."""
    from westminster_weather.data.stage_map import STAGE_ID_TO_STATE

    getter = getattr(src, "get_scheduled_sittings", None)
    if getter is None:
        return None
    scheduled = getter(as_of)
    if scheduled.empty:
        return None
    mine = scheduled[(scheduled["bill_id"] == bill_id) & (scheduled["event_date"] > as_of)]
    later = []
    for stage_id, when in zip(mine["api_stage_id"], mine["event_date"], strict=True):
        state = STAGE_ID_TO_STATE.get(int(stage_id))
        if state is not None and state.value in start.chain:
            k = start.chain.index(state.value)
            if start.k0 < k < start.chain.index(RA):
                later.append((when, k))
    if not later:
        return None
    when, k = min(later)
    return k, when


def _average_with_quantile_model(
    sim: TimelineQuantiles, model: Any, start: StartState, bill: pd.Series
) -> TimelineQuantiles:
    """Average the simulator's days to Royal Assent with the quantile model's (step 5)."""
    from westminster_weather.model.quantile import start_features

    q = model.predict_days(pd.DataFrame([start_features(start, bill)]))[0]
    days = [
        round((s + float(v)) / 2)
        for s, v in zip((sim.p10_days, sim.p50_days, sim.p90_days), q, strict=True)
    ]
    p10, p50, p90 = sorted(days)
    as_of = start.as_of
    return TimelineQuantiles(
        p10_days=p10, p50_days=p50, p90_days=p90,
        estimated_median_date=as_of + timedelta(days=p50),
        p10_date=as_of + timedelta(days=p10), p90_date=as_of + timedelta(days=p90),
    )  # fmt: skip


def _resolved(last: pd.Series, as_of: date, common: dict[str, Any]) -> BillForecastResponse:
    """A bill that already reached Royal Assent or failed by ``as_of``: certain outcome."""
    if last["model_state"] == RA:
        days = (last["entered_date"] - as_of).days
        timeline = TimelineQuantiles(
            p10_days=days, p50_days=days, p90_days=days,
            estimated_median_date=last["entered_date"],
            p10_date=last["entered_date"], p90_date=last["entered_date"],
        )  # fmt: skip
        return BillForecastResponse(
            **common,
            current_phase=LifecyclePhase.AFTER_ROYAL_ASSENT,
            current_stage=BillStage.ROYAL_ASSENT,
            passage_probability=1.0,
            p_ra_before_session_end=1.0,
            time_to_royal_assent=timeline,
            primary_stop_risks={r: 0.0 for r in OUTCOME_REASON.values()},
            n_paths=0,
        )
    reason = FAIL_CAUSE_REASON[last["fail_cause"]]
    return BillForecastResponse(
        **common,
        current_phase=LifecyclePhase.INSIDE_PARLIAMENT,
        current_stage=BillStage(last["model_state"]),
        passage_probability=0.0,
        p_ra_before_session_end=0.0,
        primary_stop_risks={r: float(r == reason) for r in OUTCOME_REASON.values()},
        n_paths=0,
    )


def plot_examples(rows: list[tuple[BillForecastResponse, pd.Series]], path: Path) -> Path:
    """Forecast Royal Assent window and passage chance next to what actually happened."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    fig, ax = plt.subplots(figsize=(11, 1.6 * len(rows) + 1.5))
    for i, (resp, actual) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([resp.as_of], [y], marker="|", ms=18, color="black")
        t = resp.time_to_royal_assent
        if t:
            ax.plot(
                [t.p10_date, t.p90_date],
                [y, y],
                lw=8,
                color="#4c72b0",
                alpha=0.35,
                solid_capstyle="butt",
                label="80% forecast window" if i == 0 else None,
            )
            ax.plot(
                [t.estimated_median_date],
                [y],
                "o",
                color="#4c72b0",
                label="median forecast" if i == 0 else None,
            )
        happened = actual["outcome_date"]
        if actual["outcome"] == "royal_assent":
            ax.plot(
                [happened],
                [y],
                "*",
                ms=16,
                color="#55a868",
                label="actual Royal Assent" if i == 0 else None,
            )
        else:
            ax.plot([happened], [y], "X", ms=12, color="#c44e52", label="actually failed")
        stage = resp.current_stage.value.replace("_", " ")
        ax.text(
            0.01,
            y + 0.3,
            f"forecast on {resp.as_of} at {stage}: "
            f"{100 * resp.passage_probability:.0f}% chance of Royal Assent",
            fontsize=9,
            transform=blended_transform_factory(ax.transAxes, ax.transData),
        )
    ax.set_yticks(range(len(rows)), [r.short_title for r, _ in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_title("Historical forecasts (models trained on 2006-07 to 2017-19) vs what happened")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
