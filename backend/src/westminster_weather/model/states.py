"""Model states and observed transitions (BUILD_PLAN task 3, MODEL_SPEC §2).

The stage mapping, state chains, skipped-state rule and failure labels live on the data side
(``data/stage_map.py`` and ``get_transitions``), so this module reads transitions only through
the DATA_CONTRACT §5 interface and adds model-side labels (``BillStage``, ``StopReason``),
the MODEL_SPEC population filter, consistency checks and a sanity report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import pandas as pd

from westminster_weather.data import interface
from westminster_weather.data.stage_map import STAGE_ID_TO_STATE, chain
from westminster_weather.domain.stages import BillStage, StopReason

__all__ = [
    "STAGE_ID_TO_STATE",
    "FAIL_CAUSE_TO_STOP_REASON",
    "MIN_SESSION_ID",
    "build_transitions",
    "chain",
    "check_transitions",
    "final_states",
    "sanity_report",
]

# MODEL_SPEC §1: sessions below 20 contain 0-5 bills and are excluded.
MIN_SESSION_ID = 20

FAIL_CAUSE_TO_STOP_REASON: dict[str, StopReason] = {
    "session_end": StopReason.SESSION_RUNS_OUT,
    "dissolution": StopReason.ELECTION_CALLED,
    "withdrawn": StopReason.PAUSED_OR_WITHDRAWN,
    "defeated": StopReason.DEFEATED,
}


class DataSource(Protocol):
    """Anything exposing the DATA_CONTRACT §5 functions (the ``interface`` module, or a
    ``DatabaseInterface`` built from in-memory tables in tests)."""

    def get_bills(self, as_of: date) -> pd.DataFrame: ...
    def get_transitions(self, as_of: date) -> pd.DataFrame: ...
    def get_calendar(self, as_of: date) -> dict[str, pd.DataFrame]: ...


def _source(data: DataSource | ModuleType | None) -> Any:
    return interface if data is None else data


def build_transitions(as_of: date, data: DataSource | None = None) -> pd.DataFrame:
    """Entry and exit per state per bill as of ``as_of``, for bills in the model population.

    Columns from ``get_transitions`` plus ``state`` (``BillStage``), ``stop_reason``
    (``StopReason`` value on failed rows), ``originating_house``, ``introduced_session_id``
    and ``is_last`` (the bill's latest state). The failure stage is the state the bill was in
    when it failed.
    """
    src = _source(data)
    bills = src.get_bills(as_of)
    bills = bills[bills["introduced_session_id"] >= MIN_SESSION_ID]
    trans = src.get_transitions(as_of)
    trans = trans.merge(
        bills[["bill_id", "originating_house", "introduced_session_id"]], on="bill_id"
    )
    trans["state"] = trans["model_state"].map(BillStage)
    trans["stop_reason"] = trans["fail_cause"].map(
        lambda c: FAIL_CAUSE_TO_STOP_REASON[c].value if isinstance(c, str) else None
    )
    trans = trans.sort_values(["bill_id", "state_order"]).reset_index(drop=True)
    trans["is_last"] = ~trans["bill_id"].duplicated(keep="last")
    return trans


def final_states(transitions: pd.DataFrame, bills: pd.DataFrame) -> pd.DataFrame:
    """One row per bill: the last state reached and how the bill ended (or is ending)."""
    last = transitions[transitions["is_last"]].set_index("bill_id")
    out = bills[bills["bill_id"].isin(transitions["bill_id"])].set_index("bill_id")[
        ["short_title", "introduced_session_id", "outcome", "carry_over_date"]
    ]
    out["final_state"] = last["model_state"]
    out["exit_type"] = last["exit_type"]
    out["stop_reason"] = last["stop_reason"]
    return out.reset_index()


@dataclass
class CheckResult:
    name: str
    failures: pd.DataFrame

    @property
    def ok(self) -> bool:
        return self.failures.empty


def check_transitions(as_of: date, data: DataSource | None = None) -> list[CheckResult]:
    """BUILD_PLAN task 3 'done when' checks.

    1. Every Act ends in ``ROYAL_ASSENT``.
    2. Every bill that is not an Act is either in progress (in the current session, or carried
       over into it) with an open current state, or has a ``StopReason``.
    3. No negative durations (exit before entry, or negative sitting days).
    """
    src = _source(data)
    bills = src.get_bills(as_of)
    trans = build_transitions(as_of, src)
    finals = final_states(trans, bills)
    sessions = src.get_calendar(as_of)["sessions"]
    current = sessions.loc[sessions["end_date"].isna(), "session_id"].max()

    acts = finals[finals["outcome"] == "royal_assent"]
    not_ra = acts[acts["final_state"] != BillStage.ROYAL_ASSENT.value]

    in_progress = finals[finals["outcome"] == "in_progress"]
    bad_progress = in_progress[
        in_progress["exit_type"].notna()
        | (
            (in_progress["introduced_session_id"] != current)
            & in_progress["carry_over_date"].isna()
        )
    ]
    failed = finals[~finals["outcome"].isin(["royal_assent", "in_progress"])]
    no_reason = failed[failed["stop_reason"].isna()]

    exited = trans.dropna(subset=["exited_date"])
    negative = pd.concat(
        [
            exited[exited["exited_date"] < exited["entered_date"]],
            trans[trans["sitting_days_in_state"] < 0],
        ]
    ).drop_duplicates()

    return [
        CheckResult("Acts that do not end in royal_assent", not_ra),
        CheckResult("In-progress bills not in the current session or carried over", bad_progress),
        CheckResult("Failed bills without a StopReason", no_reason),
        CheckResult("Negative durations", negative),
    ]


def sanity_report(as_of: date, data: DataSource | None = None) -> dict[str, pd.DataFrame | int]:
    """Counts of bills by final state and outcome, median sitting days per state, and the
    number of bills with ping-pong."""
    src = _source(data)
    bills = src.get_bills(as_of)
    trans = build_transitions(as_of, src)
    finals = final_states(trans, bills)
    finals["ending"] = finals["stop_reason"].fillna(finals["outcome"])

    order = [s.value for s in chain("Commons")]
    by_final = pd.crosstab(finals["final_state"], finals["ending"]).reindex(
        [s for s in order if s in set(finals["final_state"])]
    )
    completed = trans[trans["exit_type"].notna() & (trans["state"] != BillStage.ROYAL_ASSENT)]
    durations = (
        completed.groupby("model_state")["sitting_days_in_state"]
        .agg(bills="size", median="median", p90=lambda s: s.quantile(0.9), max="max")
        .reindex([s for s in order if s != BillStage.ROYAL_ASSENT.value])
    )
    pingpong = int(trans.loc[trans["state"] == BillStage.PING_PONG, "bill_id"].nunique())
    return {
        "bills": len(finals),
        "by_final_state": by_final,
        "durations": durations,
        "bills_with_ping_pong": pingpong,
        "transitions": trans,
    }


def plot_durations(transitions: pd.DataFrame, path: Path) -> Path:
    """MODEL_SPEC §7 chart 5: distribution of sitting days spent in each state."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [s.value for s in chain("Commons") if s != BillStage.ROYAL_ASSENT]
    done = transitions[transitions["exit_type"] == "advance"]
    data = [done.loc[done["model_state"] == s, "sitting_days_in_state"].to_numpy() for s in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, vert=False, showfliers=True, flierprops={"markersize": 2, "alpha": 0.4})
    ax.set_yticks(range(1, len(order) + 1), [s.replace("_", " ") for s in order])
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xlabel("Sitting days in state before advancing (log scale above 5)")
    ax.set_title("How long bills spend in each stage (completed stages, all sessions ≥ 2006-07)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
