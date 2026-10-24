"""DATA_CONTRACT §5: the only functions the model uses to read data, backed by parquet files.

Every function takes ``as_of`` and returns nothing dated after it. A database
implementation must keep the same signatures and columns so ``model/`` never changes.
"""

from __future__ import annotations

import logging
from datetime import date
from functools import cache, cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from westminster_weather.data import calendar
from westminster_weather.data.paths import PROCESSED_DIR
from westminster_weather.data.stage_map import (
    CARRY_OVER_STAGE_IDS,
    COMMITTEE_OF_WHOLE_HOUSE_STAGE_ID,
    OPTIONAL_STATES,
    STAGE_ID_TO_STATE,
    chain,
    state_house,
)
from westminster_weather.domain.stages import BillStage

logger = logging.getLogger(__name__)

INTRO_COLUMNS = [
    "bill_id",
    "short_title",
    "originating_house",
    "introduced_session_id",
    "introduced_date",
    "department",
    "kings_speech_flag",
    "size_clauses",
    "size_pages",
]
TRANSITION_COLUMNS = [
    "bill_id",
    "model_state",
    "state_order",
    "entered_date",
    "exited_date",
    "sitting_days_in_state",
    "exit_type",
    "fail_cause",
]
COVARIATE_COLUMNS = ["bill_id", "feature", "valid_from", "value"]
ANNOUNCED_COLUMNS = ["id", "date_type", "house", "target_date", "announced_on"]

FAIL_CAUSE = {
    "fell_session_end": "session_end",
    "fell_dissolution": "dissolution",
    "withdrawn": "withdrawn",
    "defeated": "defeated",
}
STAGE_COMMONS_2R, STAGE_COMMONS_3R = 7, 10


def _outcome_as_of(bills: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """``outcome``/``outcome_date`` as they stood on ``as_of``."""
    bills = bills.copy()
    resolved = bills["outcome_date"].notna() & (bills["outcome_date"] <= as_of)
    bills.loc[~resolved, "outcome"] = "in_progress"
    bills.loc[~resolved, "outcome_date"] = None
    return bills


class ParquetInterface:
    """DATA_CONTRACT §5 functions over the parquet tables written by ``cli fetch``."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        self.processed_dir = processed_dir

    def _read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.processed_dir / f"{name}.parquet")

    @cached_property
    def bills(self) -> pd.DataFrame:
        return self._read("bills")

    @cached_property
    def stage_sittings(self) -> pd.DataFrame:
        return self._read("stage_sittings")

    @cached_property
    def sitting_days(self) -> pd.DataFrame:
        return self._read("sitting_days")

    @cached_property
    def sessions(self) -> pd.DataFrame:
        return self._read("sessions")

    @cached_property
    def parliaments(self) -> pd.DataFrame:
        return self._read("parliaments")

    @cached_property
    def scheduled_sittings(self) -> pd.DataFrame:
        return self._read("scheduled_sittings")

    @cached_property
    def _house_dates(self) -> dict[str | None, np.ndarray]:
        return {
            house: calendar.sitting_dates(self.sitting_days, house)
            for house in ("Commons", "Lords", None)
        }

    # ------------------------------------------------------------------ §5.1

    def get_bills(self, as_of: date) -> pd.DataFrame:
        """Bills introduced on or before ``as_of``: columns known at introduction, plus
        ``outcome``/``outcome_date`` as of ``as_of`` and ``carry_over_date`` if ≤ ``as_of``."""
        bills = self.bills[self.bills["introduced_date"] <= as_of]
        bills = _outcome_as_of(bills, as_of)
        carry = bills["carry_over_date"]
        bills["carry_over_date"] = carry.where(carry.notna() & (carry <= as_of), None)
        return bills[[*INTRO_COLUMNS, "outcome", "outcome_date", "carry_over_date"]].reset_index(
            drop=True
        )

    # ------------------------------------------------------------------ §5.2

    def get_transitions(self, as_of: date) -> pd.DataFrame:
        """One row per bill per model state entered by ``as_of`` (MODEL_SPEC §2.2).

        - A state is entered on its first sitting. A non-optional state with no sitting,
          but with a later state entered, counts as entered and exited on that later
          state's entry date (zero duration). Entry dates never go backwards along the chain.
        - ``exited_date`` is the next entered state's entry date. For the last state it is
          the failure date if the bill had failed by ``as_of`` (``exit_type = 'fail'``),
          otherwise NULL (censored). ``ROYAL_ASSENT`` is absorbing: exit columns NULL.
        - ``sitting_days_in_state`` counts the state's House's sitting days in
          (entered, exited or ``as_of``].
        """
        bills = _outcome_as_of(self.bills[self.bills["introduced_date"] <= as_of], as_of)
        sittings = self.stage_sittings[self.stage_sittings["event_date"] <= as_of]
        sittings = sittings.assign(state=sittings["api_stage_id"].map(STAGE_ID_TO_STATE))
        first = sittings.dropna(subset=["state"]).groupby(["bill_id", "state"])["event_date"].min()
        first_by_bill: dict[int, dict[BillStage, date]] = {}
        for (bill_id, state), d in first.items():
            first_by_bill.setdefault(bill_id, {})[BillStage(state)] = d

        rows = []
        for bill in bills.itertuples(index=False):
            rows.extend(self._bill_transitions(bill, first_by_bill.get(bill.bill_id, {}), as_of))
        return pd.DataFrame(rows, columns=TRANSITION_COLUMNS)

    def _bill_transitions(self, bill, first: dict[BillStage, date], as_of: date) -> list[dict]:
        states = chain(bill.originating_house)
        entries: list[tuple[BillStage, date]] = []
        for i, state in enumerate(states):
            entry = first.get(state)
            if entry is None:
                if state in OPTIONAL_STATES:
                    continue
                later = [first[s] for s in states[i + 1 :] if s in first]
                if not later:
                    break
                entry = min(later)
            if entries:
                entry = max(entry, entries[-1][1])
            entries.append((state, entry))

        rows = []
        for order, (state, entered) in enumerate(entries, start=1):
            exited, exit_type, fail_cause = None, None, None
            if order < len(entries):
                exited, exit_type = entries[order][1], "advance"
            elif bill.outcome in FAIL_CAUSE:
                exited = max(bill.outcome_date, entered)
                exit_type, fail_cause = "fail", FAIL_CAUSE[bill.outcome]
            if state == BillStage.ROYAL_ASSENT:
                days = 0
            else:
                dates = self._house_dates[state_house(state)]
                days = calendar.count_sitting_days(dates, entered, exited or as_of)
            rows.append(
                {
                    "bill_id": bill.bill_id,
                    "model_state": state.value,
                    "state_order": states.index(state) + 1,
                    "entered_date": entered,
                    "exited_date": exited,
                    "sitting_days_in_state": days,
                    "exit_type": exit_type,
                    "fail_cause": fail_cause,
                }
            )
        return rows

    # ------------------------------------------------------------------ §5.3

    def get_covariates(self, as_of: date) -> pd.DataFrame:
        """Time-varying bill features from the Bills API in long format, one row per change.

        Covers ``originating_house_lords``, ``committee_whole_house``, ``carry_over``,
        ``same_day_2r_3r`` and ``n_stage_sittings_cum``; task 4 adds the rest of the
        registry."""
        bills = self.bills[self.bills["introduced_date"] <= as_of]
        sittings = self.stage_sittings[
            (self.stage_sittings["event_date"] <= as_of)
            & self.stage_sittings["bill_id"].isin(bills["bill_id"])
        ]
        intro = bills[["bill_id", "introduced_date"]].rename(
            columns={"introduced_date": "valid_from"}
        )
        parts = [
            intro.assign(
                feature="originating_house_lords",
                value=(bills["originating_house"] == "Lords").astype(float).to_numpy(),
            )
        ]
        for feature, stage_ids in (
            ("committee_whole_house", (COMMITTEE_OF_WHOLE_HOUSE_STAGE_ID,)),
            ("carry_over", CARRY_OVER_STAGE_IDS),
        ):
            hit = (
                sittings[sittings["api_stage_id"].isin(stage_ids)]
                .groupby("bill_id")["event_date"]
                .min()
                .rename("valid_from")
                .reset_index()
            )
            parts += [
                intro.assign(feature=feature, value=0.0),
                hit.assign(feature=feature, value=1.0),
            ]

        firsts = sittings.groupby(["bill_id", "api_stage_id"])["event_date"].min().unstack()
        if STAGE_COMMONS_2R in firsts and STAGE_COMMONS_3R in firsts:
            same = firsts[firsts[STAGE_COMMONS_2R] == firsts[STAGE_COMMONS_3R]][STAGE_COMMONS_3R]
            parts.append(
                same.rename("valid_from").reset_index().assign(feature="same_day_2r_3r", value=1.0)
            )
        parts.append(intro.assign(feature="same_day_2r_3r", value=0.0))

        cum = sittings.groupby(["bill_id", "event_date"]).size().groupby(level=0).cumsum()
        parts.append(
            cum.rename("value")
            .astype(float)
            .reset_index()
            .rename(columns={"event_date": "valid_from"})
            .assign(feature="n_stage_sittings_cum")
        )

        out = pd.concat([p[COVARIATE_COLUMNS] for p in parts], ignore_index=True)
        out = out[out["valid_from"] <= as_of]
        return out.sort_values(["bill_id", "feature", "valid_from"]).reset_index(drop=True)

    # ------------------------------------------------------------------ §5.4

    def get_calendar(self, as_of: date) -> dict[str, pd.DataFrame]:
        """``sitting_days`` up to ``as_of``; ``announced`` dates known by ``as_of`` (none yet:
        no source); ``sessions`` and ``parliaments`` that had started, with end details
        hidden when they lie after ``as_of``."""
        days = self.sitting_days[self.sitting_days["date"] <= as_of].reset_index(drop=True)

        sessions = self.sessions[self.sessions["start_date"] <= as_of].copy()
        future_end = sessions["end_date"].notna() & (sessions["end_date"] > as_of)
        sessions.loc[future_end, ["end_date", "end_type"]] = None

        parls = self.parliaments[self.parliaments["first_meeting_date"] <= as_of].copy()
        future_dis = parls["dissolution_date"].notna() & (parls["dissolution_date"] > as_of)
        parls.loc[future_dis, ["dissolution_date", "election_date"]] = None

        return {
            "sitting_days": days,
            "announced": pd.DataFrame(columns=ANNOUNCED_COLUMNS),
            "sessions": sessions.reset_index(drop=True),
            "parliaments": parls.reset_index(drop=True),
        }

    # ------------------------------------------------------------------ live only

    def get_scheduled_sittings(self, as_of: date) -> pd.DataFrame:
        """Future sittings already scheduled when the data was downloaded, for **live
        forecasts only**. We do not know when each was announced, so for any ``as_of`` before
        the download date this returns an empty table (never use it in training or backtests).
        """
        scheduled = self.scheduled_sittings
        if scheduled.empty:
            return scheduled
        downloaded_on = scheduled["ingested_at"].min().date()
        if as_of < downloaded_on:
            logger.warning(
                "scheduled sittings hidden for as_of %s (before %s)", as_of, downloaded_on
            )
            return scheduled.iloc[0:0]
        return scheduled[scheduled["event_date"] > as_of].reset_index(drop=True)


@cache
def _default() -> ParquetInterface:
    return ParquetInterface()


def get_bills(as_of: date) -> pd.DataFrame:
    return _default().get_bills(as_of)


def get_transitions(as_of: date) -> pd.DataFrame:
    return _default().get_transitions(as_of)


def get_covariates(as_of: date) -> pd.DataFrame:
    return _default().get_covariates(as_of)


def get_calendar(as_of: date) -> dict[str, pd.DataFrame]:
    return _default().get_calendar(as_of)


def get_scheduled_sittings(as_of: date) -> pd.DataFrame:
    return _default().get_scheduled_sittings(as_of)
