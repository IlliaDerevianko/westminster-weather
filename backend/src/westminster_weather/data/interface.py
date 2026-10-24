"""DATA_CONTRACT §5: model data interface backed by Neon Lakebase Postgres.

Every function takes ``as_of`` and returns nothing dated after it.

"""

from __future__ import annotations

import logging
from datetime import date
from functools import cache, cached_property

import numpy as np
import pandas as pd

from westminster_weather.data import calendar
from westminster_weather.data.stage_map import (
    CARRY_OVER_STAGE_IDS,
    COMMITTEE_OF_WHOLE_HOUSE_STAGE_ID,
    OPTIONAL_STATES,
    STAGE_ID_TO_STATE,
    chain,
    state_house,
)
from westminster_weather.db import (
    fetch_act_commencements_df,
    fetch_bills_df,
    fetch_parliaments_df,
    fetch_post_royal_assent_events_df,
    fetch_scheduled_sittings_df,
    fetch_sessions_df,
    fetch_sitting_days_df,
    fetch_stage_sittings_df,
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


class DatabaseInterface:
    """DATA_CONTRACT §5 functions reading from Neon Lakebase Postgres (or in-memory tables)."""

    def __init__(self, tables: dict[str, pd.DataFrame] | None = None) -> None:

        self._tables = tables

    @cached_property
    def bills(self) -> pd.DataFrame:
        if self._tables is not None:
            return self._tables["bills"]
        return fetch_bills_df()

    @cached_property
    def stage_sittings(self) -> pd.DataFrame:
        if self._tables is not None:
            return self._tables["stage_sittings"]
        return fetch_stage_sittings_df()

    @cached_property
    def sitting_days(self) -> pd.DataFrame:
        if self._tables is not None:
            return self._tables["sitting_days"]
        return fetch_sitting_days_df()

    @cached_property
    def sessions(self) -> pd.DataFrame:
        if self._tables is not None:
            return self._tables["sessions"]
        return fetch_sessions_df()

    @cached_property
    def parliaments(self) -> pd.DataFrame:
        if self._tables is not None:
            return self._tables["parliaments"]
        return fetch_parliaments_df()

    @cached_property
    def scheduled_sittings(self) -> pd.DataFrame:
        if self._tables is not None:
            return self._tables["scheduled_sittings"]
        return fetch_scheduled_sittings_df()

    @cached_property
    def act_commencements(self) -> pd.DataFrame:
        if self._tables is not None and "act_commencements" in self._tables:
            return self._tables["act_commencements"]
        return fetch_act_commencements_df()

    @cached_property
    def post_royal_assent_events(self) -> pd.DataFrame:
        if self._tables is not None and "post_royal_assent_events" in self._tables:
            return self._tables["post_royal_assent_events"]
        return fetch_post_royal_assent_events_df()

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
        """One row per bill per model state entered by ``as_of`` (MODEL_SPEC §2.2)."""
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
        """Time-varying bill features from the Bills API in long format, one row per change."""
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
        """``sitting_days`` up to ``as_of``; ``announced`` dates known by ``as_of``; ``sessions``
        and ``parliaments`` that had started, with end details hidden when they lie after ``as_of``.
        """
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
        """Future sittings scheduled when the data was downloaded, for live forecasts only."""
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

    # ------------------------------------------------------------------ §5.5 Phase 3

    def get_act_commencements(self, as_of: date) -> pd.DataFrame:
        """Acts that received Royal Assent on or before ``as_of``, with post-Royal-Assent dates
        censored at ``as_of`` so no future commencement or regulation dates leak."""
        df = self.act_commencements[self.act_commencements["royal_assent_date"] <= as_of].copy()
        for col in (
            "some_sections_start_date",
            "first_commencement_order_date",
            "latest_commencement_order_date",
            "first_draft_regulation_date",
            "latest_draft_regulation_date",
            "first_laid_regulation_date",
            "latest_laid_regulation_date",
            "law_fully_in_force_date",
            "first_amendment_or_repeal_date",
            "latest_amendment_or_repeal_date",
        ):
            if col in df.columns:
                s = df[col]
                df[col] = s.where(s.notna() & (s <= as_of), None)
        return df.reset_index(drop=True)

    def get_post_royal_assent_events(self, as_of: date) -> pd.DataFrame:
        """Phase 3 (After Royal Assent) events dated on or before ``as_of``."""
        events = self.post_royal_assent_events
        if events.empty:
            return events
        return events[events["event_date"] <= as_of].reset_index(drop=True)


@cache
def _default() -> DatabaseInterface:
    return DatabaseInterface()


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


def get_act_commencements(as_of: date) -> pd.DataFrame:
    return _default().get_act_commencements(as_of)


def get_post_royal_assent_events(as_of: date) -> pd.DataFrame:
    return _default().get_post_royal_assent_events(as_of)

