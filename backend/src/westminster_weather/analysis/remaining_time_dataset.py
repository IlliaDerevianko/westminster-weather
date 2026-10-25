"""Point-in-time dataset for predicting the days left until Royal Assent.

Each row is one bill at one *forecast point*. A forecast point is either:

- a **stage anchor**: the moment a bill starts a stage (e.g. "1st House: committee"), or
- a **time anchor**: a fixed number of days after introduction (e.g. 90 days in).

Features only use sittings known at that point; for a stage anchor that means every sitting up
to and including the first sitting of that stage, in the Bills API's own order, so stages
held later on the same day are not leaked. The target is calendar days from the forecast point
to Royal Assent.

Only bills that reached Royal Assent are used: bills still in progress have no answer yet, and
bills that fell or were withdrawn have no Royal Assent date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Bills API stage IDs per step, by House. Committee covers Public Bill Committee (8),
# Committee of the whole House (49) and other committee forms (43, 48; Lords 46).
STEP_STAGE_IDS: dict[str, dict[str, frozenset[int]]] = {
    "Commons": {
        "1st reading": frozenset({6}),
        "2nd reading": frozenset({7}),
        "committee": frozenset({8, 43, 48, 49}),
        "report": frozenset({9}),
        "3rd reading": frozenset({10}),
    },
    "Lords": {
        "1st reading": frozenset({1}),
        "2nd reading": frozenset({2}),
        "committee": frozenset({3, 46}),
        "report": frozenset({4}),
        "3rd reading": frozenset({5}),
    },
}
STEP_NAMES = ("1st reading", "2nd reading", "committee", "report", "3rd reading")
PING_PONG_IDS = frozenset({12, 13, 42})
ROYAL_ASSENT_ID = 11

# Forecast points in pipeline order (used for grouping, sorting and charts).
STAGE_ANCHORS: tuple[str, ...] = (
    *(f"1st House: {s}" for s in STEP_NAMES),
    *(f"2nd House: {s}" for s in STEP_NAMES),
    "Ping-pong",
)
TIME_ANCHOR_DAYS: tuple[int, ...] = (30, 60, 90, 120, 180, 270, 365)

# Gaps between steps used as "how fast has it been so far" features (NaN until reached).
GAP_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("gap_h1_1r_2r", "1st House: 1st reading", "1st House: 2nd reading"),
    ("gap_h1_2r_cttee", "1st House: 2nd reading", "1st House: committee"),
    ("gap_h1_cttee_3r", "1st House: committee", "1st House: 3rd reading"),
    ("gap_h1_3r_h2_1r", "1st House: 3rd reading", "2nd House: 1st reading"),
    ("gap_h2_1r_2r", "2nd House: 1st reading", "2nd House: 2nd reading"),
    ("gap_h2_2r_cttee", "2nd House: 2nd reading", "2nd House: committee"),
    ("gap_h2_cttee_3r", "2nd House: committee", "2nd House: 3rd reading"),
)

# Procedural events, as "has it happened yet" flags.
EVENT_FLAGS: dict[str, frozenset[int]] = {
    "committee_whole_house": frozenset({49}),
    "public_bill_committee": frozenset({8}),
    "programme_motion": frozenset({14}),
    "money_resolution": frozenset({15}),
    "ways_and_means": frozenset({36}),
    "carry_over": frozenset({18, 19}),
    "lords_committee_negatived": frozenset({24, 27}),
    "commons_commitment_discharged": frozenset({26}),
    "ping_pong_started": PING_PONG_IDS,
}

MONEY_BILL_PATTERN = r"supply|appropriation|finance|consolidated fund|national insurance"
TOP_DEPARTMENTS = 10


@dataclass(frozen=True)
class SourceData:
    bills: pd.DataFrame
    sittings: pd.DataFrame
    sessions: pd.DataFrame
    parliaments: pd.DataFrame
    sitting_days: pd.Series


def load_source_data() -> SourceData:
    from westminster_weather.db import (
        fetch_bills_df,
        fetch_parliaments_df,
        fetch_sessions_df,
        fetch_sitting_days_df,
        fetch_stage_sittings_df,
    )

    sittings = fetch_stage_sittings_df()
    sittings = sittings[sittings["bill_type_id"] == 1].copy()  # DATA_SCOPE: Government only
    sittings["event_date"] = pd.to_datetime(sittings["event_date"])
    sessions = fetch_sessions_df()
    sessions["start_date"] = pd.to_datetime(sessions["start_date"])
    parliaments = fetch_parliaments_df()
    parliaments["first_meeting_date"] = pd.to_datetime(parliaments["first_meeting_date"])
    days = pd.Series(np.sort(pd.to_datetime(fetch_sitting_days_df()["date"]).unique()))
    return SourceData(fetch_bills_df(), sittings, sessions, parliaments, days)


def finished_bills(bills: pd.DataFrame) -> pd.DataFrame:
    """Bills that reached Royal Assent (drops in-progress, fallen and withdrawn bills)."""
    return bills[bills["outcome"] == "royal_assent"].copy()


def _step_of(stage_id: int, house: str, origin: str) -> str | None:
    if stage_id in PING_PONG_IDS:
        return "Ping-pong"
    position = "1st House" if house == origin else "2nd House"
    for step, ids in STEP_STAGE_IDS.get(house, {}).items():
        if stage_id in ids:
            return f"{position}: {step}"
    return None


class _Clock:
    """Session, Parliament and sitting-day lookups for point-in-time features."""

    def __init__(self, src: SourceData) -> None:
        s = src.sessions.sort_values("start_date")
        self._session_starts = s["start_date"].values
        self._session_parl = s["parliament_id"].values
        p = src.parliaments.set_index("parliament_id")["first_meeting_date"]
        self._parl_start = p.to_dict()
        self._sitting_days = src.sitting_days.values

    def session_start(self, d: pd.Timestamp) -> pd.Timestamp:
        i = max(np.searchsorted(self._session_starts, d.to_datetime64(), side="right") - 1, 0)
        return pd.Timestamp(self._session_starts[i])

    def parliament_age_years(self, d: pd.Timestamp) -> float:
        i = max(np.searchsorted(self._session_starts, d.to_datetime64(), side="right") - 1, 0)
        start = self._parl_start[self._session_parl[i]]
        return (d - start).days / 365.25

    def sitting_days_between(self, a: pd.Timestamp, b: pd.Timestamp) -> int:
        lo = np.searchsorted(self._sitting_days, a.to_datetime64(), side="right")
        hi = np.searchsorted(self._sitting_days, b.to_datetime64(), side="right")
        return int(hi - lo)


def _features(
    bill: pd.Series,
    known: pd.DataFrame,
    as_of: pd.Timestamp,
    clock: _Clock,
    departments: list[str],
) -> dict:
    """Features for one bill using only the sittings in ``known`` (all dated <= ``as_of``)."""
    intro = known["event_date"].min()
    first_by_step = known.dropna(subset=["step"]).groupby("step")["event_date"].min()
    reached = [a for a in STAGE_ANCHORS if a in first_by_step.index]
    furthest = max((STAGE_ANCHORS.index(a) for a in reached), default=0)
    ids = set(known["api_stage_id"])

    row: dict = {
        # Known from introduction
        "origin_lords": int(bill["originating_house"] == "Lords"),
        "money_bill_title": int(
            bool(pd.Series([bill["short_title"]]).str.contains(MONEY_BILL_PATTERN, case=False)[0])
        ),
        "intro_days_into_session": (intro - clock.session_start(intro)).days,
        "intro_parliament_age_years": clock.parliament_age_years(intro),
        # Where the bill is now
        "stage_index": furthest,
        "in_second_house": int(furthest >= STAGE_ANCHORS.index("2nd House: 1st reading")),
        "elapsed_days": (as_of - intro).days,
        "elapsed_sitting_days": clock.sitting_days_between(intro, as_of),
        "days_since_last_event": (as_of - known["event_date"].max()).days,
        "n_sittings": len(known),
        "n_sitting_dates": known["event_date"].nunique(),
        "asof_days_into_session": (as_of - clock.session_start(as_of)).days,
        "asof_parliament_age_years": clock.parliament_age_years(as_of),
        "asof_month": as_of.month,
        "h1_same_day_2r_3r": int(
            "1st House: 2nd reading" in first_by_step.index
            and first_by_step.get("1st House: 3rd reading")
            == first_by_step["1st House: 2nd reading"]
        ),
        "h1_report_skipped": int(
            "1st House: 3rd reading" in first_by_step.index
            and "1st House: report" not in first_by_step.index
        ),
    }
    for name, start, end in GAP_FEATURES:
        if start in first_by_step.index and end in first_by_step.index:
            gap = (first_by_step[end] - first_by_step[start]).days
            row[name] = gap if gap >= 0 else np.nan
        else:
            row[name] = np.nan
    for name, flag_ids in EVENT_FLAGS.items():
        row[name] = int(bool(ids & flag_ids))
    dept = bill["department"] if bill["department"] in departments else "other"
    for d in [*departments, "other"]:
        row[f"dept_{_slug(d)}"] = int(dept == d)
    return row


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")[:40]


# ---------------------------------------------------------------------------
# Extended features (``build_dataset(..., extended=True)``)
# ---------------------------------------------------------------------------

# Departments are renamed often (e.g. DCLG → MHCLG → DLUHC → MHCLG), which splits their bills
# across names. Map them to stable policy areas so each area has enough bills to learn from.
POLICY_AREAS: dict[str, str] = {
    "HM Treasury": "treasury",
    "Northern Ireland Office": "northern_ireland",
    "Scotland Office": "scotland_wales",
    "Wales Office": "scotland_wales",
    "Home Office": "home_affairs",
    "Ministry of Justice": "justice",
    "Cabinet Office": "constitution",
    "Office of the Leader of the House of Commons": "constitution",
    "Prime Minister": "constitution",
    "Office of the Deputy Prime Minister": "housing_local",
    "Government Equalities Office": "constitution",
    "Department for Exiting the European Union": "constitution",
    "Department of Trade and Industry": "business_energy",
    "Department for Business, Enterprise and Regulatory Reform": "business_energy",
    "Department for Business, Innovation and Skills": "business_energy",
    "Department for Business, Energy and Industrial Strategy": "business_energy",
    "Department for Business and Trade": "business_energy",
    "Department for International Trade": "business_energy",
    "Department for Energy Security and Net Zero": "business_energy",
    "Department of Energy and Climate Change": "business_energy",
    "Department for Communities and Local Government": "housing_local",
    "Ministry of Housing, Communities and Local Government": "housing_local",
    "Department for Levelling Up, Housing and Communities": "housing_local",
    "Department of Health": "health",
    "Department of Health and Social Care": "health",
    "Department for Education": "education",
    "Department for Children, Schools and Families": "education",
    "Department for Education and Skills": "education",
    "Department for Innovation, Universities and Skills": "education",
    "Department for Culture, Media and Sport": "culture_digital",
    "Department for Digital, Culture, Media and Sport": "culture_digital",
    "Department for Science, Innovation and Technology": "culture_digital",
    "Department for Transport": "transport",
    "Department for Environment, Food and Rural Affairs": "environment",
    "Department for Work and Pensions": "work_pensions",
    "Ministry of Defence": "defence",
    "Foreign and Commonwealth Office": "foreign",
    "Foreign, Commonwealth and Development Office": "foreign",
    "Department for International Development": "foreign",
}
POLICY_AREA_NAMES = tuple(sorted(set(POLICY_AREAS.values()) | {"other"}))

# Commons majority won at the general election that started each Parliament, and whether the
# government was a coalition or a minority (public election results, known from day one).
GOVERNMENT_MAJORITY = {1: 167, 2: 66, 3: 76, 4: 12, 5: -16, 6: 80, 7: 174}
COALITION_OR_MINORITY = {3: 1, 5: 1}

TITLE_FLAGS: dict[str, str] = {
    "title_northern_ireland": r"northern ireland",
    "title_numbered": r"\(no\. ?\d\)",
    "title_amendment": r"amendment",
    "title_finance": r"finance|supply|appropriation|consolidated fund|national insurance|tax",
    "title_emergency": r"emergency|coronavirus|covid|urgent",
    "title_scotland_wales": r"scotland|wales|welsh|scottish",
    "title_eu": r"european union|withdrawal|brexit|retained eu",
    "title_repeal": r"repeal",
}
SUMMER_RECESS_START = (7, 21)  # Houses usually rise for the summer in late July...
SUMMER_RECESS_END = (9, 4)  # ...and return in early September.


class _Context:
    """Whole-dataset lookups for extended features (congestion uses all Government Bills)."""

    def __init__(self, src: SourceData) -> None:
        first = src.sittings.groupby("bill_id")["event_date"].min()
        ra = src.sittings[src.sittings["api_stage_id"] == ROYAL_ASSENT_ID].groupby("bill_id")[
            "event_date"
        ].min()
        b = src.bills.set_index("bill_id")
        end = ra.reindex(first.index).fillna(pd.to_datetime(b["outcome_date"]).reindex(first.index))
        # Bills still in progress are active until further notice.
        end = end.where(b["outcome"].reindex(first.index) != "in_progress", pd.Timestamp.max)
        self.start = first.values
        self.end = end.fillna(pd.Timestamp.max).values
        self.session = b["introduced_session_id"].reindex(first.index).values
        self.parl_of_session = src.sessions.set_index("session_id")["parliament_id"].to_dict()
        self.parl_start = src.parliaments.set_index("parliament_id")["first_meeting_date"].to_dict()

    def active_bills(self, as_of: pd.Timestamp) -> int:
        t = as_of.to_datetime64()
        return int(((self.start <= t) & (self.end > t)).sum())

    def session_order(self, session_id: int, intro: pd.Timestamp) -> int:
        t = intro.to_datetime64()
        return int(((self.session == session_id) & (self.start < t)).sum())


def _days_to_summer_recess(d: pd.Timestamp) -> tuple[int, int]:
    start = pd.Timestamp(d.year, *SUMMER_RECESS_START)
    end = pd.Timestamp(d.year, *SUMMER_RECESS_END)
    if start <= d <= end:
        return 0, 1
    nxt = start if d < start else pd.Timestamp(d.year + 1, *SUMMER_RECESS_START)
    return (nxt - d).days, 0


def _extra_features(
    bill: pd.Series, known: pd.DataFrame, as_of: pd.Timestamp, clock: _Clock, ctx: _Context
) -> dict:
    origin = bill["originating_house"]
    first_by_step = known.dropna(subset=["step"]).groupby("step")["event_date"].min()
    steps = known["step"]

    def house_days(position: str) -> float:
        start = first_by_step.get(f"{position}: 1st reading")
        if start is None:
            return np.nan
        end = first_by_step.get(f"{position}: 3rd reading", as_of)
        return float((end - start).days)

    h1, h2 = house_days("1st House"), house_days("2nd House")
    in_h2 = not np.isnan(h2)
    furthest_step = max(
        (a for a in first_by_step.index if a in STAGE_ANCHORS), key=STAGE_ANCHORS.index,
        default=None,
    )
    stage_start = first_by_step.get(furthest_step, as_of) if furthest_step else as_of
    recess_days, in_recess = _days_to_summer_recess(as_of)
    parl = ctx.parl_of_session.get(bill["introduced_session_id"])
    limit = ctx.parl_start.get(parl, as_of) + pd.DateOffset(years=5)
    area = POLICY_AREAS.get(bill["department"], "other")
    title = str(bill["short_title"]).lower()

    row: dict = {
        "policy_area": area,  # string; encoded later from training data only
        "commons_days_so_far": h1 if origin == "Commons" else h2,
        "lords_days_so_far": h1 if origin == "Lords" else h2,
        "first_house_days_so_far": h1,
        "second_house_days_so_far": h2,
        "current_house_days_so_far": h2 if in_h2 else h1,
        "first_house_finished": int("1st House: 3rd reading" in first_by_step.index),
        "days_in_current_stage": (as_of - stage_start).days,
        "n_h1_2r_sittings": int((steps == "1st House: 2nd reading").sum()),
        "n_h1_committee_sittings": int((steps == "1st House: committee").sum()),
        "n_h1_report_sittings": int((steps == "1st House: report").sum()),
        "n_h2_committee_sittings": int((steps == "2nd House: committee").sum()),
        "n_h2_report_sittings": int((steps == "2nd House: report").sum()),
        "n_ping_pong_rounds": int(known.loc[steps == "Ping-pong", "event_date"].nunique()),
        "days_to_summer_recess": recess_days,
        "in_summer_recess": in_recess,
        "days_to_parliament_5yr_limit": (limit - as_of).days,
        "active_government_bills": ctx.active_bills(as_of),
        "bills_before_in_session": ctx.session_order(
            bill["introduced_session_id"], known["event_date"].min()
        ),
        "government_majority": GOVERNMENT_MAJORITY.get(parl, np.nan),
        "coalition_or_minority": COALITION_OR_MINORITY.get(parl, 0),
        "title_words": len(title.split()),
    }
    for name, pattern in TITLE_FLAGS.items():
        row[name] = int(pd.Series([title]).str.contains(pattern, regex=True)[0])
    for a in POLICY_AREA_NAMES:
        row[f"area_{a}"] = int(area == a)
    return row


def build_dataset(
    src: SourceData, train_sessions: set[int] | None = None, extended: bool = False
) -> pd.DataFrame:
    """One row per (finished bill, forecast point) with features, target and anchor labels.

    ``train_sessions`` picks which bills define the top departments, so test bills never
    influence the feature set.
    """
    bills = finished_bills(src.bills)
    ref = (
        bills
        if train_sessions is None
        else bills[bills["introduced_session_id"].isin(train_sessions)]
    )
    departments = ref["department"].value_counts().head(TOP_DEPARTMENTS).index.tolist()
    clock = _Clock(src)
    ctx = _Context(src) if extended else None
    rows: list[dict] = []

    def extra(bill: pd.Series, known: pd.DataFrame, as_of: pd.Timestamp) -> dict:
        return _extra_features(bill, known, as_of, clock, ctx) if ctx is not None else {}

    for _, bill in bills.iterrows():
        s = src.sittings[src.sittings["bill_id"] == bill["bill_id"]].sort_values(
            ["event_date", "sort_order"]
        )
        ra = s.loc[s["api_stage_id"] == ROYAL_ASSENT_ID, "event_date"]
        ra_date = ra.min() if len(ra) else pd.Timestamp(bill["outcome_date"])
        s = s[s["api_stage_id"] != ROYAL_ASSENT_ID].reset_index(drop=True)
        if s.empty or pd.isna(ra_date):
            continue
        s["step"] = [
            _step_of(sid, house, bill["originating_house"])
            for sid, house in zip(s["api_stage_id"], s["house"], strict=True)
        ]
        intro = pd.Timestamp(s["event_date"].min())
        base = {
            "bill_id": bill["bill_id"],
            "session_id": bill["introduced_session_id"],
            "short_title": bill["short_title"],
        }

        # Stage anchors: knowledge = sittings up to and including the stage's first sitting.
        for anchor in STAGE_ANCHORS:
            idx = s.index[s["step"] == anchor]
            if len(idx) == 0:
                continue
            known = s.loc[: idx[0]]
            as_of = known["event_date"].max()
            target = (ra_date - as_of).days
            if target < 0:
                continue
            rows.append(
                {
                    **base, "anchor_type": "stage", "anchor": anchor, "as_of": as_of,
                    **_features(bill, known, as_of, clock, departments),
                    **extra(bill, known, as_of),
                    "target_days": target,
                }
            )

        # Time anchors: knowledge = sittings dated on or before intro + N days.
        for n in TIME_ANCHOR_DAYS:
            as_of = intro + np.timedelta64(n, "D")
            if as_of >= ra_date:
                continue
            known = s[s["event_date"] <= as_of]
            rows.append(
                {
                    **base, "anchor_type": "time", "anchor": f"{n} days in", "as_of": as_of,
                    **_features(bill, known, as_of, clock, departments),
                    **extra(bill, known, as_of),
                    "target_days": (ra_date - as_of).days,
                }
            )
    return pd.DataFrame(rows)


NON_FEATURE_COLUMNS = (
    "bill_id", "session_id", "short_title", "anchor_type", "anchor", "as_of", "target_days",
    "policy_area",
)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
