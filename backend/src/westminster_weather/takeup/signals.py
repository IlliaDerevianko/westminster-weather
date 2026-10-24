"""DATA_CONTRACT §10.9: ``get_takeup_signals(as_of)``, the only way the take-up model sees data.

One row per signal visible at ``as_of``, in the format ``TakeUpModel.prepare()`` reads:
``signal_id, idea_id, title, origin, signal_date, bill_id, bill_date, data_end, z_*, x_*``
(``z_*`` numeric, known at ``signal_date``; ``x_*`` dates, ``NaT`` if not happened by
``as_of``). Nothing dated after ``as_of`` can change the output (no leakage):

Ideas come only from NLP over ministers' words (written ministerial statements and Hansard);
King's Speech data is not used.

- Mentions: ``signal_mentions`` rows with ``is_commitment`` and ``is_minister`` true and
  ``event_date <= as_of``, joined to ``idea_mentions``.
- Grouping: an idea whose linked bill had its first reading by ``as_of`` is one signal
  (``signal_id = idea_id``). Otherwise the same-bill merge is not yet known, so each
  ``idea_mentions.group_id`` is its own signal (``signal_id = '<idea_id>:<group_id>'``).
  A split row uses only what belongs to its group: ``title`` comes from the group's own
  mentions (``origin`` is always ``nlp``), and ``idea_events`` apply only to the idea's anchor
  group, chosen from mentions visible at ``as_of`` (``_anchor_groups``): the group of the
  idea's earliest mention. ``idea_id`` (and the ``idea_id`` part of ``signal_id``) on a split
  row reflects the later merge: it is an identifier only and must never be used as a feature
  or to group rows.
- ``z_src_hansard``: the first mention came from Hansard (spoken or written); written
  ministerial statements are the reference level (DATA_CONTRACT §10.11).
- ``bill_date``: first reading of the linked bill (earliest stage 6 or 1 sitting) if it is
  ``<= as_of``. A bill first read up to 7 days before the first mention counts as taken up
  in week 1: ``signal_date`` is moved back to the first reading. A signal whose bill was
  first read earlier than that refers to an existing bill and is dropped (see
  ``dropped_signals``).
- Change of government: once its election has been held (``election < as_of``), a signal
  with no bill by a change-of-government dissolution after ``signal_date`` is censored there
  (``data_end`` = dissolution) and a bill first read after it is dropped. Otherwise
  ``data_end = as_of``.
- Features with fewer than ``MIN_POSITIVE_CASES`` positive cases are dropped and logged
  (pass ``features=`` to get a fixed column set, e.g. to align a test frame with training).
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from functools import cached_property

import numpy as np
import pandas as pd

from westminster_weather.data import calendar
from westminster_weather.takeup.tables import DATA_TABLES

logger = logging.getLogger(__name__)

FIRST_READING_STAGE_IDS = (6, 1)  # Bills API: Commons 1st reading, Lords 1st reading
# Elections where the governing party changed, with their dissolutions (fallback if the
# parliaments table has no matching row).
CHANGE_OF_GOVERNMENT = {date(2010, 5, 6): date(2010, 4, 12), date(2024, 7, 4): date(2024, 5, 30)}
BILL_BEFORE_SIGNAL_DAYS = 7
MIN_POSITIVE_CASES = 15
# Department one-hots: at most this many of the most common departments, each with at least
# MIN_POSITIVE_CASES signals. All other departments and unknown departments are pooled into
# the reference level 'other' (no column), so the one-hots are not collinear with the
# intercept. Written statements ('wms') are the reference level for the first source.
MAX_DEPARTMENTS = 8
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.4375

# Tie-break between same-day mentions: the first source is the highest-ranked one.
SOURCE_RANK = {"wms": 0, "hansard_written": 1, "hansard_spoken": 2}
# idea_events.event_type -> x feature (§10.11: one `white_paper` feature for white or green
# papers, the earliest of either).
EVENT_FEATURES = {
    "consultation": "x_consultation",
    "consultation_response": "x_consultation_response",
    "green_paper": "x_white_paper",
    "white_paper": "x_white_paper",
    "draft_bill": "x_draft_bill",
}
BASE_COLUMNS = [
    "signal_id",
    "idea_id",
    "title",
    "origin",
    "signal_date",
    "bill_id",
    "bill_date",
    "data_end",
]
FIXED_Z = [
    "z_src_hansard",
    "z_initial_firmness",
    "z_years_since_election",
    "z_months_into_session",
]
CONTINUOUS_Z = {"z_initial_firmness", "z_years_since_election", "z_months_into_session"}
X_COLUMNS = [
    "x_firm_commitment",
    "x_repeat_mention",
    *dict.fromkeys(EVENT_FEATURES.values()),
    "x_new_parliament",
]
REFERENCE_TABLES = ("bills", "stage_sittings", "sessions", "parliaments", "sitting_days")
MENTION_COLUMNS = [
    "idea_id",
    "group_id",
    "mention_id",
    "source",
    "doc_id",
    "event_date",
    "firmness",
    "department",
    "topic",
]


def _dates(values: pd.Series) -> pd.Series:
    """Dates (or strings) as midnight ``Timestamp``s; missing -> ``NaT``."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _flag(values: pd.Series) -> pd.Series:
    """True only where the value is true; NULL / unknown counts as false."""
    return values.astype("boolean").fillna(False).astype(bool)


def _last_on_or_before(points: list[pd.Timestamp], d: pd.Timestamp) -> pd.Timestamp | None:
    earlier = [p for p in points if p <= d]
    return earlier[-1] if earlier else None


def department_slug(name: object) -> str | None:
    """'Department for Education' -> 'education'; 'Home Office' -> 'home_office'."""
    if name is None or pd.isna(name) or not str(name).strip():
        return None
    s = re.sub(r"^(?:the\s+)?department\s+(?:for|of)\s+", "", str(name).strip().lower())
    s = re.sub(r"[^a-z0-9]+", "_", s.replace("&", " and ")).strip("_")
    return s or None


class TakeupData:
    """Take-up data as of a date. ``tables`` (by table name) is injectable for tests; by
    default the take-up tables come from ``DATABASE_URL_TAKEUP`` and the reference tables
    (bills, stage_sittings, sessions, parliaments, sitting_days) from production, read-only.
    """

    def __init__(self, tables: dict[str, pd.DataFrame] | None = None) -> None:
        self._tables = tables
        self.dropped_features: dict[str, int] = {}
        # signal_id -> linked bill_id, for signals dropped because their bill was first read
        # more than BILL_BEFORE_SIGNAL_DAYS before the first mention (an existing bill).
        self.dropped_signals: dict[str, int] = {}
        self.n_signal_date_moved = 0  # signals moved back to a first reading 1-7 days earlier

    # ------------------------------------------------------------------ tables

    @cached_property
    def tables(self) -> dict[str, pd.DataFrame]:
        if self._tables is not None:
            tables = dict(self._tables)
        else:
            from westminster_weather.takeup import db

            tables = {**db.load_takeup_tables(), **db.load_reference_tables()}
        for t in DATA_TABLES:  # a missing take-up table is an empty one
            if t.name not in tables:
                tables[t.name] = pd.DataFrame(columns=[c.name for c in t.columns])
        missing = [name for name in REFERENCE_TABLES if name not in tables]
        if missing:
            raise KeyError(f"missing reference tables: {missing}")
        return tables

    @cached_property
    def first_readings(self) -> pd.Series:
        """bill_id -> first reading (earliest stage 6 or 1 sitting), Government Bills only."""
        ss = self.tables["stage_sittings"]
        ss = ss[ss["api_stage_id"].isin(FIRST_READING_STAGE_IDS)]
        ss = ss.assign(event_date=_dates(ss["event_date"])).dropna(subset=["event_date"])
        ss = ss[ss["bill_id"].isin(self.tables["bills"]["bill_id"])]
        return ss.groupby("bill_id")["event_date"].min()

    @cached_property
    def _parliaments(self) -> pd.DataFrame:
        p = self.tables["parliaments"].copy()
        for col in ("first_meeting_date", "dissolution_date", "election_date"):
            p[col] = _dates(p[col])
        return p.sort_values("first_meeting_date").reset_index(drop=True)

    @cached_property
    def elections(self) -> list[pd.Timestamp]:
        return sorted(self._parliaments["election_date"].dropna())

    @cached_property
    def changes_of_government(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(election, dissolution before it) for elections that changed the governing party.

        The dissolution comes from ``parliaments`` (fallback: ``CHANGE_OF_GOVERNMENT``). That
        the party changed is only known after the election, so callers must use a pair only
        when ``election < as_of`` (see ``known_dissolutions``).
        """
        p = self._parliaments
        out = []
        for election, fallback in CHANGE_OF_GOVERNMENT.items():
            match = p.loc[p["election_date"] == pd.Timestamp(election), "dissolution_date"]
            match = match.dropna()
            out.append(
                (pd.Timestamp(election), match.iloc[0] if len(match) else pd.Timestamp(fallback))
            )
        return sorted(out)

    @property
    def change_of_government_dissolutions(self) -> list[pd.Timestamp]:
        """All change-of-government dissolutions (no as_of check: not for signal building)."""
        return [d for _, d in self.changes_of_government]

    def known_dissolutions(self, as_of: pd.Timestamp) -> list[pd.Timestamp]:
        """Change-of-government dissolutions whose election result is known at ``as_of``
        (the result is known from the day after polling day)."""
        return [d for e, d in self.changes_of_government if e < as_of]

    def _anchor_groups(self, m: pd.DataFrame) -> dict[str, str]:
        """idea_id -> the group that owns the idea's ``idea_events`` while groups are split,
        from the mentions ``m`` visible at ``as_of`` only, so groups whose mentions all come
        later (which only ideas that will get a bill have) cannot move it: the group of the
        idea's earliest visible mention. That is the idea's earliest mention overall, so the
        anchor never changes while the groups are split.

        Other groups get no idea events until the bill's first reading is visible (the merge
        itself uses knowledge of the bill)."""
        ids, groups = m["idea_id"].astype(str), m["group_id"].astype(str)
        earliest = m.assign(idea_id=ids, group_id=groups).sort_values(
            ["event_date", "rank", "mention_id"]
        )
        earliest = earliest.drop_duplicates("idea_id")
        return dict(zip(earliest["idea_id"], earliest["group_id"], strict=True))

    @cached_property
    def _session_starts(self) -> list[pd.Timestamp]:
        return sorted(_dates(self.tables["sessions"]["start_date"]).dropna())

    @cached_property
    def same_party_openings(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(election, first sitting of the new Parliament's first session) for each election
        that did not change the governing party."""
        p = self._parliaments
        sess = self.tables["sessions"].copy()
        sess["start_date"] = _dates(sess["start_date"])
        cog = {pd.Timestamp(e) for e in CHANGE_OF_GOVERNMENT}
        out = []
        for election in self.elections:
            if election in cog:
                continue
            nxt = p[p["first_meeting_date"] > election]
            if nxt.empty:
                continue
            new = nxt.iloc[0]
            starts = sess.loc[sess["parliament_id"] == new["parliament_id"], "start_date"]
            if starts.dropna().empty:  # no parliament_id: first session starting after it met
                starts = sess.loc[sess["start_date"] >= new["first_meeting_date"], "start_date"]
            if not starts.dropna().empty:
                out.append((election, starts.min()))
        return out

    # --------------------------------------------------------------- calendar

    def sitting_dates(self, as_of: date | str | pd.Timestamp) -> np.ndarray:
        """Sorted unique dates (``datetime64[D]``) either House sat, up to ``as_of``."""
        days = self.tables["sitting_days"]
        days = days[_dates(days["date"]) <= pd.Timestamp(as_of)]
        return calendar.sitting_dates(days)

    def week_probability(self, as_of: date | str | pd.Timestamp) -> pd.Series:
        """P(either House sits in ISO week w), as ``calendar.week_sitting_probability`` but with
        this object's parliaments (so it needs no database), from sittings ``<= as_of``."""
        as_of_d = pd.Timestamp(as_of).date()
        p = self._parliaments
        parls = pd.DataFrame(
            {
                "dissolution_date": [
                    None if pd.isna(d) else d.date() for d in p["dissolution_date"]
                ],
                "first_meeting_date": [d.date() for d in p["first_meeting_date"]],
            }
        )
        gaps = calendar.dissolution_gaps(parls)
        days = pd.to_datetime(self.tables["sitting_days"]["date"]).dt.date
        sat = {d.isocalendar()[:2] for d in days if d <= as_of_d}
        hits, totals = np.zeros(54), np.zeros(54)
        for year in range(as_of_d.year - calendar.FUTURE_HISTORY_YEARS, as_of_d.year):
            monday = date.fromisocalendar(year, 1, 1)
            while monday.isocalendar()[0] == year:
                week = monday.isocalendar()[1]
                if not calendar.in_dissolution_gap(monday + timedelta(days=2), gaps):
                    totals[week] += 1
                    hits[week] += (year, week) in sat
                monday += timedelta(weeks=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            prob = np.where(totals > 0, hits / totals, 0.0)
        return pd.Series(prob[1:], index=pd.RangeIndex(1, 54, name="iso_week"))

    # ---------------------------------------------------------------- signals

    def visible_mentions(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Idea mentions dated ``<= as_of``: minister commitments from ``signal_mentions``."""
        sm = self.tables["signal_mentions"]
        sm = sm.assign(event_date=_dates(sm["event_date"]))
        sm = sm[_flag(sm["is_commitment"]) & _flag(sm["is_minister"]) & (sm["event_date"] <= as_of)]
        nlp = pd.DataFrame(
            {
                "mention_id": sm["mention_id"].astype(str),
                "source": sm["source"],
                "doc_id": sm["doc_id"].astype(str),
                "event_date": sm["event_date"],
                "firmness": pd.to_numeric(sm["firmness"]).fillna(0).astype(int),
                "department": sm["department"],
                "topic": sm["topic"],
            }
        )
        if nlp.empty:
            return pd.DataFrame(columns=MENTION_COLUMNS)
        im = self.tables["idea_mentions"][["idea_id", "mention_id", "group_id"]].copy()
        im["mention_id"] = im["mention_id"].astype(str)
        out = im.merge(nlp, on="mention_id", how="inner")
        return out[MENTION_COLUMNS]

    def get_takeup_signals(
        self,
        as_of: date | str | pd.Timestamp,
        features: list[str] | None = None,
        min_positive: int = MIN_POSITIVE_CASES,
    ) -> pd.DataFrame:
        """One row per signal visible at ``as_of`` (DATA_CONTRACT §10.9).

        ``features`` fixes the ``z_*`` / ``x_*`` columns (missing ones are 0 / NaT) and turns
        off the sparse-feature drop; otherwise features with fewer than ``min_positive``
        positive cases are dropped and recorded in ``self.dropped_features``.
        """
        as_of = pd.Timestamp(as_of).normalize()
        self.dropped_features, self.dropped_signals, self.n_signal_date_moved = {}, {}, 0
        m = self.visible_mentions(as_of)
        if m.empty:
            cols = BASE_COLUMNS + (features if features is not None else FIXED_Z + X_COLUMNS)
            return self._typed(pd.DataFrame(columns=cols))

        links = self.tables["idea_bill_links"].dropna(subset=["bill_id"])
        links = links.drop_duplicates("idea_id").set_index("idea_id")["bill_id"].astype("int64")
        m["bill_id"] = m["idea_id"].map(links)
        m["first_reading"] = m["bill_id"].map(self.first_readings)
        merged = m["first_reading"].notna() & (m["first_reading"] <= as_of)
        m["signal_id"] = np.where(
            merged, m["idea_id"], m["idea_id"].astype(str) + ":" + m["group_id"].astype(str)
        )
        m["rank"] = m["source"].map(SOURCE_RANK).fillna(len(SOURCE_RANK))
        m = m.sort_values(["signal_id", "event_date", "rank", "mention_id"])

        ev = self.tables["idea_events"]
        ev = ev.assign(event_date=_dates(ev["event_date"]))
        ev = ev[ev["event_date"] <= as_of]
        events = {idea: g for idea, g in ev.groupby("idea_id")}
        ideas = self.tables["ideas"].drop_duplicates("idea_id").set_index("idea_id")
        anchors = self._anchor_groups(m)

        rows = []
        for sid, g in m.groupby("signal_id", sort=False):
            idea_id = g["idea_id"].iloc[0]
            merged_row = sid == idea_id
            # While split, idea-level data applies only where the group-to-idea association
            # does not depend on the bill link (DATA_CONTRACT §10.4).
            own = merged_row or anchors.get(str(idea_id)) == str(g["group_id"].iloc[0])
            row = self._signal_row(sid, g, events.get(idea_id) if own else None, as_of)
            if row is None:
                continue
            if merged_row:
                origin = ideas["origin"].get(idea_id) if "origin" in ideas else None
                title = ideas["topic"].get(idea_id) if "topic" in ideas else None
                if origin is not None and pd.notna(origin):
                    row["origin"] = origin
                if title is not None and pd.notna(title):
                    row["title"] = title
            rows.append(row)
        if self.dropped_signals:
            logger.info(
                "dropped %d signal(s) whose bill was first read > %d days before the signal",
                len(self.dropped_signals),
                BILL_BEFORE_SIGNAL_DAYS,
            )
        if not rows:
            cols = BASE_COLUMNS + (features if features is not None else FIXED_Z + X_COLUMNS)
            return self._typed(pd.DataFrame(columns=cols))
        df = pd.DataFrame(rows)
        df = self._add_departments(df, features, min_positive)
        df = df.drop(columns="department")
        if features is not None:
            for col in features:
                if col not in df:
                    df[col] = pd.NaT if col.startswith("x_") else 0.0
            df = df[BASE_COLUMNS + list(features)]
        else:
            df = self._drop_sparse(df, min_positive)
        df = self._typed(df)
        return df.sort_values(["signal_date", "signal_id"]).reset_index(drop=True)

    def _signal_row(
        self,
        signal_id: str,
        g: pd.DataFrame,
        events: pd.DataFrame | None,
        as_of: pd.Timestamp,
    ) -> dict[str, object] | None:
        """One signal from its visible mentions ``g`` (sorted); ``events`` are the idea events
        this row may use (None while split, unless it is the idea's anchor group). Returns None
        if the linked bill was first read before the commitment (an existing bill)."""
        first = g.iloc[0]
        first_date = pd.Timestamp(first["event_date"])  # earliest visible mention
        fr, bill_id = pd.Timestamp(first["first_reading"]), first["bill_id"]

        # Outcome and censoring. A bill visible at as_of was first read either well before the
        # commitment (drop: an existing bill), up to 7 days before it (taken up in week 1:
        # the signal date moves back to the first reading, which prepare() needs) or after it.
        has_bill = pd.notna(fr) and fr <= as_of
        signal_date = first_date
        if has_bill and fr < first_date:
            if fr < first_date - pd.Timedelta(BILL_BEFORE_SIGNAL_DAYS, unit="D"):
                self.dropped_signals[signal_id] = int(bill_id)
                return None
            signal_date = fr
            self.n_signal_date_moved += 1
        data_end = as_of
        cog = next((d for d in self.known_dissolutions(as_of) if signal_date < d), None)
        if cog is not None and not (has_bill and fr <= cog):
            has_bill, data_end = False, cog

        # z: known at the first mention, from that day's visible mentions only.
        on_day = g[g["event_date"] == first_date]
        dept = on_day["department"].dropna()
        election = _last_on_or_before(self.elections, signal_date)
        session = _last_on_or_before(self._session_starts, signal_date)
        source = str(first["source"])

        # x: dates <= as_of only (every row here is already visible at as_of), strictly after
        # the first mention (anything on that day is a z feature).
        later = g[g["event_date"] > first_date]
        firm_at_start = on_day["firmness"].max() >= 2  # then z_initial_firmness says it all
        firm = pd.NaT if firm_at_start else later.loc[later["firmness"] >= 2, "event_date"].min()
        repeat = later["event_date"].min()  # a later mention is always another document
        x_events: dict[str, object] = dict.fromkeys(EVENT_FEATURES.values(), pd.NaT)
        if events is not None:
            for col in x_events:
                types = [e for e, c in EVENT_FEATURES.items() if c == col]
                x_events[col] = events.loc[events["event_type"].isin(types), "event_date"].min()

        topic = first["topic"]
        return {
            "signal_id": signal_id,
            "idea_id": first["idea_id"],
            # split rows: from the group's own mentions (merged rows: replaced from `ideas`)
            "title": None if pd.isna(topic) else str(topic),
            "origin": "nlp",
            "signal_date": signal_date,
            "bill_id": bill_id if has_bill else None,
            "bill_date": fr if has_bill else pd.NaT,
            "data_end": data_end,
            "department": dept.iloc[0] if len(dept) else None,
            "z_src_hansard": float(source.startswith("hansard")),
            "z_initial_firmness": float(on_day["firmness"].max()),
            "z_years_since_election": (signal_date - election).days / DAYS_PER_YEAR
            if election is not None
            else np.nan,
            "z_months_into_session": (signal_date - session).days / DAYS_PER_MONTH
            if session is not None
            else np.nan,
            "x_firm_commitment": firm,
            "x_repeat_mention": repeat,
            **x_events,
            "x_new_parliament": self._new_parliament(signal_date, as_of),
        }

    def _new_parliament(self, signal_date: pd.Timestamp, as_of: pd.Timestamp) -> pd.Timestamp:
        """State Opening after the first same-party election after ``signal_date``; NaT if a
        change of government came first or the opening is after ``as_of``."""
        later = [(e, o) for e, o in self.same_party_openings if e > signal_date]
        if not later:
            return pd.NaT
        election, opening = later[0]
        if any(signal_date < d < election for d in self.known_dissolutions(as_of)):
            return pd.NaT
        return opening if opening <= as_of else pd.NaT

    def _add_departments(
        self, df: pd.DataFrame, features: list[str] | None, min_positive: int
    ) -> pd.DataFrame:
        slugs = df["department"].map(department_slug)
        if features is not None:
            chosen = [c.removeprefix("z_dept_") for c in features if c.startswith("z_dept_")]
        else:
            counts = slugs.value_counts()
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            chosen = [s for s, n in ranked if n >= min_positive][:MAX_DEPARTMENTS]
            for s, n in ranked:
                if s not in chosen:
                    self.dropped_features[f"z_dept_{s}"] = int(n)
                    logger.info("department %s (%d signals) pooled into 'other'", s, n)
        for s in chosen:
            df[f"z_dept_{s}"] = (slugs == s).astype(float)
        return df

    def _drop_sparse(self, df: pd.DataFrame, min_positive: int) -> pd.DataFrame:
        drop = []
        for col in df.columns:
            if col.startswith("x_"):
                n = int(df[col].notna().sum())
            elif col.startswith("z_") and col not in CONTINUOUS_Z:
                n = int((df[col] != 0).sum())
            else:
                continue
            if n < min_positive:
                drop.append(col)
                self.dropped_features[col] = n
                logger.info("dropping %s: %d positive cases < %d", col, n, min_positive)
        return df.drop(columns=drop)

    @staticmethod
    def _typed(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col in ("signal_date", "bill_date", "data_end") or col.startswith("x_"):
                df[col] = pd.to_datetime(df[col])
            elif col.startswith("z_"):
                df[col] = pd.to_numeric(df[col]).astype(float)
        df["bill_id"] = pd.array(pd.to_numeric(df["bill_id"]), dtype="Int64")
        return df


def get_takeup_signals(
    as_of: date | str | pd.Timestamp, tables: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Convenience wrapper: ``TakeupData(tables).get_takeup_signals(as_of)``."""
    return TakeupData(tables).get_takeup_signals(as_of)
