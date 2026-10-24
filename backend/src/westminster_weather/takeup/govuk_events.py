"""Tier B1: GOV.UK publication events for ideas (DATA_CONTRACT §10.5 ``idea_events``).

Events come from the GOV.UK search API (https://www.gov.uk/api/search.json), 2006-2026:

* consultations: ``open_consultation`` / ``closed_consultation`` / ``consultation_outcome``.
  A consultation is dated by its opening (``start_date``, else ``public_timestamp``); a
  ``consultation_outcome`` also gives a ``consultation_response`` dated by its
  ``public_timestamp``.
* green / white papers: ``policy_paper`` documents, plus publication-like documents found by
  phrase search, whose title says "green paper" / "white paper" (not responses to one).
* draft bills: documents whose title contains "draft" followed by "bill".

Each event is matched to ideas by TF-IDF similarity (word and character n-grams) between the
event's title + description and the idea's topic + the titles of its mentions, with a small
bonus when the idea's department is one of the event's organisations. Only events between
``trigger - 2 years`` and ``trigger + 5 years`` are considered; the best few per idea are kept.

Leakage: nothing here reads bills, ``idea_bill_links`` or any production reference table.
Raw pages are cached under ``backend/data/cache/takeup/govuk/`` (download cache only).
"""

import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine

from westminster_weather.takeup.tier_b import CACHE_DIR, PoliteClient

log = logging.getLogger(__name__)

SEARCH_URL = "https://www.gov.uk/api/search.json"
GOVUK_CACHE = CACHE_DIR / "govuk"
BASE_URL = "https://www.gov.uk"
PAGE_SIZE = 1000
MIN_INTERVAL = 0.25  # seconds between GOV.UK requests (<= 4 per second)
FROM_DATE = date(2006, 1, 1)
TO_DATE = date(2026, 12, 31)
FIELDS = [
    "title",
    "link",
    "public_timestamp",
    "content_store_document_type",
    "organisations",
    "description",
    "start_date",
    "end_date",
]
CONSULTATION_TYPES = ["open_consultation", "closed_consultation", "consultation_outcome"]
# Publication-like document types searched for paper / draft bill titles (no news, speeches,
# press releases or tribunal decisions).
PUBLICATION_TYPES = [
    "policy_paper",
    "guidance",
    "corporate_report",
    "independent_report",
    "impact_assessment",
    "document_collection",
    "notice",
    "decision",
    *CONSULTATION_TYPES,
]
EVENT_TYPES = ("consultation", "consultation_response", "green_paper", "white_paper", "draft_bill")


@dataclass(frozen=True)
class Query:
    name: str
    params: dict[str, Any]
    max_results: int | None = None  # None = every page


QUERIES = [
    Query("consultations", {"filter_content_store_document_type": CONSULTATION_TYPES}),
    Query("policy_papers", {"filter_content_store_document_type": ["policy_paper"]}),
    Query(
        "q_white_paper",
        {"q": '"white paper"', "filter_content_store_document_type": PUBLICATION_TYPES},
        max_results=3000,
    ),
    Query(
        "q_green_paper",
        {"q": '"green paper"', "filter_content_store_document_type": PUBLICATION_TYPES},
        max_results=3000,
    ),
    Query(
        "q_draft_bill",
        {"q": '"draft bill"', "filter_content_store_document_type": PUBLICATION_TYPES},
        max_results=3000,
    ),
]

# ---------------------------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------------------------


class GovukClient(PoliteClient):
    """PoliteClient (retries on 429/5xx) with at least ``MIN_INTERVAL`` s between requests."""

    def _wait_turn(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(host, now))
            self._next_slot[host] = slot + MIN_INTERVAL
        time.sleep(max(0.0, slot - now))


def _params(query: Query, start: int) -> dict[str, Any]:
    return {
        **query.params,
        "filter_public_timestamp": f"from:{FROM_DATE},to:{TO_DATE}",
        "fields": FIELDS,
        "count": PAGE_SIZE,
        "start": start,
        **({} if "q" in query.params else {"order": "public_timestamp"}),
    }


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _pages(query: Query, client: PoliteClient | None, cache: Path) -> Iterator[dict]:
    """Every page of ``query``, from the cache when present, else fetched and cached."""
    start = 0
    while True:
        path = cache / query.name / f"{start:06d}.json"
        if path.exists():
            page = json.loads(path.read_text())
        else:
            if client is None:
                return
            page = client.get_json(SEARCH_URL, params=_params(query, start))
            _write(path, page)
        yield page
        start += PAGE_SIZE
        limit = page.get("total", 0)
        if query.max_results is not None:
            limit = min(limit, query.max_results)
        if start >= limit or not page.get("results"):
            return


def fetch_govuk(cache: Path = GOVUK_CACHE, offline: bool = False) -> list[dict]:
    """Raw search results from every query (cached); duplicates across queries are kept."""
    client = None if offline else GovukClient()
    out: list[dict] = []
    printed = False
    try:
        for query in QUERIES:
            for page in _pages(query, client, cache):
                results = page.get("results", [])
                if results and not printed:
                    print(f"GOV.UK first record keys: {sorted(results[0])}")
                    printed = True
                out.extend(results)
    finally:
        if client is not None:
            client.close()
    return out


# ---------------------------------------------------------------------------------------------
# Parsing and classification
# ---------------------------------------------------------------------------------------------

_DRAFT_BILL = re.compile(r"\bdraft\b.*\bbills?\b", re.I)
_WHITE = re.compile(r"\bwhite\s+paper\b", re.I)
_GREEN = re.compile(r"\bgreen\s+paper\b", re.I)
# Titles about someone else's paper (responses, submissions, evidence) are not the paper.
_NOT_THE_PAPER = re.compile(
    r"\b(response|responses|responding|submission|evidence|summary of|scrutiny|review of)\b",
    re.I,
)


def _to_date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _paper_type(title: str) -> str | None:
    """'draft_bill' / 'white_paper' / 'green_paper' from a title, or None."""
    if _NOT_THE_PAPER.search(title):
        return None
    if _DRAFT_BILL.search(title):
        return "draft_bill"
    if _WHITE.search(title):
        return "white_paper"
    if _GREEN.search(title):
        return "green_paper"
    return None


def classify(record: dict) -> list[tuple[str, date]]:
    """(event_type, event_date) pairs for one search result (possibly none).

    Consultations: the opening is a ``consultation`` (or the paper / draft bill its title
    names), dated ``start_date`` (else ``public_timestamp``, except for outcomes); an outcome
    adds a ``consultation_response`` dated by its ``public_timestamp``. Other documents count
    only when their title names a white / green paper or a draft bill.
    """
    title = (record.get("title") or "").strip()
    doc_type = record.get("content_store_document_type") or ""
    published = _to_date(record.get("public_timestamp"))
    opened = _to_date(record.get("start_date"))
    out: list[tuple[str, date]] = []
    if doc_type in CONSULTATION_TYPES:
        opening = opened or (published if doc_type != "consultation_outcome" else None)
        if opening is not None:
            out.append((_paper_type(title) or "consultation", opening))
        if doc_type == "consultation_outcome" and published is not None:
            out.append(("consultation_response", published))
    else:
        kind = _paper_type(title)
        if kind is not None and published is not None:
            out.append((kind, published))
    return [(k, d) for k, d in out if FROM_DATE <= d <= TO_DATE]


def _organisations(record: dict) -> str:
    orgs = record.get("organisations") or []
    names = [o.get("title") or o.get("acronym") or "" for o in orgs if isinstance(o, dict)]
    return "; ".join(n for n in names if n)


def parse_results(results: list[dict]) -> pd.DataFrame:
    """One row per (event_type, source_url, event_date); columns ``event_type``,
    ``event_date``, ``title``, ``description``, ``source_url``, ``organisations``,
    ``doc_type``."""
    rows = []
    for record in results:
        link = record.get("link") or record.get("_id") or ""
        if not link:
            continue
        url = link if link.startswith("http") else BASE_URL + link
        for event_type, event_date in classify(record):
            rows.append(
                {
                    "event_type": event_type,
                    "event_date": event_date,
                    "title": (record.get("title") or "").strip(),
                    "description": (record.get("description") or "").strip(),
                    "source_url": url,
                    "organisations": _organisations(record),
                    "doc_type": record.get("content_store_document_type") or "",
                }
            )
    cols = [
        "event_type",
        "event_date",
        "title",
        "description",
        "source_url",
        "organisations",
        "doc_type",
    ]
    df = pd.DataFrame(rows, columns=cols)
    return df.drop_duplicates(["event_type", "source_url", "event_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------------------------

# Words that say what kind of document something is, not what it is about.
GENERIC_WORDS = {
    "act",
    "amendment",
    "bill",
    "bills",
    "call",
    "consultation",
    "consultations",
    "department",
    "draft",
    "evidence",
    "government",
    "green",
    "introduce",
    "legislation",
    "legislate",
    "minister",
    "new",
    "outcome",
    "paper",
    "papers",
    "plans",
    "policy",
    "proposals",
    "proposed",
    "reform",
    "reforms",
    "regulations",
    "response",
    "review",
    "secretary",
    "state",
    "statement",
    "strategy",
    "uk",
    "white",
    "year",
}
WORD_WEIGHT = 0.7
CHAR_WEIGHT = 0.3
ORG_BONUS = 0.05
MIN_WORD_SIM = 0.15  # an event must share real words with the idea, not just letters
THRESHOLD = 0.28  # chosen on the hand-made fixture in tests/takeup/test_takeup_govuk.py
TOP_K = 3
WINDOW_BEFORE = timedelta(days=730)
WINDOW_AFTER = timedelta(days=round(5 * 365.25))
MAX_MENTION_TITLES = 10

_ORG_NOISE = re.compile(
    r"\b(the|hm|her majesty's|his majesty's|department|ministry|office|for|of|and|"
    r"cabinet)\b",
    re.I,
)


def _org_key(name: str) -> str:
    return re.sub(r"[^a-z]+", " ", _ORG_NOISE.sub(" ", name.lower())).strip()


def _org_match(department: Any, organisations: str) -> bool:
    if not isinstance(department, str) or not department.strip() or not organisations:
        return False
    key = _org_key(department)
    if not key:
        return False
    return any(key == _org_key(o) for o in organisations.split(";"))


def idea_texts(ideas_df: pd.DataFrame, mentions_df: pd.DataFrame) -> pd.Series:
    """idea_id -> topic + the distinct titles of its mentions (``mentions_df``: idea_id, title)."""
    titles: dict[str, list[str]] = {}
    if not mentions_df.empty:
        m = mentions_df.dropna(subset=["title"])
        if "event_date" in m.columns:
            m = m.sort_values("event_date")
        for idea_id, group in m.groupby("idea_id", sort=False):
            titles[idea_id] = list(dict.fromkeys(group["title"].astype(str)))[:MAX_MENTION_TITLES]
    return pd.Series(
        {
            row.idea_id: " . ".join([str(row.topic), *titles.get(row.idea_id, [])])
            for row in ideas_df.itertuples()
        },
        dtype=object,
    )


def _vectorizers():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    words = TfidfVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS | GENERIC_WORDS),
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), sublinear_tf=True)
    return words, chars


def _strip_generic(text: str) -> str:
    return re.sub(
        r"\b(" + "|".join(sorted(GENERIC_WORDS, key=len, reverse=True)) + r")\b",
        " ",
        text,
        flags=re.I,
    )


def score_matrix(idea_text: list[str], event_text: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(word similarity, char similarity) matrices, ideas x events (rows L2-normalised)."""
    words, chars = _vectorizers()
    corpus = idea_text + event_text
    wm = words.fit_transform(corpus)
    cm = chars.fit_transform([_strip_generic(t).lower() for t in corpus])
    n = len(idea_text)
    w = (wm[:n] @ wm[n:].T).toarray()
    c = (cm[:n] @ cm[n:].T).toarray()
    return w, c


IDEA_EVENT_COLUMNS = ["idea_id", "event_type", "event_date", "title", "source_url", "score"]


def match_events(
    ideas_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    events_df: pd.DataFrame,
    threshold: float = THRESHOLD,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """``idea_events`` candidate rows (plus ``score``) for ideas.

    ``ideas_df``: idea_id, topic, trigger_date, department (optional).
    ``mentions_df``: idea_id, title (+ optional event_date) - the mention titles of each idea.
    ``events_df``: from :func:`parse_results`. Bills are never used.
    Rows are unique on (idea_id, event_type, event_date), the ``idea_events`` primary key.
    """
    if ideas_df.empty or events_df.empty:
        return pd.DataFrame(columns=IDEA_EVENT_COLUMNS)
    ideas = ideas_df.reset_index(drop=True)
    events = events_df.reset_index(drop=True)
    texts = idea_texts(ideas, mentions_df)
    idea_text = [texts[i] for i in ideas["idea_id"]]
    # De-duplicate event texts so the same document (e.g. a consultation and its response)
    # costs one column.
    event_text_all = (
        events["title"].fillna("") + " . " + events["description"].fillna("")
    ).tolist()
    w, c = score_matrix(idea_text, event_text_all)
    score = WORD_WEIGHT * w + CHAR_WEIGHT * c
    orgs = events["organisations"].fillna("") if "organisations" in events else None
    ev_dates = pd.to_datetime(events["event_date"]).to_numpy()
    rows = []
    for i, idea in ideas.iterrows():
        trigger = pd.Timestamp(idea["trigger_date"])
        in_window = (ev_dates >= (trigger - WINDOW_BEFORE).to_datetime64()) & (
            ev_dates <= (trigger + WINDOW_AFTER).to_datetime64()
        )
        s = score[i].copy()
        if orgs is not None:
            dept = idea.get("department")
            bonus = np.array([_org_match(dept, o) for o in orgs]) * ORG_BONUS
            s = s + bonus
        ok = in_window & (w[i] >= MIN_WORD_SIM) & (s >= threshold)
        for j in np.flatnonzero(ok):
            ev = events.iloc[j]
            rows.append(
                {
                    "idea_id": idea["idea_id"],
                    "event_type": ev["event_type"],
                    "event_date": ev["event_date"],
                    "title": ev["title"],
                    "source_url": ev["source_url"],
                    "score": float(s[j]),
                }
            )
    out = pd.DataFrame(rows, columns=IDEA_EVENT_COLUMNS)
    if out.empty:
        return out
    out = out.sort_values(["idea_id", "score"], ascending=[True, False])
    out = out.drop_duplicates(["idea_id", "event_type", "event_date"])
    out = out.groupby("idea_id", sort=False).head(top_k)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Loading and the pipeline step
# ---------------------------------------------------------------------------------------------


def load_idea_events(bind: Engine | Connection, rows: pd.DataFrame) -> int:
    """Upsert ``idea_events`` rows via ``takeup.db.upsert_rows`` (which refuses production)."""
    from westminster_weather.takeup.db import upsert_rows

    if rows.empty:
        return 0
    now = datetime.now(UTC)
    records = rows[["idea_id", "event_type", "event_date", "title", "source_url"]].copy()
    records["ingested_at"] = now
    return upsert_rows(bind, "idea_events", records)


def _mention_titles(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    im = tables["idea_mentions"][["idea_id", "mention_id"]]
    sm = tables["signal_mentions"][["mention_id", "title", "event_date"]]
    return im.merge(sm, on="mention_id", how="inner")[["idea_id", "title", "event_date"]]


def run_govuk_events(dry_run: bool = True, offline: bool = False) -> dict[str, Any]:
    """Fetch (cached) GOV.UK events, match them to the take-up DB's ideas, and upsert
    ``idea_events`` unless ``dry_run``; returns counts. Reads only ``ideas``,
    ``idea_mentions`` and ``signal_mentions`` (never ``idea_bill_links`` or bills)."""
    from westminster_weather.takeup import db

    events = parse_results(fetch_govuk(offline=offline))
    tables = db.load_takeup_tables()
    ideas = tables["ideas"]
    matched = match_events(ideas, _mention_titles(tables), events)
    written = 0
    if not dry_run and not matched.empty:
        engine = db.get_takeup_engine()
        try:
            written = load_idea_events(engine, matched)
        finally:
            engine.dispose()
    return {
        "events": len(events),
        "events_by_type": events["event_type"].value_counts().to_dict(),
        "ideas": len(ideas),
        "ideas_with_events": int(matched["idea_id"].nunique()) if not matched.empty else 0,
        "idea_events": len(matched),
        "idea_events_by_type": matched["event_type"].value_counts().to_dict()
        if not matched.empty
        else {},
        "written": written,
        "dry_run": dry_run,
        "matches": matched,
    }
