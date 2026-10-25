"""Loaders: Tier B fetch cache -> raw take-up tables (DATA_CONTRACT §10.0a, §10.0b).

``build_*`` functions read only local files (the fetch cache under ``data/cache/takeup``)
and return DataFrames with exactly the contract's columns; they never touch a database, so
``--dry-run`` is safe. ``load_*`` functions upsert a frame through a given engine or connection
(``takeup.db.upsert_rows``), in batches and by primary key, so re-runs are idempotent.

Each build reports what it skipped (malformed files or records) in ``df.attrs["report"]``
and in the log, rather than failing on one bad record.
"""

import html
import json
import logging
import re
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Connection, Engine

from westminster_weather.takeup.db import upsert_rows
from westminster_weather.takeup.tables import member_government_posts, source_documents
from westminster_weather.takeup.tier_b import CACHE_DIR, _slug
from westminster_weather.takeup.triggers import SEARCH_PHRASES

log = logging.getLogger(__name__)

SOURCE_COLUMNS = [c.name for c in source_documents.columns]
POST_COLUMNS = [c.name for c in member_government_posts.columns]
HOUSES = {"Commons", "Lords"}
HANSARD_KINDS = {"spoken": ("hansard_spoken", "han"), "written": ("hansard_written", "hanw")}
PHRASE_OF_SLUG = {_slug(p): p for p in SEARCH_PHRASES}
PHRASE_ORDER = {p: k for k, p in enumerate(SEARCH_PHRASES)}
LOAD_BATCH = 500

# HTML -> text: block-level tags become line breaks, other tags are dropped, entities decoded.
_BLOCK_TAG_RE = re.compile(
    r"<\s*(?:br|/p|/div|/li|/h[1-6]|/tr|/table|/ul|/ol|/blockquote)\b[^>]*>", re.I
)
_TAG_RE = re.compile(r"<[^>]*>")
_SPACES_RE = re.compile(r"[ \t\r\f\v\u00a0]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def strip_html(text: Any) -> str:
    """Plain text from an HTML fragment, keeping paragraph breaks as blank lines."""
    if not isinstance(text, str) or not text:
        return ""
    text = _BLOCK_TAG_RE.sub("\n\n", text)
    text = html.unescape(_TAG_RE.sub(" ", text))
    lines = (_SPACES_RE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() in {"", "None", "null"})


def _member_id(value: Any) -> int | None:
    if _blank(value):
        return None
    try:
        member_id = int(value)
    except (TypeError, ValueError):
        return None
    return member_id if member_id > 0 else None


def _date(value: Any) -> date | None:
    """Date from an ISO date or datetime string ('2018-11-05T00:00:00'); None if invalid."""
    if _blank(value):
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _house(value: Any) -> str | None:
    return value if value in HOUSES else None


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _read_json(path: Path, skipped: Counter) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        skipped["unreadable_file"] += 1
        log.warning("skipping unreadable cache file %s: %s", path, exc)
        return None


def _frame(rows: list[dict[str, Any]], columns: list[str], report: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    for col in ("member_id", "bill_id"):
        if col in df:
            df[col] = df[col].astype("Int64")
    for col in ("fetched_at", "ingested_at"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True)
    df.attrs["report"] = report
    return df


# ------------------------------------------------------------- source_documents ----
def _wms_rows(cache_dir: Path, skipped: Counter) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((cache_dir / "wms" / "full").glob("*.json")):
        rec = _read_json(path, skipped)
        if rec is None:
            continue
        if not isinstance(rec, dict):
            skipped["wms_not_a_record"] += 1
            continue
        statement_id, made = rec.get("id"), _date(rec.get("dateMade"))
        if _blank(statement_id) or made is None:
            skipped["wms_missing_id_or_date"] += 1
            log.warning("skipping written statement %s: no id or dateMade", path.name)
            continue
        statement_id = str(statement_id).strip()
        rows.append(
            {
                "doc_id": f"wms-{statement_id}",
                "source": "wms",
                "source_id": statement_id,
                "event_date": made,
                "member_id": _member_id(rec.get("memberId")),
                "house": _house(rec.get("house")),
                "title": "" if _blank(rec.get("title")) else str(rec["title"]).strip(),
                "department": None
                if _blank(rec.get("answeringBodyName"))
                else str(rec["answeringBodyName"]).strip(),
                "text": strip_html(rec.get("text")),
                "matched_phrases": None,
                "fetched_at": _mtime(path),
            }
        )
    return rows


def _hansard_rows(cache_dir: Path, skipped: Counter) -> list[dict[str, Any]]:
    """One row per unique contribution; ``matched_phrases`` = every phrase that found it."""
    docs: dict[str, dict[str, Any]] = {}
    phrases: dict[str, set[str]] = {}
    for kind, (source, prefix) in HANSARD_KINDS.items():
        for path in sorted((cache_dir / "hansard" / kind).glob("*/*.json")):
            slug = path.parent.name
            phrase = PHRASE_OF_SLUG.get(slug)
            if phrase is None:
                log.warning("unknown Hansard phrase slug %r; recorded as the slug", slug)
                phrase = slug.replace("-", " ")
            results = _read_json(path, skipped)
            if results is None:
                continue
            if not isinstance(results, list):
                skipped["hansard_not_a_list"] += 1
                continue
            for rec in results:
                if not isinstance(rec, dict):
                    skipped["hansard_not_a_record"] += 1
                    continue
                ext_id = rec.get("ContributionExtId")
                source_id = rec.get("ItemId") if _blank(ext_id) else ext_id
                sitting = _date(rec.get("SittingDate"))
                if _blank(source_id) or sitting is None:
                    skipped["hansard_missing_id_or_date"] += 1
                    continue
                source_id = str(source_id).strip()
                doc_id = f"{prefix}-{source_id}"
                phrases.setdefault(doc_id, set()).add(phrase)
                if doc_id in docs:
                    continue
                text = rec.get("ContributionTextFull")
                text = rec.get("ContributionText") if _blank(text) else text
                title = rec.get("DebateSection")
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "source": source,
                    "source_id": source_id,
                    "event_date": sitting,
                    "member_id": _member_id(rec.get("MemberId")),
                    "house": _house(rec.get("House")),
                    "title": "" if _blank(title) else str(title).strip(),
                    "department": None,
                    "text": strip_html(text),
                    "matched_phrases": None,
                    "fetched_at": _mtime(path),
                }
    for doc_id, doc in docs.items():
        ordered = sorted(phrases[doc_id], key=lambda p: (PHRASE_ORDER.get(p, len(PHRASE_ORDER)), p))
        doc["matched_phrases"] = ",".join(ordered)
    return list(docs.values())


def build_source_documents(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """§10.0a rows: every cached written statement plus every unique Hansard contribution.

    ``attrs["report"]`` holds the counts per source and the skipped records by reason."""
    skipped: Counter = Counter()
    rows = _wms_rows(Path(cache_dir), skipped) + _hansard_rows(Path(cache_dir), skipped)
    df = _frame(rows, SOURCE_COLUMNS, {})
    dupes = int(df["doc_id"].duplicated().sum())
    if dupes:
        skipped["duplicate_doc_id"] += dupes
        df = df.drop_duplicates("doc_id").reset_index(drop=True)
    df.attrs["report"] = {
        "rows": len(df),
        "by_source": df["source"].value_counts().to_dict(),
        "text_bytes": int(df["text"].map(lambda t: len(t.encode("utf-8"))).sum()),
        "skipped": dict(skipped),
    }
    if skipped:
        log.warning("source_documents: skipped %s", dict(skipped))
    return df


# ------------------------------------------------------- member_government_posts ----
def build_member_government_posts(cache_dir: Path = CACHE_DIR) -> tuple[pd.DataFrame, set[int]]:
    """§10.0b rows from the cached Members API posts, and the set of member IDs fetched.

    A fetched member with no rows held no government post; a member not in the set was not
    fetched. Duplicate (member_id, post_name, start_date) keys keep the first row."""
    skipped: Counter = Counter()
    fetched: set[int] = set()
    rows = []
    for path in sorted((Path(cache_dir) / "members").glob("*.json")):
        member_id = _member_id(path.stem)
        posts = _read_json(path, skipped)
        if posts is None:
            continue
        if member_id is None or not isinstance(posts, list):
            skipped["malformed_member_file"] += 1
            log.warning("skipping malformed members cache file %s", path.name)
            continue
        fetched.add(member_id)
        for post in posts:
            start = _date(post.get("start")) if isinstance(post, dict) else None
            if start is None or _blank(post.get("name")):
                skipped["post_missing_name_or_start"] += 1
                continue
            rows.append(
                {
                    "member_id": member_id,
                    "post_name": str(post["name"]).strip(),
                    "start_date": start,
                    "end_date": _date(post.get("end")),
                    "fetched_at": _mtime(path),
                }
            )
    df = _frame(rows, POST_COLUMNS, {})
    pk = ["member_id", "post_name", "start_date"]
    dupes = int(df.duplicated(pk).sum())
    if dupes:
        skipped["duplicate_pk"] += dupes
        df = df.drop_duplicates(pk).reset_index(drop=True)
    with_posts = int(df["member_id"].nunique())
    df.attrs["report"] = {
        "rows": len(df),
        "members_fetched": len(fetched),
        "members_with_posts": with_posts,
        "members_without_posts": len(fetched) - with_posts,
        "open_posts": int(df["end_date"].isna().sum()),
        "skipped": dict(skipped),
    }
    log.info(
        "member_government_posts: %d members fetched (%d with posts, %d with none), %d rows",
        len(fetched),
        with_posts,
        len(fetched) - with_posts,
        len(df),
    )
    if skipped:
        log.warning("member_government_posts: skipped %s", dict(skipped))
    return df, fetched


# ------------------------------------------------------------------------- load ----
Bind = Engine | Connection


def load_source_documents(bind: Bind, df: pd.DataFrame, batch_size: int = LOAD_BATCH) -> int:
    return upsert_rows(bind, source_documents, df[SOURCE_COLUMNS], batch_size)


def load_member_government_posts(bind: Bind, df: pd.DataFrame, batch_size: int = LOAD_BATCH) -> int:
    return upsert_rows(bind, member_government_posts, df[POST_COLUMNS], batch_size)
