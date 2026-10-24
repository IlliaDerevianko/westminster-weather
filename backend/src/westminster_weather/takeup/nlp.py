"""Tier B2-B4 + B6: commitments -> topic groups -> ideas -> bill links; review and score.

Ported from the prototype ``law-in-time/nlp_signals.py`` (label / group / link / merge /
review / score), reading the raw sources from the take-up database instead of JSONL files.

    label   every ``source_documents`` row is split into sentences; each sentence matching the
            trigger lexicon (``triggers.classify_sentence``) becomes a ``signal_mentions`` row
            with firmness 0/1/2, the matched phrase, a snippet and a rule topic. A cheap extra
            gate (``commitment_check``) rejects hits that are not commitments to legislate
            ("I will update the House in due course", "the draft Bill lets platforms off the
            hook", "we are introducing the national living wage"). The speaker must be a
            minister on the day: always for written ministerial statements; for Hansard, from
            ``member_government_posts``. Every trigger sentence is stored; only rows with
            ``is_commitment`` and ``is_minister`` true are grouped (``get_takeup_signals``
            reads only those).
    group   TF-IDF (words + character n-grams, ``DOMAIN_STOP`` removed) of topic + title,
            agglomerative clustering (cosine, average linkage), then clusters are split where
            consecutive mentions are more than ``MAX_GAP_DAYS`` apart. ``group_id`` is
            ``g-<earliest mention_id>``: deterministic and stable while that mention exists.
    link    each group is matched to the Government Bill first read in
            [trigger - 7 days, trigger + 5 years] with the most similar title + long title:
            ``auto`` above ``LINK_AUTO``, ``rule`` (link if above the midpoint) or an optional
            Gemini judge between ``LINK_CHECK`` and ``LINK_AUTO``, no link below.
    merge   groups linked to the same bill form one idea (earliest trigger kept); each
            mention's pre-merge ``group_id`` is kept in ``idea_mentions`` so
            ``get_takeup_signals(as_of)`` can split them before the bill's first reading
            (DATA_CONTRACT §10.4). Ideas are numbered ``nlp-00001`` ... by trigger date, then
            group id.
    write   the derived tables (signal_mentions, ideas, idea_mentions, idea_bill_links) are
            cleared and rewritten in one transaction; the raw sources are never touched.
    review  samples of mentions and links to hand-label; ``score`` reports precision.

Rules-only by default. ``GeminiClient`` (needs ``GEMINI_API_KEY``) can confirm commitments,
clean topics and judge borderline links; it is used only when passed in explicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine, func, select

from westminster_weather.takeup import db as takeup_db
from westminster_weather.takeup.bill_text import first_readings
from westminster_weather.takeup.tables import (
    idea_bill_links,
    idea_events,
    idea_mentions,
    ideas,
    signal_mentions,
)
from westminster_weather.takeup.triggers import SENT_SPLIT, classify_sentence

log = logging.getLogger(__name__)

REVIEW_DIR = Path(__file__).resolve().parents[3] / "data" / "takeup" / "review"

# Thresholds for the TF-IDF backend. Grouping as in the prototype; linking raised from the
# prototype's 0.25 / 0.125 because a bill now scores max(sim to title, sim to title + long
# title) (checked on the synthetic self-test and on bills named in the real sentences).
GROUP_SIM, LINK_AUTO, LINK_CHECK = 0.30, 0.30, 0.20
MAX_GAP_DAYS = 730  # split a topic group where mentions are more than 2 years apart
LINK_BEFORE_DAYS = 7  # a bill first read up to 7 days before the trigger can still match
LINK_MAX_YEARS = 5
SOURCE_RANK = {"wms": 0, "hansard_written": 1, "hansard_spoken": 2}  # as in signals.py
FIRMNESS = {"vague": 0, "medium": 1, "firm": 2}

MENTION_COLUMNS = [c.name for c in signal_mentions.columns]
IDEA_COLUMNS = [c.name for c in ideas.columns]
IDEA_MENTION_COLUMNS = [c.name for c in idea_mentions.columns]
LINK_COLUMNS = [c.name for c in idea_bill_links.columns]
DERIVED_TABLES = [idea_bill_links, idea_mentions, ideas, signal_mentions]  # delete order

# Words that say nothing about what a bill is about (prototype list + long-title boilerplate).
DOMAIN_STOP = {
    "legislate", "legislation", "legislating", "legislative", "bill", "bills", "government",
    "introduce", "introducing", "bring", "forward", "new", "will", "ensure", "make", "measures",
    "act", "uk", "reform", "reforms", "provision", "provisions", "connected", "purposes",
    "relation", "respect", "amend", "draft", "primary",
}  # fmt: skip
# Topic words that carry no subject on their own ("ensure that this happens").
GENERIC_WORDS = {
    "this", "that", "these", "those", "it", "its", "them", "they", "matter", "matters", "issue",
    "issues", "area", "areas", "change", "changes", "proposal", "proposals", "necessary",
    "needed", "required", "requirement", "happens", "happen", "place", "effect", "deliver",
    "delivering", "implement", "implementing", "implementation", "give", "giving", "put",
    "set", "out", "take", "taking", "need", "needs", "appropriate", "relevant", "further",
    "details", "detail", "house", "update", "shortly", "soon", "possible", "opportunity",
    "earliest", "accordingly", "there", "here", "also", "already", "way", "ways", "point",
    "commitment", "commitments", "plans", "plan", "steps", "step", "work", "position",
    "approach", "parliament", "parliamentary", "time", "session", "course", "due", "allows",
    "permits", "soonest", "pledge", "promise", "manifesto", "things", "thing", "what",
}  # fmt: skip

# ------------------------------------------------------------------- commitment gate ----
_GOV = r"(?:we|i|the government|this government|ministers|my department|the department)"
LEG_WORD_RE = re.compile(
    r"\b(?:legislat\w*|bills?|statut\w*|acts? of parliament|primary powers)\b"
    r"|\bamend\w*\s+(?:the\s+)?(?:[\w’'()-]+\s+){0,6}?act\b",
    re.I,
)
COMMIT_RE = re.compile(
    rf"\b{_GOV}\s+(?:\w+\s+){{0,2}}?(?:will|shall|intends?|plans?|proposes?|are going to|"
    r"is going to|am going to|aims?|expects?|hopes?|want|are committed|is committed|"
    r"am committed|remains? committed)\b"
    r"|\bwill be (?:introduced|brought forward|laid|published|legislated|included)\b"
    r"|\bis to be (?:introduced|published|brought forward)\b"
    r"|\b(?:and|but) will (?:introduce|legislate|bring forward|bring in|publish)\b"
    r"|\b(?:have|has|are|is|remains?) (?:already |firmly |now )?committed to "
    r"(?:bring\w*|introduc\w*|legislat\w*)"
    r"|\blook forward to (?:introducing|bringing forward|legislating)\b"
    r"|\bthe department\b[^.]{0,60}\bwill\b",
    re.I,
)
# Secondary legislation is not a bill: rejected unless primary legislation / a bill is named.
SECONDARY_RE = re.compile(
    r"\b(?:secondary legislation|statutory instruments?|remedial order|regulations|"
    r"commencement order|order in council)\b",
    re.I,
)
PRIMARY_RE = re.compile(r"\b(?:primary legislation|bills?|legislate|legislating)\b", re.I)
DRAFT_COMMIT_RE = re.compile(
    rf"\b{_GOV}\s+(?:\w+\s+){{0,3}}?(?:will|shall|intends? to|plans? to|proposes? to|"
    r"are going to|is going to|am going to|aims? to|expects? to|hopes? to|"
    r"(?:are|is|am|remains?) committed to)\s+(?:\w+\s+){0,2}?"
    r"(?:publish\w*|bring\w* forward|introduc\w*|lay\w*)\b"
    r"|\bdraft (?:bill|legislation)\b[^.]{0,80}?\bwill be (?:published|laid|brought forward)\b",
    re.I,
)
TIMING_ONLY_RE = re.compile(
    r"^(?:at the earliest|as soon as parliamentary|when parliamentary|when legislative|"
    r"in due course|at an appropriate time|in the fullness of time)",
    re.I,
)


def commitment_check(sentence: str, phrase: str) -> tuple[bool, str | None]:
    """Extra rules on a trigger hit: ``(is_commitment, reason if not)``.

    Every hit must mention legislation (a bare "in due course" or "we are introducing" is not
    a promise of a bill). Timing-only phrases also need the government (or a minister) to
    commit to something ("we will", "legislation will be introduced"); "draft bill" needs a
    promise to publish one (not "the draft Bill lets ...", not "we published a draft Bill").
    """
    p = (phrase or "").strip().lower()
    if not LEG_WORD_RE.search(sentence):
        return False, "no_legislation_word"
    if SECONDARY_RE.search(sentence) and not PRIMARY_RE.search(sentence):
        return False, "secondary_legislation"
    if p.startswith("draft"):
        if not DRAFT_COMMIT_RE.search(sentence):
            return False, "draft_bill_not_promised"
    elif TIMING_ONLY_RE.match(p) and not COMMIT_RE.search(sentence):
        return False, "timing_phrase_without_commitment"
    return True, None


# -------------------------------------------------------------------------- topics ----
STOP_START = re.compile(
    r"^(?:to|on|for|that|which|in order to|so that|about|with|a|an|the|this|our|its|their|"
    r"some|any|new|primary|further|necessary|legislation|legislative|bill|bills|measures|"
    r"provisions?|now|also|shortly|soon|then|and|or|as|of)\s+",
    re.I,
)
CUT_RE = re.compile(
    r"[.;:,(\[–—]|\b(?:when|as soon as|at the earliest|in due course|this session|shortly|"
    r"later this year|early next year|next year|in the (?:coming|next|new|forthcoming)|"
    r"at an appropriate|in the fullness|subject to|which will|so that|once|by the end of)\b",
    re.I,
)
LEG_ANCHOR_RE = re.compile(r"\b(?:legislat\w*|bill|legislation)\b", re.I)
ANCHOR_TAIL_RE = re.compile(r"\s+(?:to|on|for|about|that|which|covering|tackling)\b", re.I)
GOV_LEAD_RE = re.compile(
    rf"^.*?\b{_GOV}\s+(?:\w+\s+){{0,3}}?(?:will|shall|intends? to|plans? to|are|is)\s+", re.I
)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z-]+")
TITLE_NOISE_RE = re.compile(
    r"\s*(?:\[HL\]|\((?:morning|afternoon|\w+) sitting\)|\(\d+\w* sitting\))\s*", re.I
)
GENERIC_TITLE_RE = re.compile(
    r"^(?:business of the house|topical questions|engagements|oral answers.*|"
    r"points? of order|(?:the )?(?:queen|king)['’]s speech.*|debate on the address.*|"
    r"business without debate|arrangement of business|prime minister|speaker['’]s statement|"
    r"preamble|orders? of the day|written statements?|deferred divisions?|questions?|"
    r"(?:new )?clause \d+.*|schedule \d+.*|amendments?\b.*|budget resolutions.*|"
    r"contingencies fund advance|written ministerial statements?|opposition day.*|"
    r"backbench business|bill presented|royal assent|business of the lords)$",
    re.I,
)
TOPIC_WORDS = 12


def _content_words(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = (w.lower() for w in WORD_RE.findall(text or ""))
    return [
        w
        for w in words
        if len(w) > 2 and w not in ENGLISH_STOP_WORDS and w not in DOMAIN_STOP
        and w not in GENERIC_WORDS
    ]  # fmt: skip


def is_specific(topic: str) -> bool:
    """A topic is specific if it has at least two words that say what it is about."""
    return len(_content_words(topic)) >= 2


VERB_START_RE = re.compile(
    r"^(?:will|would|shall|is|are|was|were|has|have|had|be|been|can|could|should|may|might|"
    r"must|did|does|do)\b",
    re.I,
)
PHRASE_MODIFIER_RE = re.compile(
    r"(?:introduce|bring forward|bring in|lay|publish|introducing|bringing forward)\s+"
    r"((?:[\w’'-]+\s+){1,3}?)(?:bill|legislation)\b",
    re.I,
)
ARTICLES = {"a", "an", "the", "new", "primary", "emergency", "this", "that", "our", "further"}


def _clip(text: str) -> str:
    """The clause up to the first boundary, without leading filler, at most 12 words."""
    text = (text or "").lstrip(" ,;:—–-\"'“”‘’")
    text = CUT_RE.split(text, maxsplit=1)[0].strip()
    if VERB_START_RE.match(text):  # "the draft Bill will be published ..." names no topic
        return ""
    prev = None
    while prev != text:
        prev, text = text, STOP_START.sub("", text.strip())
    words = text.split()[:TOPIC_WORDS]
    while words and words[-1].lower() in {"and", "or", "the", "a", "an", "to", "of", "for"}:
        words.pop()
    return " ".join(words)


def rule_topic(sentence: str, phrase: str | None) -> str:
    """Topic from the words after the trigger phrase, up to the end of the clause (prototype),
    with leading filler stripped. If that is generic ("ensure that this happens"), the words
    before the phrase are tried ("legislation on X will be introduced in due course")."""
    pos = sentence.lower().find(phrase.lower()) if phrase else -1
    after = sentence[pos + len(phrase) :] if pos >= 0 else sentence
    # words inside the phrase that name the bill: "we will introduce a water Bill"
    mod = PHRASE_MODIFIER_RE.search(phrase or "")
    words = [w for w in (mod.group(1).split() if mod else []) if w.lower() not in ARTICLES]
    prefix = " ".join(words)
    candidates = [_clip(after)]
    anchors = list(LEG_ANCHOR_RE.finditer(after))
    if anchors:  # "..., to introduce legislation that tackles SLAPPs"
        candidates.append(_clip(after[anchors[0].end() :]))
    if pos > 0:
        # only "legislation on X will be ..." or "the Government are committed to X, and ...",
        # not an unrelated clause ("They received a positive response, and we intend ...")
        before = sentence[:pos]
        anchors = [
            a for a in LEG_ANCHOR_RE.finditer(before) if ANCHOR_TAIL_RE.match(before[a.end() :])
        ]
        if anchors:
            candidates.append(_clip(before[anchors[-1].end() :]))
        elif GOV_LEAD_RE.match(before):
            candidates.append(_clip(GOV_LEAD_RE.sub("", before)))
    candidates = [f"{prefix} {c}".strip() if prefix else c for c in candidates]
    topic = next((c for c in candidates if is_specific(c)), candidates[0])
    return topic or prefix or _clip(sentence) or sentence[:80].strip()


def clean_title(title: Any) -> str:
    """Statement title / debate section without sitting markers and '[HL]'."""
    if not isinstance(title, str):
        return ""
    return re.sub(r"\s+", " ", TITLE_NOISE_RE.sub(" ", title)).strip()


def _norm_bill_title(title: str) -> str:
    t = clean_title(title).lower().replace("’", "'")
    t = re.sub(r"^draft\s+", "", t)
    t = re.sub(r"\s+(?:act|bill)(?:\s+\d{4})?(?:\s+\[hl\])?$", "", t)
    t = re.sub(r"\s+\d{4}$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def is_generic_title(title: str) -> bool:
    t = clean_title(title)
    return not t or bool(GENERIC_TITLE_RE.match(t))


# --------------------------------------------------------------------- minister check ----
class MinisterLookup:
    """Was member X in a government post on day D? From ``member_government_posts``.

    A member with no rows held no post (False) unless ``fetched`` is given and the member is
    not in it (not fetched: None, unknown). A missing member_id is unknown (None)."""

    def __init__(self, posts: pd.DataFrame, fetched: set[int] | None = None) -> None:
        self.spans: dict[int, list[tuple[date, date | None]]] = {}
        for r in posts.itertuples(index=False):
            end = None if pd.isna(r.end_date) else pd.Timestamp(r.end_date).date()
            self.spans.setdefault(int(r.member_id), []).append(
                (pd.Timestamp(r.start_date).date(), end)
            )
        self.fetched = fetched

    def __call__(self, member_id: Any, day: date) -> bool | None:
        if member_id is None or pd.isna(member_id):
            return None
        mid = int(member_id)
        spans = self.spans.get(mid)
        if spans is None:
            if self.fetched is not None and mid not in self.fetched:
                return None
            return False
        return any(start <= day and (end is None or day <= end) for start, end in spans)


def fetched_members_from_cache(cache_dir: Path | None = None) -> set[int] | None:
    """Members whose posts were fetched (one file per member in the fetch cache), or None if
    the cache is not there (then a member with no rows counts as never a minister)."""
    from westminster_weather.takeup.tier_b import CACHE_DIR

    folder = Path(cache_dir or CACHE_DIR) / "members"
    if not folder.is_dir():
        return None
    return {int(p.stem) for p in folder.glob("*.json") if p.stem.isdigit()}


# ---------------------------------------------------------------------------- gemini ----
GEMINI_LABEL_PROMPT = """You label extracts from the UK Parliament. Answer in JSON only, with exactly these keys:
{{"is_commitment": true or false,
  "topic": "short noun phrase, max 10 words, what the legislation is about",
  "firmness": "firm" | "medium" | "vague",
  "speaker_role": "minister" | "other"}}
is_commitment: the UK government itself promises or plans legislation (not a question, demand, quote or opposition promise).
firmness: firm = will legislate / bill coming / dated; medium = intend or plan, draft bill, earliest opportunity; vague = when time allows, in due course, under review.
Debate or statement title: {title}
Speaker known to be a minister: {is_minister}
Extract: \"\"\"{snippet}\"\"\""""  # noqa: E501


class GeminiClient:
    """Optional LLM check (off by default): confirms commitments, cleans topics, judges
    borderline links. Needs ``GEMINI_API_KEY``; answers are cached in memory per run."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; run without --gemini (rules only)")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.cache: dict[str, Any] = {}

    def _ask(self, prompt: str) -> dict | None:
        import httpx

        key = hashlib.sha1(prompt.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        for attempt in range(6):
            r = httpx.post(
                self.URL.format(model=self.model),
                params={"key": self.api_key},
                json=body,
                timeout=60,
            )
            if r.status_code in (429, 500, 503):
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            out = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            self.cache[key] = out
            return out
        return None

    def label(self, snippet: str, title: str, is_minister: bool | None) -> dict | None:
        prompt = GEMINI_LABEL_PROMPT.format(title=title, is_minister=is_minister, snippet=snippet)
        return self._ask(prompt)

    def same_bill(self, idea_text: str, bill_text: str) -> bool:
        out = self._ask(
            'Answer in JSON only: {"same": true or false}. Is this bill the legislation that '
            f'the UK government promised here?\nPromise: "{idea_text}"\nBill: "{bill_text}"'
        )
        return bool(out and out.get("same"))


Labeller = Callable[[str, str, bool | None], dict | None]
LinkJudge = Callable[[str, str], bool]


# --------------------------------------------------------------------------- label ----
_PARA_RE = re.compile(r"\n+")


def split_sentences(text: str) -> list[str]:
    """Sentences of a document: paragraphs (line breaks) first, then the trigger splitter, so a
    heading is not glued to the next sentence. The index in this list is the mention index."""
    out = []
    for para in _PARA_RE.split(text or ""):
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            out.extend(s for s in SENT_SPLIT.split(para) if s.strip())
    return out


def existing_bill_titles(bills: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Normalised bill title -> earliest first reading, for 'is this debate about a bill that
    already exists' (``bills`` as from ``prepare_bills``)."""
    out: dict[str, pd.Timestamp] = {}
    for r in bills.itertuples(index=False):
        if pd.isna(r.first_reading):
            continue
        key = _norm_bill_title(r.title)
        fr = pd.Timestamp(r.first_reading)
        if key and (key not in out or fr < out[key]):
            out[key] = fr
    return out


def _title_for_embedding(title: str, day: pd.Timestamp, existing: dict[str, pd.Timestamp]) -> str:
    """The title, or '' if it says nothing about the topic: a generic debate section
    ('Business of the House') or a debate on a bill already first read by that day (the
    commitment is then about something else, or about amendments to that bill)."""
    if is_generic_title(title):
        return ""
    fr = existing.get(_norm_bill_title(title))
    if fr is not None and fr <= day:
        return ""
    return clean_title(title)


def label_documents(
    docs: pd.DataFrame,
    posts: pd.DataFrame,
    fetched_members: set[int] | None = None,
    bills: pd.DataFrame | None = None,
    labeller: Labeller | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Every trigger sentence of every document, as ``signal_mentions`` rows plus helper
    columns: ``kept`` (commitment by a minister), ``drop_reason``, ``embed_title``,
    ``topic_specific``. ``docs`` has the ``source_documents`` columns."""
    now = now or datetime.now(UTC)
    is_minister = MinisterLookup(posts, fetched_members)
    existing = existing_bill_titles(bills) if bills is not None and len(bills) else {}
    rows = []
    for d in docs.sort_values("doc_id").itertuples(index=False):
        sents = split_sentences(d.text)
        hits = []
        for k, s in enumerate(sents):
            level, phrase = classify_sentence(s)
            if level is not None:
                hits.append((k, s, level, phrase))
        if not hits:
            continue
        day = pd.Timestamp(d.event_date).date()
        minister = True if d.source == "wms" else is_minister(d.member_id, day)
        title = d.title if isinstance(d.title, str) else ""
        embed_title = _title_for_embedding(title, pd.Timestamp(day), existing)
        dept = d.department if d.source == "wms" and isinstance(d.department, str) else None
        for k, s, level, phrase in hits:
            snippet = " ".join(sents[max(0, k - 1) : k + 2])
            commit, reason = commitment_check(s, phrase)
            topic = rule_topic(s, phrase)
            specific = is_specific(topic)
            if not specific and embed_title:
                topic, specific = embed_title, is_specific(embed_title)
            row = {
                "mention_id": f"{d.doc_id}-{k}",
                "source": d.source,
                "doc_id": d.doc_id,
                "event_date": day,
                "member_id": None if pd.isna(d.member_id) else int(d.member_id),
                "is_minister": minister,
                "department": dept,
                "title": title,
                "sentence": s,
                "snippet": snippet,
                "phrase": phrase,
                "firmness": int(level),
                "is_commitment": commit,
                "topic": topic,
                "labelled_by": "rules",
                "ingested_at": now,
                "drop_reason": reason,
                "embed_title": embed_title,
                "topic_specific": specific,
            }
            if labeller is not None:
                g = labeller(snippet, title, minister)
                if g:
                    row["is_commitment"] = bool(g.get("is_commitment"))
                    row["topic"] = g.get("topic") or row["topic"]
                    row["firmness"] = FIRMNESS.get(g.get("firmness"), row["firmness"])
                    row["labelled_by"] = "gemini"
                    row["drop_reason"] = None if row["is_commitment"] else "gemini"
                    if g.get("speaker_role") == "other" and d.source != "wms":
                        row["is_minister"] = False
            rows.append(row)
    cols = [*MENTION_COLUMNS, "drop_reason", "embed_title", "topic_specific"]
    m = pd.DataFrame(rows, columns=cols)
    m["is_minister"] = m["is_minister"].astype(object)
    minister_ok = m["is_minister"].map(lambda v: v is True)
    m["kept"] = m["is_commitment"].astype(bool) & minister_ok
    no_minister = m["is_commitment"].astype(bool) & ~minister_ok
    m.loc[no_minister, "drop_reason"] = np.where(
        m.loc[no_minister, "is_minister"].isna(), "minister_unknown", "not_minister"
    )
    return m


# ------------------------------------------------------------------------ embedding ----
class Embedder:
    """TF-IDF on words (1-2 grams) and character 4-5 grams, domain words removed; rows are
    L2-normalised sparse vectors, so a dot product is cosine similarity."""

    backend = "tfidf"

    def __init__(
        self,
        group_sim: float = GROUP_SIM,
        link_auto: float = LINK_AUTO,
        link_check: float = LINK_CHECK,
    ) -> None:
        self.group_sim, self.link_auto, self.link_check = group_sim, link_auto, link_check

    @staticmethod
    def _clean(text: str, stop: set[str] | frozenset[str]) -> str:
        return " ".join(w for w in WORD_RE.findall((text or "").lower()) if w not in stop)

    def fit(self, texts: list[str]) -> Embedder:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

        self.stop = ENGLISH_STOP_WORDS | DOMAIN_STOP
        clean = [self._clean(t, self.stop) or "empty" for t in texts]
        self.wvec = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]+\b"
        ).fit(clean)
        self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), sublinear_tf=True)
        self.cvec.fit(clean)
        return self

    def encode(self, texts: list[str]):
        from scipy.sparse import hstack
        from sklearn.preprocessing import normalize

        clean = [self._clean(t, self.stop) for t in texts]
        w = normalize(self.wvec.transform(clean))
        c = normalize(self.cvec.transform(clean))
        return normalize(hstack([w, c]).tocsr())


def cluster_vectors(X, min_sim: float, block: int = 2000) -> np.ndarray:
    """Agglomerative clustering (cosine, average linkage, distance threshold 1 - min_sim) of
    the rows of sparse ``X``, as ``sklearn.AgglomerativeClustering`` would give on all rows.

    Two clusters merge only if their average similarity exceeds ``min_sim``, which needs at
    least one pair above it, so clusters never cross the connected components of the graph
    "similarity >= min_sim". Each component is clustered on its own (exactly the same result,
    without an N x N matrix). Labels are numbered in order of first row."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import AgglomerativeClustering

    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    rows, cols = [], []
    for i in range(0, n, block):
        s = (X[i : i + block] @ X.T).toarray()
        r, c = np.nonzero(s >= min_sim)
        rows.append(r + i)
        cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    labels = np.empty(n, dtype=object)
    for k in np.unique(comp):
        idx = np.flatnonzero(comp == k)
        if len(idx) == 1:
            labels[idx[0]] = (k, 0)
            continue
        dist = 1.0 - (X[idx] @ X[idx].T).toarray()
        np.fill_diagonal(dist, 0.0)
        sub = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - min_sim,
        ).fit_predict(np.clip(dist, 0.0, 2.0))
        for j, lab in zip(idx, sub, strict=True):
            labels[j] = (k, int(lab))
    order: dict[tuple, int] = {}
    return np.array([order.setdefault(lab, len(order)) for lab in labels], dtype=int)


# --------------------------------------------------------------------------- group ----
def _order(m: pd.DataFrame) -> pd.DataFrame:
    """Mentions in time order; same-day ties by source rank (wms first), then mention_id."""
    m = m.assign(
        _date=pd.to_datetime(m["event_date"]),
        _rank=m["source"].map(SOURCE_RANK).fillna(len(SOURCE_RANK)),
    )
    return m.sort_values(["_date", "_rank", "mention_id"], kind="stable")


def embed_text(m: pd.DataFrame) -> pd.Series:
    title = m["embed_title"] if "embed_title" in m else m["title"]
    return m["topic"].fillna("") + ". " + title.fillna("")


def group_mentions(
    kept: pd.DataFrame, embedder: Embedder | None = None, max_gap_days: int = MAX_GAP_DAYS
) -> pd.DataFrame:
    """Kept mentions with ``cluster`` and ``group_id`` columns (``g-<earliest mention_id>``)."""
    m = _order(kept).reset_index(drop=True)
    if m.empty:
        return m.assign(cluster=pd.Series(dtype=int), group_id=pd.Series(dtype=str))
    emb = (embedder or Embedder()).fit(embed_text(m).tolist())
    m["cluster"] = cluster_vectors(emb.encode(embed_text(m).tolist()), emb.group_sim)
    # split a cluster where consecutive mentions are > max_gap_days apart (a separate attempt)
    m = m.sort_values(["cluster", "_date", "_rank", "mention_id"], kind="stable")
    gap = m.groupby("cluster")["_date"].diff().dt.days.fillna(0) > max_gap_days
    m["_segment"] = gap.groupby(m["cluster"]).cumsum()
    first = m.groupby(["cluster", "_segment"])["mention_id"].transform("first")
    m["group_id"] = "g-" + first.astype(str)
    return _order(m.drop(columns=["_segment"])).reset_index(drop=True)


def _first_day_department(g: pd.DataFrame) -> str | None:
    """Most common department among the first day's mentions (ties: the earliest)."""
    day = g[g["_date"] == g["_date"].iloc[0]]["department"].dropna()
    if day.empty:
        return None
    counts = day.value_counts(sort=False)
    best = counts.max()
    return next(d for d in day if counts[d] == best)


def _representative(g: pd.DataFrame) -> pd.Series:
    """Mention whose topic names the group: first firm specific, first specific, first."""
    spec = g[g["topic_specific"].astype(bool)] if "topic_specific" in g else g.iloc[:0]
    firm = spec[spec["firmness"] >= 2]
    return (firm if len(firm) else spec if len(spec) else g).iloc[0]


GROUP_TEXT_TOPICS = 5  # distinct mention topics (and titles) used to describe a group


def _group_text(g: pd.DataFrame, rep: pd.Series) -> str:
    """Link text of a group: its representative topic and title, then up to
    ``GROUP_TEXT_TOPICS`` distinct specific topics / titles of its other mentions."""
    parts = [rep["topic"], rep.get("embed_title") or ""]
    spec = g[g["topic_specific"].astype(bool)] if "topic_specific" in g else g
    extra = list(spec["topic"]) + list(g.get("embed_title", pd.Series(dtype=str)).fillna(""))
    for t in extra:
        if len(parts) >= 2 + GROUP_TEXT_TOPICS:
            break
        if t and t not in parts:
            parts.append(t)
    return ". ".join(p for p in parts if p)


def summarize_groups(m: pd.DataFrame) -> pd.DataFrame:
    """One row per topic group: trigger, topic, link text, first source, firmness, department."""
    rows = []
    for gid, g in _order(m).groupby("group_id", sort=False):
        first, rep = g.iloc[0], _representative(g)
        rows.append(
            {
                "group_id": gid,
                "trigger_date": first["_date"],
                "topic": rep["topic"],
                "title": rep.get("embed_title") or "",
                "text": _group_text(g, rep),
                "example": rep["sentence"],
                "names": " ".join([*g["sentence"], *g["title"].fillna("")]),
                "n_mentions": len(g),
                "first_source": first["source"],
                "initial_firmness": int(g.loc[g["_date"] == first["_date"], "firmness"].max()),
                "department": _first_day_department(g),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["trigger_date", "group_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------- link ----
def prepare_bills(
    bills: pd.DataFrame, stage_sittings: pd.DataFrame, long_titles: dict[int, str]
) -> pd.DataFrame:
    """Government Bills with a first reading: ``bill_id, title, long_title, first_reading``."""
    fr = first_readings(stage_sittings)
    title_col = "short_title" if "short_title" in bills else "title"
    out = pd.DataFrame(
        {
            "bill_id": bills["bill_id"].astype(int),
            "title": bills[title_col].fillna("").astype(str),
        }
    )
    long_title = out["bill_id"].map(long_titles).fillna("").astype(str)
    # placeholder text on carried-over / reintroduced bills is not a long title
    out["long_title"] = long_title.where(~long_title.str.match(r"(?i)\s*please note"), "")
    out["first_reading"] = pd.to_datetime(out["bill_id"].map(fr))
    out = out.dropna(subset=["first_reading"])
    return out.sort_values(["first_reading", "bill_id"]).reset_index(drop=True)


def _bill_short(b: pd.DataFrame) -> pd.Series:
    return b["title"].map(lambda t: re.sub(r"\s+Act(?:\s+\d{4})?$|\s*\[HL\]$", "", t))


def _bill_text(b: pd.DataFrame) -> pd.Series:
    return _bill_short(b) + ". " + b["long_title"].fillna("")


MIN_NAME_CHARS = 5


def bill_name_patterns(bills: pd.DataFrame) -> list[re.Pattern | None]:
    """Per bill (row order), a pattern for its name followed by 'Bill' ('the Domestic Abuse
    Bill', 'Finance Bill 2017'); None for names too short to be safe."""
    out = []
    for t in bills["title"]:
        name = _norm_bill_title(t)
        ok = len(name) >= MIN_NAME_CHARS and is_specific(name + " x") or len(name) >= 8
        out.append(re.compile(rf"\b{re.escape(name)}\s+bill\b") if ok and name else None)
    return out


def _named_bill(text: str, names: list[re.Pattern | None], ok: np.ndarray) -> int | None:
    """Row of the earliest candidate bill (in the window ``ok``) named in ``text``."""
    if not text:
        return None
    low = text.lower().replace("’", "'")
    for j in np.flatnonzero(ok):  # bills are sorted by first reading
        if names[j] is not None and names[j].search(low):
            return int(j)
    return None


def link_groups(
    groups: pd.DataFrame,
    bills: pd.DataFrame,
    embedder: Embedder | None = None,
    judge: LinkJudge | None = None,
    max_years: int = LINK_MAX_YEARS,
    before_days: int = LINK_BEFORE_DAYS,
) -> pd.DataFrame:
    """Best candidate bill per group (first read in [trigger - 7 days, trigger + 5 years]) and
    the decision: ``bill_id`` set if linked; ``decided_by`` 'auto', 'rule', 'gemini' or None
    (no candidate above ``link_check``)."""
    cols = ["group_id", "candidate_bill_id", "similarity", "bill_id", "decided_by"]
    if groups.empty:
        return pd.DataFrame(columns=cols)
    if bills.empty:
        return pd.DataFrame(
            {"group_id": groups["group_id"], "candidate_bill_id": None, "similarity": np.nan,
             "bill_id": None, "decided_by": None}
        )  # fmt: skip
    btext = _bill_text(bills).tolist()
    emb = (embedder or Embedder()).fit(groups["text"].tolist() + btext)
    gvec = emb.encode(groups["text"].tolist())
    # a long long title dilutes the match, so the short title alone also counts
    sim = np.maximum(
        (gvec @ emb.encode(btext).T).toarray(),
        (gvec @ emb.encode(_bill_short(bills).tolist()).T).toarray(),
    )
    fr = bills["first_reading"].to_numpy(dtype="datetime64[ns]")
    names = bill_name_patterns(bills)
    mid = (emb.link_auto + emb.link_check) / 2
    rows = []
    for i, g in enumerate(groups.itertuples(index=False)):
        t = pd.Timestamp(g.trigger_date)
        lo = (t - pd.Timedelta(days=before_days)).to_datetime64()
        hi = (t + pd.DateOffset(years=max_years)).to_datetime64()
        ok = (fr >= lo) & (fr <= hi)
        if not ok.any():
            rows.append((g.group_id, None, np.nan, None, None))
            continue
        s = np.where(ok, sim[i], -1.0)
        named = _named_bill(getattr(g, "names", ""), names, ok)
        j = named if named is not None else int(np.argmax(s))
        best, cand = float(s[j]), int(bills["bill_id"].iloc[j])
        decision, how = None, None
        if named is not None:  # the group's own words name this bill
            decision, how = cand, "auto"
        elif best >= emb.link_auto:
            decision, how = cand, "auto"
        elif best >= emb.link_check:
            if judge is not None:
                decision, how = (cand if judge(g.text, btext[j]) else None), "gemini"
            else:
                decision, how = (cand if best >= mid else None), "rule"
        rows.append((g.group_id, cand, round(best, 4), decision, how))
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- merge ----
def build_ideas(
    grouped: pd.DataFrame,
    groups: pd.DataFrame,
    links: pd.DataFrame,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge groups linked to the same bill into one idea (earliest trigger kept).

    Returns (ideas, idea_mentions, idea_bill_links) with exactly the contract's columns. Each
    idea_mentions row keeps the mention's pre-merge ``group_id`` (DATA_CONTRACT §10.4)."""
    now = now or datetime.now(UTC)
    if groups.empty:
        return (
            pd.DataFrame(columns=IDEA_COLUMNS),
            pd.DataFrame(columns=IDEA_MENTION_COLUMNS),
            pd.DataFrame(columns=LINK_COLUMNS),
        )
    g = groups.merge(links, on="group_id", how="left")
    g["key"] = [
        f"bill:{int(b)}" if pd.notna(b) else f"group:{gid}"
        for b, gid in zip(g["bill_id"], g["group_id"], strict=True)
    ]
    first = g.sort_values(["trigger_date", "group_id"]).groupby("key", sort=False).first()
    order = first.sort_values(["trigger_date", "group_id"]).index
    idea_of_key = {k: f"nlp-{n:05d}" for n, k in enumerate(order, start=1)}
    g["idea_id"] = g["key"].map(idea_of_key)

    m = _order(grouped.merge(g[["group_id", "idea_id"]], on="group_id"))
    idea_rows, link_rows = [], []
    for idea_id, im in m.groupby("idea_id", sort=True):
        gs = g[g["idea_id"] == idea_id].sort_values(["trigger_date", "group_id"])
        top = gs.sort_values(["n_mentions", "trigger_date"], ascending=[False, True]).iloc[0]
        day = im[im["_date"] == im["_date"].iloc[0]]
        idea_rows.append(
            {
                "idea_id": idea_id,
                "origin": "nlp",
                "topic": top["topic"],
                "trigger_date": im["_date"].iloc[0].date(),
                "department": _first_day_department(im),
                "first_source": im["source"].iloc[0],
                "initial_firmness": int(day["firmness"].max()),
                "n_mentions": len(im),
                "ingested_at": now,
            }
        )
        linked = gs[gs["bill_id"].notna()]
        if len(linked):
            how = "auto" if (linked["decided_by"] == "auto").any() else linked["decided_by"].iloc[0]
            bill_id, sim = int(linked["bill_id"].iloc[0]), float(linked["similarity"].max())
        else:
            row = gs.iloc[0]
            bill_id = None
            sim = None if pd.isna(row["similarity"]) else float(row["similarity"])
            how = row["decided_by"] if isinstance(row["decided_by"], str) else "auto"
        link_rows.append(
            {
                "idea_id": idea_id,
                "bill_id": bill_id,
                "similarity": sim,
                "decided_by": how,
                "checked": False,
                "ingested_at": now,
            }
        )
    out_ideas = pd.DataFrame(idea_rows, columns=IDEA_COLUMNS)
    out_im = m[["idea_id", "mention_id", "group_id"]].sort_values(["idea_id", "mention_id"])
    out_links = pd.DataFrame(link_rows, columns=LINK_COLUMNS)
    out_links["bill_id"] = out_links["bill_id"].astype("Int64")
    return out_ideas, out_im.reset_index(drop=True), out_links


# ---------------------------------------------------------------------------- run ----
@dataclass
class NlpResult:
    mentions: pd.DataFrame  # every trigger sentence (signal_mentions + helper columns)
    grouped: pd.DataFrame  # kept mentions with group_id
    groups: pd.DataFrame  # one row per topic group, with its link decision
    ideas: pd.DataFrame
    idea_mentions: pd.DataFrame
    idea_bill_links: pd.DataFrame
    bills: pd.DataFrame
    n_documents: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    def tables(self) -> dict[str, pd.DataFrame]:
        """The four derived tables, with exactly the contract's columns."""
        return {
            "signal_mentions": self.mentions[MENTION_COLUMNS],
            "ideas": self.ideas[IDEA_COLUMNS],
            "idea_mentions": self.idea_mentions[IDEA_MENTION_COLUMNS],
            "idea_bill_links": self.idea_bill_links[LINK_COLUMNS],
        }


def run_pipeline(
    docs: pd.DataFrame,
    posts: pd.DataFrame,
    bills: pd.DataFrame,
    stage_sittings: pd.DataFrame,
    long_titles: dict[int, str],
    fetched_members: set[int] | None = None,
    labeller: Labeller | None = None,
    judge: LinkJudge | None = None,
    embedder: Embedder | None = None,
    now: datetime | None = None,
) -> NlpResult:
    """label -> group -> link -> merge, entirely in memory (nothing is written)."""
    now = now or datetime.now(UTC)
    timings = {}
    t0 = time.perf_counter()
    bills_p = prepare_bills(bills, stage_sittings, long_titles)
    mentions = label_documents(docs, posts, fetched_members, bills_p, labeller, now)
    timings["label"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    grouped = group_mentions(mentions[mentions["kept"]], embedder)
    groups = summarize_groups(grouped)
    timings["group"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    links = link_groups(groups, bills_p, embedder, judge)
    out_ideas, out_im, out_links = build_ideas(grouped, groups, links, now)
    timings["link"] = time.perf_counter() - t0
    groups = groups.merge(links, on="group_id", how="left") if len(groups) else groups
    return NlpResult(
        mentions=mentions,
        grouped=grouped,
        groups=groups,
        ideas=out_ideas,
        idea_mentions=out_im,
        idea_bill_links=out_links,
        bills=bills_p,
        n_documents=len(docs),
        timings=timings,
    )


def summarize(result: NlpResult) -> dict[str, Any]:
    """Counts for the checkpoint report."""
    m, links = result.mentions, result.idea_bill_links
    kept = m[m["kept"]]
    linked = links[links["bill_id"].notna()]
    ideas_df = result.ideas.assign(year=pd.to_datetime(result.ideas["trigger_date"]).dt.year)
    by_year = (
        ideas_df.assign(linked=ideas_df["idea_id"].isin(linked["idea_id"]))
        .groupby("year")["linked"]
        .agg(ideas="size", linked="sum")
    )
    return {
        "documents": result.n_documents,
        "trigger_sentences": len(m),
        "trigger_by_source": m["source"].value_counts().to_dict(),
        "not_commitment": int((~m["is_commitment"].astype(bool)).sum()),
        "not_commitment_by_reason": m.loc[~m["is_commitment"].astype(bool), "drop_reason"]
        .value_counts()
        .to_dict(),
        "dropped_not_minister": int((m["drop_reason"] == "not_minister").sum()),
        "dropped_minister_unknown": int((m["drop_reason"] == "minister_unknown").sum()),
        "kept": len(kept),
        "kept_by_source_firmness": kept.groupby(["source", "firmness"]).size().to_dict(),
        "groups": int(result.grouped["group_id"].nunique()) if len(result.grouped) else 0,
        "ideas": len(result.ideas),
        "ideas_linked": len(linked),
        "linked_by": linked["decided_by"].value_counts().to_dict(),
        "unlinked_by": links.loc[links["bill_id"].isna(), "decided_by"].value_counts().to_dict(),
        "never_taken_up": int(links["bill_id"].isna().sum()),
        "merged_ideas": int(
            (result.idea_mentions.groupby("idea_id")["group_id"].nunique() > 1).sum()
        )
        if len(result.idea_mentions)
        else 0,
        "ideas_by_year": {int(y): (int(r.ideas), int(r.linked)) for y, r in by_year.iterrows()},
        "timings_s": {k: round(v, 1) for k, v in result.timings.items()},
    }


# --------------------------------------------------------------------------- write ----
def write_results(
    bind: Engine | Connection, result: NlpResult, clear_events: bool = False
) -> dict[str, int]:
    """Replace the derived NLP tables in one transaction (they are fully derived, so a rerun
    gives the same rows). ``source_documents`` and ``member_government_posts`` are never
    touched. ``idea_events`` refer to idea ids, which a rerun renumbers: if any exist the
    write is refused unless ``clear_events`` (they must then be re-matched)."""
    if isinstance(bind, Engine):
        with bind.begin() as conn:
            return write_results(conn, result, clear_events)
    n_events = bind.execute(select(func.count()).select_from(idea_events)).scalar_one()
    tables = list(DERIVED_TABLES)
    if n_events:
        if not clear_events:
            raise RuntimeError(
                f"idea_events has {n_events} rows keyed on idea ids that this run renumbers; "
                "rerun with --clear-events and re-match the events afterwards"
            )
        tables.insert(0, idea_events)
    deleted = takeup_db.clear_derived_tables(bind, tables)
    log.info("cleared %s", deleted)
    out = {}
    for name, df in result.tables().items():
        out[name] = takeup_db.upsert_rows(bind, name, df)
    return out


# -------------------------------------------------------------------------- review ----
class ReviewAlreadyLabelled(RuntimeError):
    """A review CSV already has hand labels in its 'correct' column."""


def has_labels(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return True  # unreadable: do not overwrite what may be someone's work
    return "correct" in df and df["correct"].str.strip().ne("").any()


def write_review_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a review CSV unless the existing file already has any 'correct' value."""
    if has_labels(path):
        raise ReviewAlreadyLabelled(
            f"{path} already has hand labels in 'correct'; not overwritten "
            "(move it away first to export a new sample)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[:0]
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[np.sort(idx)]


def _kept_mask(sm: pd.DataFrame) -> pd.Series:
    def flag(s: pd.Series) -> pd.Series:
        return s.astype("boolean").fillna(False).astype(bool)

    return flag(sm["is_commitment"]) & flag(sm["is_minister"])


def review_mentions(
    sm: pd.DataFrame, n: int = 100, n_dropped: int = 15, seed: int = 0
) -> pd.DataFrame:
    """``n`` random trigger sentences (``n_dropped`` of them dropped ones), fixed seed."""
    rng = np.random.default_rng(seed)
    sm = sm.sort_values("mention_id").reset_index(drop=True)
    kept = _kept_mask(sm)
    dropped = _sample(sm[~kept], n_dropped, rng)
    picked = pd.concat([_sample(sm[kept], n - len(dropped), rng), dropped])
    picked = picked.sort_values(["event_date", "mention_id"])
    return pd.DataFrame(
        {
            "id": picked["mention_id"],
            "date": picked["event_date"],
            "source": picked["source"],
            "title": picked["title"],
            "sentence": picked["sentence"],
            "firmness": picked["firmness"],
            "topic": picked["topic"],
            "kept": _kept_mask(picked),
            "correct": "",
        }
    )


def recompute_groups(
    sm: pd.DataFrame, im: pd.DataFrame, bills: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Topic groups and their link candidates rebuilt from the stored tables (the same
    functions as the run, so the candidates match what was decided)."""
    existing = existing_bill_titles(bills)
    kept = sm[_kept_mask(sm)].merge(im[["mention_id", "group_id"]], on="mention_id")
    kept = kept.assign(
        embed_title=[
            _title_for_embedding(t, pd.Timestamp(d), existing)
            for t, d in zip(kept["title"], kept["event_date"], strict=True)
        ],
        topic_specific=kept["topic"].map(is_specific),
    )
    groups = summarize_groups(kept)
    return groups, link_groups(groups, bills)


def review_links(
    tables: dict[str, pd.DataFrame], bills: pd.DataFrame, n: int = 50, seed: int = 0
) -> pd.DataFrame:
    """``n`` idea -> bill decisions, stratified across auto links, rule links and rule
    rejections (borderline ideas left unlinked, shown with their best candidate bill)."""
    rng = np.random.default_rng(seed)
    sm, im = tables["signal_mentions"], tables["idea_mentions"]
    links = tables["idea_bill_links"].merge(tables["ideas"], on="idea_id")
    groups, glinks = recompute_groups(sm, im, bills)
    cand = (
        im.drop_duplicates("idea_id")[["idea_id", "group_id"]]
        .merge(glinks[["group_id", "candidate_bill_id"]], on="group_id", how="left")
        .set_index("idea_id")["candidate_bill_id"]
    )
    links = links.assign(
        shown_bill=[
            b if pd.notna(b) else cand.get(i)
            for i, b in zip(links["idea_id"], links["bill_id"], strict=True)
        ]
    )
    links = links.sort_values("idea_id").reset_index(drop=True)
    strata = [
        links[links["bill_id"].notna() & (links["decided_by"] == "auto")],
        links[links["bill_id"].notna() & (links["decided_by"] != "auto")],
        links[links["bill_id"].isna() & links["decided_by"].isin(["rule", "gemini"])],
    ]
    picked, left = [], n
    for k, s in enumerate(strata):
        take = left // (len(strata) - k)
        part = _sample(s, take, rng)
        picked.append(part)
        left -= len(part)
    p = pd.concat(picked)
    # example: the representative sentence of the idea's earliest topic group
    rep = groups.set_index("group_id")["example"]
    first_group = groups.merge(im[["group_id", "idea_id"]].drop_duplicates(), on="group_id")
    first_group = first_group.sort_values(["trigger_date", "group_id"]).drop_duplicates("idea_id")
    example_of = dict(zip(first_group["idea_id"], first_group["group_id"].map(rep), strict=True))
    b = bills.set_index("bill_id")
    return pd.DataFrame(
        {
            "idea_id": p["idea_id"],
            "decided_by": p["decided_by"],
            "trigger_date": p["trigger_date"],
            "topic": p["topic"],
            "example_sentence": p["idea_id"].map(example_of),
            "bill_id": p["shown_bill"].astype("Int64"),
            "bill_title": [b["title"].get(x) if pd.notna(x) else None for x in p["shown_bill"]],
            "bill_long_title": [
                b["long_title"].get(x) if pd.notna(x) else None for x in p["shown_bill"]
            ],
            "bill_first_reading": [
                b["first_reading"].get(x).date() if pd.notna(x) else None for x in p["shown_bill"]
            ],
            "similarity": p["similarity"],
            "linked": p["bill_id"].notna(),
            "correct": "",
            "correct_bill_id": "",
        }
    )


def export_review(
    tables: dict[str, pd.DataFrame],
    bills: pd.DataFrame,
    out_dir: Path = REVIEW_DIR,
    seed: int = 0,
) -> dict[str, str]:
    """Write review_mentions.csv and review_links.csv; a file that already has hand labels is
    kept as it is. Returns path -> 'written' or the reason it was skipped."""
    out = {}
    frames = {
        "review_mentions.csv": lambda: review_mentions(tables["signal_mentions"], seed=seed),
        "review_links.csv": lambda: review_links(tables, bills, seed=seed),
    }
    for name, build in frames.items():
        path = Path(out_dir) / name
        if has_labels(path):
            out[str(path)] = "skipped: already has hand labels"
            continue
        write_review_csv(build(), path)
        out[str(path)] = "written"
    return out


def _yes(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["y", "yes", "1", "true"])


def score(out_dir: Path = REVIEW_DIR) -> dict[str, dict[str, Any]]:
    """Precision of kept mentions / links from the hand labels (prototype ``cmd_score``):
    rows whose 'correct' is y or n count; for rejected rows, 'n' means wrongly rejected."""
    out = {}
    for name, col in (("review_mentions", "kept"), ("review_links", "linked")):
        path = Path(out_dir) / f"{name}.csv"
        if not path.exists():
            continue
        r = pd.read_csv(path, dtype=str, keep_default_na=False)
        r = r[r["correct"].str.strip().str.lower().isin(["y", "n", "yes", "no"])]
        if r.empty:
            out[name] = {"labelled": 0}
            continue
        flag = r[col].str.strip().str.lower().isin(["true", "1", "y", "yes"])
        pos, neg = r[flag], r[~flag]
        tp = int(_yes(pos["correct"]).sum())
        res: dict[str, Any] = {
            "labelled": len(r),
            "positives_checked": len(pos),
            "precision": tp / len(pos) if len(pos) else None,
            "rejected_checked": len(neg),
            "wrongly_rejected": int((~_yes(neg["correct"])).sum()),
        }
        if name == "review_links" and "correct_bill_id" in r:
            res["corrected_bill_ids"] = int(r["correct_bill_id"].str.strip().ne("").sum())
        out[name] = res
    return out
