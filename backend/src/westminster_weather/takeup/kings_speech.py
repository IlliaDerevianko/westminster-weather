"""Tier A: bills announced in King's/Queen's Speech briefing notes, linked to Government Bills.

Pipeline (no database writes; reference tables are read in a READ ONLY transaction, and
loading into ``kings_speech_items`` / ``idea_bill_links`` is a later step):

1. ``discover_speeches``: find each speech's briefing-note publication on GOV.UK (search API,
   then content API for the PDF attachment), download the PDF to the cache and settle the
   speech date (the State Opening, not the GOV.UK publication timestamp).
2. ``parse_items``: read the PDF's contents page and keep every entry naming a Bill (or
   "... legislation"), plus bracketed bill tags in the speech text that no entry covers, with
   draft / carried-over / hybrid-bill flags and the lead department when the notes have a
   contact table.
3. ``link_items``: match each item to a Government Bill first read between 60 days before and
   730 days after the speech (4 years for drafts; never after a change of governing party),
   or carried over into the speech's session, by normalised title (exact match first, then
   TF-IDF character n-gram cosine over short title and long title); one bill per speech.
4. ``apply_carry_over``: an item whose bill was carried over into the session is flagged
   ``is_carried_over`` whatever the notes say.
5. ``export_review_csv``: one row per item for the hand-check (DATA_CONTRACT §10.6).
"""

import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from westminster_weather.takeup.bill_text import (
    bill_long_titles,
    first_readings,
    get_json,
)
from westminster_weather.takeup.tier_b import CACHE_DIR, USER_AGENT

log = logging.getLogger(__name__)

SPEECHES_DIR = CACHE_DIR / "kings_speech"
REVIEW_DIR = Path(__file__).resolve().parents[3] / "data" / "takeup" / "review"
REVIEW_CSV = REVIEW_DIR / "kings_speech_links.csv"

SEARCH_URL = "https://www.gov.uk/api/search.json"
CONTENT_URL = "https://www.gov.uk/api/content{link}"
SEARCH_QUERIES = (
    "Queen's Speech background briefing notes",
    "King's Speech background briefing notes",
)
TITLE_RE = re.compile(r"(?:queen|king)['’]s speech.*background briefing notes", re.I)
FIRST_YEAR = 2012
# One speech per year except 2019 (October and December). There was no speech in 2011,
# 2018, 2020 or 2025.
EXPECTED_SPEECH_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2019, 2019, 2021, 2022, 2023, 2024]
EXPECTED_SPEECH_YEARS += [2026]

# Linking window and thresholds (similarity is a cosine in [0, 1]).
WINDOW_BEFORE_DAYS = 60
WINDOW_AFTER_DAYS = 730
# Draft bills often take more than two years to be introduced (e.g. Draft Patient Safety Bill
# 2017 -> Health Service Safety Investigations Bill 2019, 846 days).
WINDOW_AFTER_DRAFT_DAYS = 1461
CARRIED_OVER_BEFORE_DAYS = 730  # carried-over bills were first read in an earlier session
AUTO_MIN_SIMILARITY = 0.75
AUTO_MIN_MARGIN = 0.15
SUGGEST_MIN_SIMILARITY = 0.40
# Dissolutions after which the governing party changed (2010 and 2024 elections). A bill first
# read after one cannot take up an earlier government's idea: such ideas are censored at the
# dissolution (DATA_CONTRACT §10.9), so later bills are never candidates.
PARTY_CHANGE_ELECTION_YEARS = (2010, 2024)
PARTY_CHANGE_DISSOLUTIONS = (date(2010, 4, 12), date(2024, 5, 30))
# Hybrid bills (High Speed Rail / HS2, Holocaust Memorial) are not in ``bills`` (Government
# Bills only), so they cannot be linked; flagged for exclusion from the take-up denominator.
# "High Speed Rail (Preparation) Bill" (2013) is an ordinary public bill.
HYBRID_RE = re.compile(
    r"\bHS2\b|\bHigh\s+Speed\s+(?:Rail|Two)\b(?!\s*\(Preparation\))"
    r"|\bNorthern\s+Powerhouse\s+Rail\b|\bHolocaust\s+Memorial\b",
    re.I,
)
# The briefing notes name the formal title of a renamed bill: "(formally titled the Courts and
# Tribunals Bill)".
FORMAL_TITLE_RE = re.compile(
    r"formally\s+(?:titled|named|known\s+as)\s+(?:the\s+)?([A-Z][^.;:]{2,120}?\bBill)\b"
)
# An item announced only as a draft: "The Bill will be published in draft" in its own brief
# (or "We will also publish a draft Building Safety Bill" anywhere, see ``_draft_in_text``).
DRAFT_TEXT_RE = re.compile(r"\b(?:will|to)\s+be\s+published\s+in\s+draft\b", re.I)
# Bracketed bill tags in the speech text: "[Animal Welfare (Sentencing) Bill]".
SPEECH_TAG_RE = re.compile(r"\[([^\[\]]{3,250})\]")
TAG_PART_RE = re.compile(
    r"\s*(.*?\b(?:Bill\s+of\s+Rights|Bill(?:\s+\d{4})?|[Ll]egislation|Law))"
    r"(?=\s*,\s*|\s+and\s+|\s*$)"
)
TAG_MATCH_MIN = 0.5  # a tag is the same bill as a contents item at this word overlap
# The session start in ``sessions`` is its first bill sitting: the State Opening or the day after.
SESSION_START_MAX_LAG_DAYS = 3

REVIEW_COLUMNS = [
    "item_id",
    "speech_date",
    "announced_name",
    "is_draft",
    "is_carried_over",
    "carried_over_source",
    "exclude_reason",
    "item_source",
    "alt_names",
    "department",
    "bill_id",
    "bill_title",
    "bill_first_reading",
    "similarity",
    "second_bill_id",
    "second_bill_title",
    "second_similarity",
    "decided_by",
    "review_note",
    "correct",
    "correct_bill_id",
]

MONTHS = (
    "january february march april may june july august september october november december"
).split()
DATE_RE = re.compile(
    r"\b(?:(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+)?"
    r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b",
    re.I,
)
# A contents page number: "12", "Pages 3-6", "Page 29". Digits glued to words, slashes or
# brackets ("COP26", "2015/16", "(No. 2)") are not page numbers.
PAGE_NUM_RE = re.compile(
    r"(?:\bPages?\s*)?(?<![\w/.,£'’(\-–])(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?(?![\w/%)’'])", re.I
)
DOT_LEADER_RE = re.compile(r"(?:\s*\.){3,}")
BILL_RE = re.compile(r"\bbill\b", re.I)
LEGISLATION_RE = re.compile(r"\blegislation\b", re.I)
DRAFT_RE = re.compile(r"^\s*draft\b|\(draft\)", re.I)
HEADING_NOISE_RE = re.compile(
    r"the\s+(?:queen|king)['’]s\s+speech(?:\s+\d{4})?(?:\s*[–-]\s*(?:briefing notes|contents))?"
    r"|\bcontents\b|\bbackground briefs\b|\bother measures\b",
    re.I,
)
CARRY_RE = re.compile(
    r"carried\s+over|carry[\s-]+over|continuing\s+from|carried[\s-]+forward"
    r"|introduced\s+in\s+the\s+(?:last|previous)\s+session",
    re.I,
)
CARRY_NEG_RE = re.compile(
    r"\b(?:not|never|insufficient time (?:for it )?to|unable to|cannot|could not)\s+"
    r"(?:be\s+|been\s+)?(?:carried|carry)",
    re.I,
)

DEPARTMENTS = [
    "Cabinet Office",
    "HM Treasury",
    "Home Office",
    "Ministry of Justice",
    "Ministry of Defence",
    "Foreign, Commonwealth and Development Office",
    "Foreign and Commonwealth Office",
    "Department for Business, Innovation and Skills",
    "Department for Business, Energy and Industrial Strategy",
    "Department for Business and Trade",
    "Department for Energy Security and Net Zero",
    "Department of Energy and Climate Change",
    "Department for Science, Innovation and Technology",
    "Department for Digital, Culture, Media and Sport",
    "Department for Culture, Media and Sport",
    "Department for Education",
    "Department for Environment, Food and Rural Affairs",
    "Department of Health and Social Care",
    "Department of Health",
    "Department for Transport",
    "Department for Work and Pensions",
    "Department for Communities and Local Government",
    "Ministry of Housing, Communities and Local Government",
    "Department for Levelling Up, Housing and Communities",
    "Department for International Trade",
    "Department for International Development",
    "Department for Exiting the European Union",
    "Northern Ireland Office",
    "Scotland Office",
    "Wales Office",
    "Office of the Secretary of State for Scotland",
    "Office of the Secretary of State for Wales",
    "Attorney General's Office",
    "Government Equalities Office",
    "Office of the Leader of the House of Commons",
    "HM Revenue and Customs",
    "Office for Equality and Opportunity",
]


# --------------------------------------------------------------------- text helpers ----
def _clean_quotes(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def _squash(text: str) -> str:
    """Collapse all whitespace (including one-word-per-line extraction) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def normalise_title(title: str) -> str:
    """Comparable form of a bill title: lowercase, no 'draft' / 'bill' / '[HL]' / '(No. n)' /
    trailing 'Act <year>' (``bills`` holds enacted titles), no punctuation or curly quotes."""
    t = _clean_quotes(str(title)).lower()
    t = re.sub(r"\[\s*hl\s*\]|\(\s*hl\s*\)", " ", t)
    t = re.sub(r"\(\s*no\.?\s*\d+\s*\)", " ", t)
    t = t.replace("&", " and ")
    t = re.sub(r"'s?\b", "", t)
    t = re.sub(r"\b(?:draft|bill)\b", " ", t)
    t = re.sub(r"\s+ch\.?\s*\d+\s*$", " ", t.strip())  # "Bus Services Act 2017 ch 21"
    t = re.sub(r"\bact(?:\s+\d{4})?\s*$", " ", t.strip())  # enacted titles: "... Act 2013"
    t = re.sub(r"^the\s+", " ", t.strip())
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\beu\b", "european union", t)
    return _squash(t)


def _name_pattern(name: str) -> re.Pattern[str]:
    """Regex matching ``name`` across any whitespace, case- and quote-insensitively."""
    words = _clean_quotes(name).split()
    parts = [
        r"(?:&|and)" if w.lower() in ("&", "and") else re.escape(w).replace("'", "['’]")
        for w in words
    ]
    return re.compile(r"(?<![\w])" + r"\s+".join(parts) + r"(?![\w])", re.I)


def _smart_title(name: str) -> str:
    """Title case for ALL-CAPS contents entries when the body has no mixed-case copy."""
    small = {"and", "of", "the", "for", "to", "in", "on", "a", "an", "at", "by"}
    keep_upper = {"NHS", "EU", "UK", "HL", "HS2", "UIFSM", "ISAS", "COP26", "BBC", "GB"}
    words = []
    for i, w in enumerate(name.split()):
        core = re.sub(r"[^\w]", "", w).upper()
        if core in keep_upper:
            words.append(w.upper())
        elif i > 0 and w.lower() in small:
            words.append(w.lower())
        else:
            k = next((j for j, c in enumerate(w) if c.isalpha()), 0)
            words.append(w[:k] + w[k : k + 1].upper() + w[k + 1 :].lower())
    return " ".join(words)


def _cover_date(pages: Sequence[str], year_hint: int | None = None) -> date | None:
    """The date printed on the briefing notes' cover / contents pages."""
    text = _squash(" ".join(pages[:4]))
    for m in DATE_RE.finditer(text):
        d = date(int(m.group(3)), MONTHS.index(m.group(2).lower()) + 1, int(m.group(1)))
        if year_hint is None or abs(d.year - year_hint) <= 1:
            return d
    return None


# ------------------------------------------------------------------- contents parse ----
def _read_pages(source: str | Path) -> list[str]:
    """PDF pages as text; a str that is not a PDF path is taken as text, pages split by \\f."""
    path = Path(source) if isinstance(source, Path) else None
    if path is None and isinstance(source, str) and source.lower().endswith(".pdf"):
        path = Path(source)
    if path is not None:
        from pypdf import PdfReader

        logging.getLogger("pypdf").setLevel(logging.ERROR)
        return [page.extract_text() or "" for page in PdfReader(path).pages]
    return str(source).split("\f")


def _bill_entry_count(text: str) -> int:
    t = DOT_LEADER_RE.sub(" ", _squash(text))
    return len(
        re.findall(
            r"\b(?:bill|legislation)(?:\s*\[HL\])?(?:\s*\(draft\))?\s+(?:pages?\s*)?\d", t, re.I
        )
    )


def _contents_pages(pages: Sequence[str]) -> list[int]:
    """Indexes of the contents pages: the first page (of the first 8) that looks like a
    contents list, plus following pages that still list bills with page numbers."""
    start = None
    for i, text in enumerate(pages[:8]):
        if _bill_entry_count(text) >= 3 or (
            re.search(r"\bcontents\b", text, re.I) and _bill_entry_count(text) >= 1
        ):
            start = i
            break
    if start is None:
        return []
    found = [start]
    for i in range(start + 1, min(start + 4, len(pages))):
        if _bill_entry_count(pages[i]) == 0:
            break
        found.append(i)
    return found


def _contents_entries(text: str, max_page: int) -> list[dict[str, Any]]:
    """Split a contents text into (name, first page, last page) entries. Page numbers must
    not decrease, which rejects stray numbers such as page footers and 'Rail 2'."""
    t = DATE_RE.sub(" ", _squash(text))
    t = DOT_LEADER_RE.sub(" ", t)
    entries, last, cursor = [], 0, 0
    for m in PAGE_NUM_RE.finditer(t):
        first = int(m.group(1))
        if first < last or first > max_page:
            continue
        name = t[cursor : m.start()]
        cursor, last = m.end(), first
        end = int(m.group(2)) if m.group(2) else first
        entries.append({"raw": name, "page": first, "page_end": max(end, first)})
    return entries


def _clean_entry(raw: str) -> str:
    name = HEADING_NOISE_RE.sub(" ", raw)
    name = _squash(name)
    name = re.sub(r"^(?:[\d\s.:;,)–-]+)", "", name)  # stray footers / list numbers
    name = re.sub(r"[\s\d.:;,–-]+$", "", name) if not name.endswith(")") else name
    return _clean_quotes(name.strip(" -–|"))


def _is_heading(name: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", name)
    return bool(letters) and letters.isupper() and not BILL_RE.search(name)


def _is_item(name: str) -> bool:
    return bool(BILL_RE.search(name) or LEGISLATION_RE.search(name)) and len(name) <= 160


def _fallback_items(pages: Sequence[str]) -> list[dict[str, Any]]:
    """Bill-heading lines (short lines ending in 'Bill'), for PDFs without a usable
    contents page."""
    seen, items = set(), []
    for i, text in enumerate(pages):
        for line in text.splitlines():
            line = _clean_quotes(_squash(line))
            if (
                re.search(r"\bBill(?:\s*\[HL\])?$", line)
                and 2 <= len(line.split()) <= 14
                and line[:1].isupper()
            ):
                key = normalise_title(line)
                if key and key not in seen:
                    seen.add(key)
                    items.append({"name": line, "page": i + 1, "page_end": i + 1, "carry": False})
    return items


def _section_offset(items: list[dict[str, Any]], pages: Sequence[str]) -> int:
    """PDF page index = printed page number - 1 + offset; the offset most items agree on."""
    votes: Counter[int] = Counter()
    for item in items:
        pat = _name_pattern(item["name"])
        for off in range(-4, 8):
            i = item["page"] - 1 + off
            if 0 <= i < len(pages) and pat.search(pages[i]):
                votes[off] += 1
    return votes.most_common(1)[0][0] if votes else 0


def _draft_in_text(name: str, own: str, full_text: str) -> bool:
    """Announced only as a draft: its own brief says the Bill will be published in draft, or
    anywhere in the notes '(will) publish a draft <name>'."""
    if DRAFT_TEXT_RE.search(own):
        return True
    core = re.sub(r"^\s*draft\s+|\s*\(draft\)\s*$", "", name, flags=re.I)
    pat = re.compile(r"\bpublish\s+(?:a\s+|the\s+)?draft\s+" + _name_pattern(core).pattern, re.I)
    return bool(pat.search(_clean_quotes(full_text)))


def _carry_in_text(text: str) -> bool:
    for sentence in re.split(r"(?<=[.!?\]])\s+", _squash(text)):
        if CARRY_RE.search(sentence) and not CARRY_NEG_RE.search(sentence):
            return True
    return False


def _carry_after_name(name: str, full_text: str) -> bool:
    """Notes like '[Energy Bill – carried over from previous session]' in the speech text."""
    pat = _name_pattern(name)
    for m in pat.finditer(full_text):
        tail = re.split(r"(?<=[.\]])\s", full_text[m.end() : m.end() + 250])[0]
        if CARRY_RE.search(tail) and not CARRY_NEG_RE.search(tail) and not BILL_RE.search(tail):
            return True
    return False


def _norm_dept_text(text: str) -> str:
    t = _clean_quotes(text).lower().replace("&", " and ")
    t = re.sub(r"[,]", " ", t)
    return _squash(t)


# Spellings used in the contact tables that differ from the canonical names.
DEPARTMENT_ALIASES = {
    "His Majesty's Treasury": "HM Treasury",
    "Her Majesty's Treasury": "HM Treasury",
    "Department for Health and Social Care": "Department of Health and Social Care",
    "Department for Health": "Department of Health",
    "Department for Energy and Climate Change": "Department of Energy and Climate Change",
    # 2026: the name is cut by a page break ("... Communities and Local 0303 444 1209")
    "Ministry of Housing, Communities and Local": (
        "Ministry of Housing, Communities and Local Government"
    ),
}
_DEPT_PATTERNS = [(d, _norm_dept_text(d)) for d in DEPARTMENTS] + [
    (canonical, _norm_dept_text(alias)) for alias, canonical in DEPARTMENT_ALIASES.items()
]
_DEPT_RE = re.compile(
    "|".join(re.escape(p) for _, p in sorted(_DEPT_PATTERNS, key=lambda dp: -len(dp[1])))
)
_DEPT_OF = {p: d for d, p in _DEPT_PATTERNS}
ROW_MATCH_MIN = 0.6  # word overlap for a contact-table row whose name is not verbatim


def _words(title: str) -> set[str]:
    """Content words of a normalised title (no years or other numbers)."""
    return {w for w in _content_words(normalise_title(title)).split() if not w.isdigit()}


def _stems(title: str) -> set[str]:
    """Content words of a title cut to 4 letters ('railways' ~ 'rail'), for overlap tests."""
    return {w[:4] for w in _words(title)}


def _overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two titles' content-word stems."""
    sa, sb = _stems(a), _stems(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def _contact_rows(region: str) -> list[tuple[str, str]]:
    """(row name, department) pairs of a contact table: each department mention, with the
    text since the previous row's phone number / department as the name. Rows whose name is
    empty (a second department, 'Press Office') are dropped."""
    text = _norm_dept_text(region)
    rows, prev = [], 0
    for m in _DEPT_RE.finditer(text):
        name = re.split(r"\d{3,}|\S+@\S+|press office|contact details", text[prev : m.start()])[-1]
        name = name.strip(" -–|")
        prev = m.end()
        if re.search(r"[a-z]{3,}", name):
            rows.append((name, _DEPT_OF[m.group(0)]))
    return rows


def _department_after(name: str, region: str) -> str | None:
    """Lead department from the contact table: the row naming the bill verbatim, else the
    row with the closest title (word order and '(draft)' ignored)."""
    core = re.sub(r"^\s*draft\s+|\s*\(draft\)\s*$", "", name, flags=re.I)
    m = _name_pattern(core).search(region)
    if m:
        # the department comes before the row's phone number; stop there so the next row's
        # department is never picked up
        window = re.split(r"\d{3,}", region[m.end() : m.end() + 160])[0]
        window = _norm_dept_text(window)
        hits = [(window.find(p), -len(p), d) for d, p in _DEPT_PATTERNS if p in window]
        if hits:
            return min(hits)[2]
    scored = [(_overlap(core, row), dept) for row, dept in _contact_rows(region)]
    best = max(scored, default=(0.0, None), key=lambda sd: sd[0])
    return best[1] if best[0] >= ROW_MATCH_MIN else None


ITEM_COLUMNS = [
    "order",
    "announced_name",
    "is_draft",
    "is_carried_over",
    "department",
    "from_contents",
    "item_source",
    "alt_names",
    "exclude_reason",
]


def _own_text(name: str, next_name: str | None, section: str) -> str:
    """The part of a section that belongs to one item: from its name to the next item's."""
    text = _squash(section)
    m = _name_pattern(name).search(text)
    start = m.start() if m else 0
    if next_name:
        n = _name_pattern(next_name).search(text, (m.end() if m else 0))
        if n:
            return text[start : n.start()]
    return text[start:]


def _alt_names(name: str, own: str) -> list[str]:
    """Other titles for the item in its own brief: a stated formal title, or a body heading
    that adds a bracketed part ('Northern Ireland (Stormont House Agreement) Bill')."""
    alts = [_squash(_clean_quotes(m.group(1))) for m in FORMAL_TITLE_RE.finditer(own)]
    core = re.sub(r"\s*\bBill\b\s*$", "", name)
    if core != name and core:
        pat = re.compile(_name_pattern(core).pattern + r"\s+(\([^()]{3,80}\))\s+Bill\b", re.I)
        alts += [f"{core} {m.group(1)} Bill" for m in pat.finditer(_clean_quotes(own))]
    seen = {normalise_title(name)}
    out = []
    for a in alts:
        if normalise_title(a) not in seen:
            seen.add(normalise_title(a))
            out.append(a)
    return out


def speech_tags(full_text: str) -> list[str]:
    """Bill names tagged in brackets in the speech text, multi-bill tags split:
    '[Domestic Abuse Bill, Divorce, Dissolution and Separation Bill]' gives two names."""
    names: list[str] = []
    for tag in SPEECH_TAG_RE.findall(_clean_quotes(full_text)):
        tag = re.split(r"\s+[–-]\s+", _squash(tag))[0]  # "Energy Bill – carried over ..."
        for m in TAG_PART_RE.finditer(tag):
            name = re.sub(r"^[,\s]*(?:and\s+)?", "", m.group(1)).strip(" ,")
            if re.fullmatch(r"(?:other|further)\s+legislation", name, re.I) or len(name) > 160:
                continue
            name = _smart_title(name) if re.sub(r"[^A-Za-z]", "", name).isupper() else name
            if name not in names:
                names.append(name)
    return names


def _untagged_items(tags: list[str], names: list[str]) -> list[str]:
    """Speech tags that are no contents item: each item absorbs at most its one best tag
    (word overlap >= TAG_MATCH_MIN), so a combined item ('Animal Welfare Plan and
    Legislation') covering several tagged bills keeps one and the others become items. A tag
    whose words are all in an item's name is covered by it.
    Hybrid-bill tags are covered by any hybrid-bill item of the speech."""
    has_hybrid = any(HYBRID_RE.search(n) for n in names)
    words = [_words(n) for n in names]
    tag_words = [_words(t) for t in tags]
    # a tag whose words all appear in an item's name is covered ('NHS Funding Bill' in 'NHS
    # Funding Bill and NHS Long Term Plan'), however many tags that item covers; such an item
    # takes no other tag by overlap
    contains = [
        (ti, ni) for ti, tw in enumerate(tag_words) for ni, x in enumerate(words) if tw and tw <= x
    ]
    pairs = sorted(
        ((_overlap(t, n), ti, ni) for ti, t in enumerate(tags) for ni, n in enumerate(names)),
        reverse=True,
    )
    used_tags = {ti for ti, _ in contains}
    used_items = {ni for _, ni in contains}
    for score, ti, ni in pairs:
        if score < TAG_MATCH_MIN:
            break
        if ti in used_tags or ni in used_items:
            continue
        used_tags.add(ti)
        used_items.add(ni)
    return [
        t
        for ti, t in enumerate(tags)
        if ti not in used_tags and not (has_hybrid and HYBRID_RE.search(t))
    ]


def parse_items(source: str | Path) -> pd.DataFrame:
    """Bills announced in one set of briefing notes (a PDF path, or text with pages split by
    form feeds), one row each (columns ``ITEM_COLUMNS``).

    Items come from the contents page (``item_source = 'contents'``; bill headings when there
    is no usable contents page), then from bracketed bill tags in the speech text that match no
    contents item (``'speech_tag'``, appended so contents numbering is unchanged).
    ``alt_names`` lists other titles for the bill found in its brief ('|'-separated);
    ``exclude_reason = 'hybrid_bill'`` marks hybrid bills, which ``bills`` cannot hold."""
    pages = _read_pages(source)
    contents_idx = _contents_pages(pages)
    items: list[dict[str, Any]] = []
    if contents_idx:
        text = "\n".join(pages[i] for i in contents_idx)
        carry_heading = False
        for entry in _contents_entries(text, max_page=len(pages) + 20):
            name = _clean_entry(entry["raw"])
            if not name:  # only a heading such as "Other measures": a new section
                carry_heading = False
                continue
            if _is_heading(name):
                carry_heading = bool(CARRY_RE.search(name))
                continue
            if _is_item(name):
                items.append({**entry, "name": name, "carry": carry_heading})
    from_contents = len(items) >= 3
    if not from_contents:
        items = _fallback_items(pages)

    body_start = (contents_idx[-1] + 1) if contents_idx else 0
    body_text = _squash(" ".join(pages[body_start:]))
    full_text = _squash(" ".join(pages))
    offset = _section_offset(items, pages) if from_contents else 0
    # contact table at the end: "Bill | Lead Department | Press Office contact"
    lead = [
        m.start()
        for m in re.finditer(r"lead\s+department", full_text, re.I)
        if m.start() > 0.6 * len(full_text)
    ]
    contact_region = full_text[lead[0] :] if lead else ""
    starts = sorted({it["page"] for it in items})

    rows = []
    for order, item in enumerate(items, start=1):
        name = item["name"]
        letters = re.sub(r"[^A-Za-z]", "", name)
        if letters.isupper():
            mixed = [
                m.group(0)
                for m in _name_pattern(name).finditer(body_text)
                if not re.sub(r"[^A-Za-z]", "", m.group(0)).isupper()
            ]
            name = _squash(_clean_quotes(mixed[0])) if mixed else _smart_title(name)
        later = [p for p in starts if p > item["page"]]
        last_page = max(item["page_end"], (later[0] - 1) if later else item["page_end"])
        section = " ".join(
            pages[i]
            for i in range(item["page"] - 1 + offset, last_page + offset)
            if 0 <= i < len(pages)
        )
        next_name = items[order]["name"] if order < len(items) else None
        own = _own_text(item["name"], next_name, section) if from_contents else ""
        carried = (
            item["carry"]
            or (from_contents and _carry_in_text(section))
            or _carry_after_name(item["name"], full_text)
        )
        rows.append(
            {
                "order": order,
                "announced_name": name,
                "is_draft": bool(DRAFT_RE.search(name))
                or _draft_in_text(item["name"], own, full_text),
                "is_carried_over": bool(carried),
                "department": _department_after(item["name"], contact_region)
                if contact_region
                else None,
                "from_contents": from_contents,
                "item_source": "contents" if from_contents else "headings",
                "alt_names": "|".join(_alt_names(name, own)) or None,
            }
        )
    for name in _untagged_items(speech_tags(full_text), [r["announced_name"] for r in rows]):
        rows.append(
            {
                "order": len(rows) + 1,
                "announced_name": name,
                "is_draft": bool(DRAFT_RE.search(name)),
                "is_carried_over": _carry_after_name(name, full_text),
                "department": _department_after(name, contact_region) if contact_region else None,
                "from_contents": False,
                "item_source": "speech_tag",
                "alt_names": None,
            }
        )
    for row in rows:
        names = [row["announced_name"], *(row["alt_names"] or "").split("|")]
        row["exclude_reason"] = (
            "hybrid_bill" if any(n and HYBRID_RE.search(n) for n in names) else None
        )
    return pd.DataFrame(rows, columns=ITEM_COLUMNS)


# -------------------------------------------------------------------- discovery ----
def _client() -> httpx.Client:
    return httpx.Client(timeout=120.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


def _search_publications(client: httpx.Client) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for q in SEARCH_QUERIES:
        url = str(
            httpx.URL(
                SEARCH_URL,
                params={
                    "q": q,
                    "count": 50,
                    "fields": ["title", "link", "public_timestamp", "content_store_document_type"],
                },
            )
        )
        for r in get_json(client, url).get("results", []):
            if TITLE_RE.search(r.get("title", "")):
                found[r["link"]] = r
    return found


def _pick_pdf(attachments: list[dict[str, Any]]) -> dict[str, Any] | None:
    pdfs = [a for a in attachments if a.get("content_type") == "application/pdf"]
    if not pdfs:
        return None
    return max(
        pdfs,
        key=lambda a: (
            "briefing" in (a.get("title") or "").lower() or "lobby" in a.get("url", "").lower(),
            a.get("number_of_pages") or 0,
        ),
    )


def _match_session(d: date, sessions: pd.DataFrame) -> pd.Series | None:
    s = sessions.copy()
    s["start"] = pd.to_datetime(s["start_date"]).dt.date
    s["gap"] = s["start"].map(lambda x: abs((x - d).days))
    best = s.sort_values("gap").iloc[0] if len(s) else None
    return best if best is not None and best["gap"] <= 30 else None


def discover_speeches(
    sessions: pd.DataFrame | None = None,
    refresh: bool = False,
    speeches_dir: Path = SPEECHES_DIR,
) -> pd.DataFrame:
    """One row per King's/Queen's Speech from 2012 with briefing notes on GOV.UK.

    Downloads each PDF to ``speeches_dir`` (cached) and sets ``speech_date`` to the State
    Opening: the date printed on the notes, checked against the GOV.UK first-published date and
    the start of the matching session (``sessions.start_date``, the session's first bill
    sitting). Disagreements are listed in the ``date_note`` column.
    """
    speeches_dir.mkdir(parents=True, exist_ok=True)
    meta_path = speeches_dir / "speeches.json"
    if meta_path.exists() and not refresh:
        records = json.loads(meta_path.read_text())
    else:
        records = []
        with _client() as client:
            for link, hit in sorted(_search_publications(client).items()):
                content = get_json(client, CONTENT_URL.format(link=link))
                published = (content.get("first_published_at") or hit["public_timestamp"])[:10]
                if int(published[:4]) < FIRST_YEAR:
                    continue
                att = _pick_pdf((content.get("details") or {}).get("attachments") or [])
                records.append(
                    {
                        "title": hit["title"],
                        "page_url": f"https://www.gov.uk{link}",
                        "document_type": hit.get("content_store_document_type"),
                        "search_timestamp": hit.get("public_timestamp", "")[:10],
                        "published_date": published,
                        "pdf_url": att["url"] if att else None,
                    }
                )
        meta_path.write_text(json.dumps(records, indent=1))

    with _client() as client:
        for rec in records:
            if not rec["pdf_url"]:
                continue
            path = speeches_dir / f"{rec['published_date']}.pdf"
            if not path.exists():
                response = client.get(rec["pdf_url"])
                response.raise_for_status()
                path.write_bytes(response.content)
            rec["pdf_path"] = str(path)

    if sessions is None:
        from westminster_weather.db import fetch_sessions_df

        sessions = fetch_sessions_df()

    rows = []
    for rec in records:
        published = date.fromisoformat(rec["published_date"])
        cover = (
            _cover_date(_read_pages(Path(rec["pdf_path"])), published.year)
            if rec.get("pdf_path")
            else None
        )
        notes = []
        speech = published
        if cover is not None and abs((cover - published).days) <= 3:
            speech = cover
        elif cover is not None:
            notes.append(f"cover date {cover} vs published {published}: used published")
        session = _match_session(speech, sessions)
        session_start = session["start"] if session is not None else None
        if session_start is None:
            notes.append("no matching session")
        else:
            lag = (session_start - speech).days
            if not 0 <= lag <= SESSION_START_MAX_LAG_DAYS:
                notes.append(f"session start {session_start} is {lag:+d} days from speech")
        search_ts = date.fromisoformat(rec["search_timestamp"]) if rec["search_timestamp"] else None
        if search_ts and abs((search_ts - speech).days) > 3:
            notes.append(
                f"search public_timestamp {search_ts} is {(search_ts - speech).days:+d} days "
                "from speech (later page update); used first-published / cover date"
            )
        rows.append(
            {
                "speech_date": speech,
                "title": _clean_quotes(rec["title"]),
                "page_url": rec["page_url"],
                "document_type": rec["document_type"],
                "pdf_url": rec["pdf_url"],
                "pdf_path": rec.get("pdf_path"),
                "published_date": published,
                "search_timestamp": search_ts,
                "cover_date": cover,
                "session_id": int(session["session_id"]) if session is not None else None,
                "session_start": session_start,
                "date_note": "; ".join(notes) or None,
            }
        )
    df = pd.DataFrame(rows).sort_values("speech_date").reset_index(drop=True)
    years = Counter(d.year for d in df["speech_date"])
    missing = Counter(EXPECTED_SPEECH_YEARS) - years
    if missing:
        log.warning("missing King's/Queen's Speeches: %s", sorted(missing.elements()))
    return df


def build_items(speeches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Parse every speech's briefing notes into one row per announced bill
    (DATA_CONTRACT §10.1 columns plus ``order``, ``from_contents``)."""
    speeches = discover_speeches() if speeches is None else speeches
    frames = []
    for sp in speeches.itertuples():
        if not sp.pdf_path:
            log.warning("no PDF for %s", sp.title)
            continue
        items = parse_items(Path(sp.pdf_path))
        items.insert(0, "speech_date", sp.speech_date)
        items.insert(
            0, "item_id", [f"ks-{sp.speech_date.isoformat()}-{n:02d}" for n in items["order"]]
        )
        items["source_url"] = sp.page_url
        frames.append(items)
    items = pd.concat(frames, ignore_index=True)
    items["ingested_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return items


# ---------------------------------------------------------------------- linking ----
# Words too common in bill titles to say which bill is meant ("Renters (Reform)" vs
# "Audit Reform"); dropped before the TF-IDF comparison, not from the exact match.
SIMILARITY_STOP_WORDS = set(
    "a an and of the for to in on with about by etc make provision provisions connected "
    "purposes reform reforms amendment amendments miscellaneous legislation bill act draft "
    "new uk united kingdom".split()
)


def _content_words(text: str) -> str:
    return " ".join(w for w in text.split() if w not in SIMILARITY_STOP_WORDS)


def _tfidf_similarity(left: list[str], right: list[str], corpus: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit([_content_words(t) for t in corpus])
    a = vec.transform([_content_words(t) for t in left])  # rows are L2-normalised
    b = vec.transform([_content_words(t) for t in right])
    return (a @ b.T).toarray()


def party_change_dissolutions(parliaments: pd.DataFrame | None = None) -> tuple[date, ...]:
    """Dissolution dates before the elections that changed the governing party, from
    ``parliaments`` when given (election year in ``PARTY_CHANGE_ELECTION_YEARS``)."""
    if parliaments is None or not len(parliaments):
        return PARTY_CHANGE_DISSOLUTIONS
    p = parliaments.dropna(subset=["dissolution_date", "election_date"])
    years = pd.to_datetime(p["election_date"]).dt.year
    found = pd.to_datetime(p.loc[years.isin(PARTY_CHANGE_ELECTION_YEARS), "dissolution_date"])
    return tuple(sorted(found.dt.date)) or PARTY_CHANGE_DISSOLUTIONS


def _speech_sessions(speech_dates: pd.Series, sessions: pd.DataFrame | None) -> dict[Any, int]:
    """Session opened by each speech: its first sitting is 0-3 days after the speech."""
    if sessions is None or not len(sessions):
        return {}
    starts = pd.to_datetime(sessions["start_date"])
    out = {}
    for d in pd.unique(speech_dates):
        lag = (starts - pd.Timestamp(d)).dt.days
        hit = sessions.loc[(lag >= 0) & (lag <= SESSION_START_MAX_LAG_DAYS), "session_id"]
        if len(hit):
            out[d] = int(hit.iloc[0])
    return out


def _item_names(item: Any) -> list[str]:
    alts = getattr(item, "alt_names", None)
    alts = alts.split("|") if isinstance(alts, str) and alts else []
    return [item.announced_name, *alts]


def link_items(
    items: pd.DataFrame,
    bills: pd.DataFrame,
    stage_sittings: pd.DataFrame,
    long_titles: dict[int, str] | None = None,
    sessions: pd.DataFrame | None = None,
    regime_ends: Sequence[date] = PARTY_CHANGE_DISSOLUTIONS,
) -> pd.DataFrame:
    """Best and second-best Government Bill for each item.

    Candidates are bills first read (stage 6 or 1) from 60 days before to 730 days after the
    speech (1,461 days for draft items; 730 days before for carried-over items), plus bills
    carried over into the session the speech opened (first read earlier, and either listing
    that session in ``included_session_ids`` or still sitting after the speech), whatever the
    notes say. No candidate is first read after the next dissolution that changed the
    governing party (``regime_ends``): the idea is censored there (DATA_CONTRACT §10.9).

    The score is 1 for an exact normalised title match on the announced name or an
    ``alt_names`` title, otherwise the larger TF-IDF character n-gram cosine of those names
    against the bill's short title, or short title + long title (generic words such as
    'reform' dropped first). Among several exact matches (a bill re-introduced in a later
    session) the one carried over into the session wins; other ties go to the bill first read
    soonest after the speech (else the latest before it).

    One bill per speech: items are settled in order (exact matches, then by score) and an
    item whose best bill is already proposed for another item of the same speech falls back
    to its next candidate. ``decided_by``: ``auto`` (exact, or similarity >=
    AUTO_MIN_SIMILARITY with a margin of AUTO_MIN_MARGIN over the second best and not
    displaced), ``rule`` (similarity >= SUGGEST_MIN_SIMILARITY: suggested, needs a check) or
    ``none`` (no link; the best candidate is still shown). Hybrid-bill items
    (``exclude_reason``) are always ``none``.
    """
    long_titles = long_titles or {}
    fr = first_readings(stage_sittings)
    cols = [c for c in ("introduced_session_id", "included_session_ids") if c in bills]
    b = bills[["bill_id", "short_title", *cols]].copy()
    b["bill_id"] = b["bill_id"].astype(int)
    b["first_reading"] = b["bill_id"].map(fr)
    last_sitting = (
        pd.to_datetime(stage_sittings["event_date"])
        .groupby(stage_sittings["bill_id"].astype(int))
        .max()
    )
    b["last_sitting"] = b["bill_id"].map(last_sitting)
    b = b[b["first_reading"].notna()].reset_index(drop=True)
    b["long_title"] = b["bill_id"].map(lambda i: long_titles.get(int(i), "")).fillna("")
    b["norm"] = b["short_title"].map(normalise_title)
    b["norm_full"] = (b["short_title"] + " " + b["long_title"]).map(normalise_title)
    included = (
        b["included_session_ids"].map(lambda v: {int(x) for x in (v or [])})
        if "included_session_ids" in b
        else pd.Series([set()] * len(b))
    )
    introduced = (
        b["introduced_session_id"].astype("Int64")
        if "introduced_session_id" in b
        else pd.Series([pd.NA] * len(b), dtype="Int64")
    )

    item_list = list(items.itertuples(index=False))
    names = [[normalise_title(n) for n in _item_names(it)] for it in item_list]
    flat = [n for ns in names for n in ns]
    corpus = flat + b["norm_full"].tolist()
    if len(b) and flat:
        sim_flat = np.maximum(
            _tfidf_similarity(flat, b["norm"].tolist(), corpus),
            _tfidf_similarity(flat, b["norm_full"].tolist(), corpus),
        )
    else:
        sim_flat = np.zeros((len(flat), len(b)))
    first = pd.to_datetime(b["first_reading"]).to_numpy()
    last = pd.to_datetime(b["last_sitting"]).to_numpy()
    norm_arr = b["norm"].to_numpy()
    speech_session = _speech_sessions(items["speech_date"], sessions)
    ends = sorted(pd.Timestamp(d) for d in regime_ends)

    cands: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    carried_vecs: list[np.ndarray] = []
    exact_flags: list[bool] = []
    censored_notes: list[str | None] = []
    k = 0
    for i, item in enumerate(item_list):
        rows = sim_flat[k : k + len(names[i])]
        k += len(names[i])
        sim = rows.max(axis=0) if len(rows) else np.zeros(len(b))
        speech = pd.Timestamp(item.speech_date)
        s64 = speech.to_datetime64()
        before = CARRIED_OVER_BEFORE_DAYS if item.is_carried_over else WINDOW_BEFORE_DAYS
        after = WINDOW_AFTER_DRAFT_DAYS if item.is_draft else WINDOW_AFTER_DAYS
        upper = speech + timedelta(days=after)
        regime_end = next((e for e in ends if e > speech), None)
        cap = min(upper, regime_end) if regime_end is not None else upper
        lower = (speech - timedelta(days=before)).to_datetime64()
        in_window = (first >= lower) & (first <= cap.to_datetime64())
        # carried over into the session this speech opened
        sess = speech_session.get(item.speech_date)
        listed = np.array(
            [
                sess is not None and sess in inc and not (pd.notna(intro) and intro == sess)
                for inc, intro in zip(included, introduced, strict=True)
            ],
            dtype=bool,
        )
        carried = (first < s64) & ((last >= s64) | listed)
        ok = in_window | carried
        score = np.where(ok, sim, -1.0)
        name_match = np.isin(norm_arr, [n for n in names[i] if n])
        exact = ok & name_match
        score[exact] = 1.0
        days = (first - s64) / np.timedelta64(1, "D")
        # among exact matches (a re-introduced bill) the one carried over into the session
        # wins; otherwise ties go to the bill first read soonest after the speech
        order = np.lexsort(
            (np.where(days >= 0, days, 1e6 - days), ~(carried & exact), -np.round(score, 6))
        )
        order = order[score[order] >= 0]
        cands.append(order)
        scores.append(score)
        carried_vecs.append(carried)
        exact_flags.append(bool(exact.any()))
        # a matching bill excluded only because a new governing party introduced it
        note = None
        if regime_end is not None and regime_end < upper:
            late = (first > regime_end.to_datetime64()) & (first <= upper.to_datetime64())
            late_score = np.where(late, np.where(name_match, 1.0, sim), -1.0)
            j = int(late_score.argmax()) if len(late_score) else -1
            if j >= 0 and late_score[j] >= SUGGEST_MIN_SIMILARITY:
                note = (
                    f"censored at {regime_end.date()} (change of governing party); later bill "
                    f"{int(b.at[j, 'bill_id'])} {b.at[j, 'short_title']} first read "
                    f"{b.at[j, 'first_reading']} (similarity {late_score[j]:.2f})"
                )
        censored_notes.append(note)

    # one bill per speech: exact matches first, then by best score
    def top(i: int) -> float:
        return float(scores[i][cands[i][0]]) if len(cands[i]) else -1.0

    excluded = [isinstance(getattr(it, "exclude_reason", None), str) for it in item_list]
    settle = sorted(range(len(item_list)), key=lambda i: (not exact_flags[i], -top(i)))
    taken: dict[tuple[Any, int], int] = {}
    picks: dict[int, tuple[int | None, str, str | None]] = {}
    for i in settle:
        c, sc, sp = cands[i], scores[i], item_list[i].speech_date
        if not len(c) or excluded[i]:
            note = "hybrid bill: not a Government Bill in bills" if excluded[i] else None
            picks[i] = (c[0] if len(c) else None, "none", note)
            continue
        if exact_flags[i]:
            j = c[0]
            picks[i] = (j, "auto", None)
            taken.setdefault((sp, j), i)
            continue
        free = [j for j in c if (sp, j) not in taken]
        j = free[0] if free else c[0]
        displaced = j != c[0]
        s1 = float(sc[j])
        s2 = max((float(sc[x]) for x in c if x != j), default=0.0)
        if not displaced and s1 >= AUTO_MIN_SIMILARITY and s1 - s2 >= AUTO_MIN_MARGIN:
            decided = "auto"
        elif s1 >= SUGGEST_MIN_SIMILARITY and (sp, j) not in taken:
            decided = "rule"
        else:
            decided = "none"
        note = None
        if (sp, c[0]) in taken and float(sc[c[0]]) >= SUGGEST_MIN_SIMILARITY:
            other = item_list[taken[(sp, c[0])]].item_id
            note = f"best candidate {int(b.at[c[0], 'bill_id'])} is linked to {other}"
        if decided == "none":
            j = c[0]  # show the best candidate for the hand-check
        else:
            taken[(sp, j)] = i
        picks[i] = (j, decided, note)

    rows = []
    for i, item in enumerate(item_list):
        j, decided, note = picks[i]
        censored = censored_notes[i] if decided != "auto" else None
        source = getattr(item, "item_source", None)
        tag_note = (
            "added from a speech tag: check it is not the same commitment as a contents item"
            if source == "speech_tag"
            else None
        )
        notes = [n for n in (tag_note, note, censored) if n]
        others = [x for x in cands[i] if x != j]
        second = others[0] if others else None
        s1 = float(scores[i][j]) if j is not None else None
        s2 = float(scores[i][second]) if second is not None else None
        rows.append(
            {
                "item_id": item.item_id,
                "bill_id": int(b.at[j, "bill_id"]) if j is not None else None,
                "bill_title": b.at[j, "short_title"] if j is not None else None,
                "bill_first_reading": b.at[j, "first_reading"] if j is not None else None,
                "similarity": round(s1, 3) if s1 is not None else None,
                "second_bill_id": int(b.at[second, "bill_id"]) if second is not None else None,
                "second_bill_title": b.at[second, "short_title"] if second is not None else None,
                "second_similarity": round(s2, 3) if s2 is not None else None,
                "decided_by": decided,
                "exact_match": exact_flags[i],
                "bill_carried_over": bool(carried_vecs[i][j]) if j is not None else False,
                "review_note": "; ".join(notes) or None,
            }
        )
    links = pd.DataFrame(rows)
    for col in ("bill_id", "second_bill_id"):
        links[col] = links[col].astype("Int64")
    return links


def apply_carry_over(items: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """``is_carried_over`` = the notes say so, or the proposed bill (``auto`` / ``rule``) was
    carried over into the session the speech opened; ``carried_over_source`` says which
    (``'text'``, ``'bill'`` or ``'text+bill'``)."""
    out = items.copy()
    linked = links.set_index("item_id")
    bill = (
        (
            out["item_id"].map(
                linked["bill_carried_over"] & linked["decided_by"].isin(["auto", "rule"])
            )
        )
        .fillna(False)
        .astype(bool)
    )
    text = out["is_carried_over"].astype(bool)
    out["carried_over_source"] = [
        "text+bill" if t and bl else "text" if t else "bill" if bl else None
        for t, bl in zip(text, bill, strict=True)
    ]
    out["is_carried_over"] = text | bill
    return out


def export_review_csv(items: pd.DataFrame, links: pd.DataFrame, path: Path = REVIEW_CSV) -> Path:
    """Write the hand-check sheet. ``bill_id`` is always the best candidate; it is a proposed
    link only when ``decided_by`` is ``auto`` or ``rule``. Fill ``correct`` (y/n) for the
    proposed outcome (a link, or no link when ``decided_by`` is ``none``) and, when it is wrong,
    ``correct_bill_id`` (a bill ID, or ``none``)."""
    df = items.merge(links, on="item_id", how="left").sort_values(["speech_date", "order"])
    df["correct"] = ""
    df["correct_bill_id"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=REVIEW_COLUMNS).to_csv(path, index=False)
    return path


# ------------------------------------------------------------------------ run ----
def summarise(speeches: pd.DataFrame, items: pd.DataFrame, links: pd.DataFrame) -> dict[str, Any]:
    df = items.merge(links, on="item_id", how="left")
    n = len(df)
    shares = df["decided_by"].value_counts().to_dict()
    years = Counter(d.year for d in speeches["speech_date"])
    return {
        "speeches_found": len(speeches),
        "speeches_expected": len(EXPECTED_SPEECH_YEARS),
        "missing_years": sorted((Counter(EXPECTED_SPEECH_YEARS) - years).elements()),
        "items_per_speech": {str(k): int(v) for k, v in df.groupby("speech_date").size().items()},
        "items": n,
        "drafts": int(df["is_draft"].sum()),
        "carried_over": int(df["is_carried_over"].sum()),
        "carried_over_by_source": {
            str(k): int(v) for k, v in df["carried_over_source"].value_counts().items()
        }
        if "carried_over_source" in df
        else {},
        "excluded_hybrid": int((df["exclude_reason"] == "hybrid_bill").sum()),
        "with_department": int(df["department"].notna().sum()),
        "items_by_source": {str(k): int(v) for k, v in df["item_source"].value_counts().items()},
        "from_fallback": int((df["item_source"] == "headings").sum()),
        "auto": int(shares.get("auto", 0)),
        "rule": int(shares.get("rule", 0)),
        "none": int(shares.get("none", 0)),
        "pct_auto": round(100 * shares.get("auto", 0) / n, 1) if n else 0.0,
        "pct_rule": round(100 * shares.get("rule", 0) / n, 1) if n else 0.0,
        "pct_none": round(100 * shares.get("none", 0) / n, 1) if n else 0.0,
        "date_notes": {
            str(r.speech_date): r.date_note for r in speeches.itertuples() if r.date_note
        },
    }


REFERENCE_QUERIES = {
    "bills": (
        "SELECT bill_id, short_title, introduced_session_id, introduced_date, carry_over_date, "
        "included_session_ids FROM bills ORDER BY bill_id"
    ),
    "stage_sittings": (
        "SELECT bill_id, api_stage_id, session_id, event_date FROM stage_sittings "
        "ORDER BY bill_id, event_date"
    ),
    "sessions": (
        "SELECT session_id, parliament_id, start_date, end_date, end_type FROM sessions "
        "ORDER BY session_id"
    ),
    "parliaments": (
        "SELECT parliament_id, first_meeting_date, dissolution_date, election_date "
        "FROM parliaments ORDER BY parliament_id"
    ),
}


def read_reference_tables() -> dict[str, pd.DataFrame]:
    """``bills``, ``stage_sittings``, ``sessions`` and ``parliaments`` from the production
    database in a READ ONLY transaction.

    Plain SELECTs, unlike the ``westminster_weather.db.fetch_*`` adapters, which first run
    ``ensure_reference_tables_seeded_sync()`` (create_all plus seed merges and a commit)."""
    from sqlalchemy import text

    from westminster_weather.db import get_sync_pooled_engine

    engine = get_sync_pooled_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            out = {name: pd.read_sql(text(q), conn) for name, q in REFERENCE_QUERIES.items()}
            conn.rollback()
    finally:
        engine.dispose()
    out["bills"]["included_session_ids"] = out["bills"]["included_session_ids"].map(
        lambda v: list(json.loads(v) if isinstance(v, str) else (v or []))
    )
    for name, col in (("stage_sittings", "event_date"), ("sessions", "start_date")):
        out[name][col] = pd.to_datetime(out[name][col]).dt.date
    return out


def run_tier_a(
    refresh: bool = False,
    review_path: Path = REVIEW_CSV,
    reference: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Discover, parse, link and export the review CSV.

    ``reference`` holds the ``bills``, ``stage_sittings``, ``sessions`` and ``parliaments``
    frames; when omitted they are read with :func:`read_reference_tables` (SELECTs in a
    READ ONLY transaction, so nothing is written to the database)."""
    reference = read_reference_tables() if reference is None else reference
    bills, stage_sittings = reference["bills"], reference["stage_sittings"]
    speeches = discover_speeches(reference["sessions"], refresh=refresh)
    items = build_items(speeches)
    long_titles = bill_long_titles(bills["bill_id"])
    regime_ends = party_change_dissolutions(reference.get("parliaments"))
    links = link_items(
        items,
        bills,
        stage_sittings,
        long_titles,
        sessions=reference["sessions"],
        regime_ends=regime_ends,
    )
    items = apply_carry_over(items, links)
    export_review_csv(items, links, review_path)
    summary = summarise(speeches, items, links)
    summary["review_csv"] = str(review_path)
    return items, links, summary
