"""Offline tests for Tier A: briefing-note parsing, title normalisation and bill linking."""

from datetime import date

import pandas as pd
import pytest

from westminster_weather.takeup.bill_text import first_readings
from westminster_weather.takeup.kings_speech import (
    apply_carry_over,
    export_review_csv,
    link_items,
    normalise_title,
    parse_items,
    party_change_dissolutions,
    speech_tags,
)

COVER = "THE KING’S SPEECH 2024\nPrime Minister’s Office\n10 Downing Street\n17 July 2024\n1"

CONTENTS = """2
CONTENTS
AN INTRODUCTION FROM THE PRIME MINISTER 3
ECONOMIC STABILITY 3
Energy Bill 3
Draft Water Bill 4
Housing 4
Renters’ Rights Bill [HL] 5
High Speed Rail 2 (West Midlands - Crewe) Bill 6
Animal welfare legislation 7
NON-LEGISLATIVE BRIEFS 8
Public finances 8
CONTACT DETAILS 9
"""

BODY = [
    "3\nEnergy Bill\nThe purpose of the Bill is to reform the energy market.",
    "4\nDraft Water Bill\nA draft Bill will be published.\nHousing\nMore homes.",
    "5\nRenters’ Rights Bill [HL]\nThe Bill was carried over from the previous session.",
    "6\nHigh Speed Rail 2 (West Midlands - Crewe) Bill\nThere was insufficient time for it "
    "to be carried over before dissolution.",
    "7\nAnimal welfare legislation\nWe will legislate.",
    "8\nPublic finances\nNothing here.",
    "9\nCONTACT DETAILS\nBill Lead Department Contact details\n"
    "Energy Bill Department for Energy Security & Net Zero 020 7215 1000\n"
    "Water Bill (draft) Department for Environment, Food and\nRural Affairs 0330 041 6560\n"
    "Renters’ Rights Bill [HL] Ministry of Housing, Communities and Local Government "
    "030 3444 1209",
]

EXPECTED_NAMES = [
    "Energy Bill",
    "Draft Water Bill",
    "Renters' Rights Bill [HL]",
    "High Speed Rail 2 (West Midlands - Crewe) Bill",
    "Animal welfare legislation",
]


def _document(contents: str) -> str:
    return "\f".join([COVER, contents, *BODY])


def _one_word_per_line(text: str) -> str:
    return "\n".join(text.split())


# ------------------------------------------------------------------ contents parsing ----
def test_contents_normal_lines():
    items = parse_items(_document(CONTENTS))
    assert items["announced_name"].tolist() == EXPECTED_NAMES
    assert items["order"].tolist() == [1, 2, 3, 4, 5]
    assert items["from_contents"].all()


def test_contents_one_word_per_line():
    """Some PDFs (e.g. 2024) extract one word per line; whitespace is joined first."""
    doc = "\f".join(_one_word_per_line(p) for p in _document(CONTENTS).split("\f"))
    items = parse_items(doc)
    assert items["announced_name"].tolist() == EXPECTED_NAMES


def test_contents_pages_layout_with_date_header():
    """2012-2015 layout: name on one line, 'Pages 3-6' on the next, a dated page header."""
    contents = (
        "1\n09 May 2012\nTHE QUEEN’S SPEECH 2012 – BRIEFING NOTES\n"
        "Energy Bill\nPages 3-4\nDraft Water Bill\nPage 4\nHousing\nPage 4\n"
        "Renters’ Rights Bill [HL]\nPages 5\nHigh Speed Rail 2 (West Midlands - Crewe) Bill\n"
        "Page 6\nAnimal welfare legislation\nPages 7-8\n"
    )
    items = parse_items(_document(contents))
    assert items["announced_name"].tolist() == EXPECTED_NAMES


def test_contents_dot_leaders_upper_case_and_carry_over_heading():
    """2016 layout: upper-case names with dot leaders and a 'continuing' section."""
    contents = (
        "CONTENTS\nAN INTRODUCTION ......... 3\nENERGY BILL ................ 3\n"
        "DRAFT WATER BILL ......... 4\nBILLS CONTINUING FROM 2015/16 SESSION ...... 5\n"
        "RENTERS’ RIGHTS BILL [HL] ....... 5\nOTHER MEASURES ...... 6\n"
        "NHS (OVERSEAS VISITORS) BILL ....... 6\n"
    )
    items = parse_items(_document(contents)).set_index("announced_name")
    # mixed-case names come from the body text when it has them
    assert list(items.index[:3]) == ["Energy Bill", "Draft Water Bill", "Renters' Rights Bill [HL]"]
    assert items.loc["Renters' Rights Bill [HL]", "is_carried_over"]
    assert not items.loc["Energy Bill", "is_carried_over"]
    assert "NHS (Overseas Visitors) Bill" in items.index  # title-cased, NHS kept
    assert not items.loc["NHS (Overseas Visitors) Bill", "is_carried_over"]  # new section


def test_draft_carry_over_and_department():
    items = parse_items(_document(CONTENTS)).set_index("announced_name")
    assert items["is_draft"].tolist() == [False, True, False, False, False]
    assert items.loc["Renters' Rights Bill [HL]", "is_carried_over"]
    # "insufficient time for it to be carried over" is not a carry-over
    assert not items.loc["High Speed Rail 2 (West Midlands - Crewe) Bill", "is_carried_over"]
    assert items.loc["Energy Bill", "department"] == "Department for Energy Security and Net Zero"
    assert items.loc["Draft Water Bill", "department"] == (
        "Department for Environment, Food and Rural Affairs"
    )
    assert items.loc["Renters' Rights Bill [HL]", "department"] == (
        "Ministry of Housing, Communities and Local Government"
    )
    assert items.loc["Animal welfare legislation", "department"] is None


def test_fallback_to_bill_headings_without_contents():
    doc = "\f".join([COVER, "Energy Bill\nSome text about energy.", "Draft Water Bill\nMore text."])
    items = parse_items(doc)
    assert items["announced_name"].tolist() == ["Energy Bill", "Draft Water Bill"]
    assert not items["from_contents"].any()
    assert items["is_draft"].tolist() == [False, True]


# --------------------------------------------------------------- title normalisation ----
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Renters’ Rights Bill", "renters rights"),
        ("Renters' Rights Act 2025", "renters rights"),
        ("Draft Water Bill", "water"),
        ("Finance (No. 2) Bill [HL]", "finance"),
        ("Bus Services Act 2017 ch 21", "bus services"),
        ("EU (Withdrawal Agreement) Bill", "european union withdrawal agreement"),
        (
            "Digital Markets, Competition and Consumer Bill (draft)",
            "digital markets competition and consumer",
        ),
        ("The Crown Estate Bill", "crown estate"),
    ],
)
def test_normalise_title(title, expected):
    assert normalise_title(title) == expected


# -------------------------------------------------------------------------- linking ----
SPEECH = date(2020, 1, 15)


def _bills_and_sittings():
    bills = pd.DataFrame(
        [
            (1, "Energy Act 2021"),  # exact match, first read after the speech
            (2, "Small Charitable Donations Act 2021"),  # renamed bill
            (3, "Housing Act 2023"),  # exact title but first read > 730 days after
            (4, "Pensions Act 2019"),  # exact title but first read 100 days before
            (5, "Domestic Abuse Act 2021"),  # first read earlier, carried over (alive)
            (6, "Fisheries Act 2020"),  # same title introduced twice: 6 then 7
            (7, "Fisheries Bill"),
        ],
        columns=["bill_id", "short_title"],
    )
    rows = [
        (1, 6, "2020-03-01"),
        (1, 7, "2020-04-01"),
        (2, 1, "2020-02-01"),
        (3, 6, "2022-06-01"),
        (4, 6, "2019-10-07"),
        (4, 7, "2019-10-20"),
        (5, 6, "2019-07-16"),
        (5, 8, "2020-03-01"),
        (6, 6, "2020-01-01"),
        (7, 6, "2020-02-10"),
        (1, 1, "2020-06-01"),  # Lords first reading after the Commons one: ignored
    ]
    sittings = pd.DataFrame(
        [(b, s, date.fromisoformat(d)) for b, s, d in rows],
        columns=["bill_id", "api_stage_id", "event_date"],
    )
    return bills, sittings


def _items(names, carried=None, speech=SPEECH, **extra):
    carried = carried or [False] * len(names)
    return pd.DataFrame(
        {
            "item_id": [f"ks-{speech}-{i + 1:02d}" for i in range(len(names))],
            "speech_date": speech,
            "order": range(1, len(names) + 1),
            "announced_name": names,
            "is_draft": [n.startswith("Draft") for n in names],
            "is_carried_over": carried,
            "department": None,
            **extra,
        }
    )


def _bills(rows):
    """(bill_id, short_title, first reading, [later sitting dates]) -> bills, stage_sittings."""
    bills = pd.DataFrame([(b, t) for b, t, *_ in rows], columns=["bill_id", "short_title"])
    sittings = [(b, 6, date.fromisoformat(fr)) for b, _, fr, *_ in rows]
    sittings += [(b, 7, date.fromisoformat(d)) for b, _, _, *later in rows for d in later]
    return bills, pd.DataFrame(sittings, columns=["bill_id", "api_stage_id", "event_date"])


def test_first_readings_takes_earliest_stage_6_or_1():
    _, sittings = _bills_and_sittings()
    fr = first_readings(sittings)
    assert fr[1] == date(2020, 3, 1)
    assert fr[2] == date(2020, 2, 1)
    assert fr[5] == date(2019, 7, 16)


def test_link_exact_renamed_and_no_match():
    bills, sittings = _bills_and_sittings()
    items = _items(["Energy Bill", "Small Donations Bill", "Votes for Life Bill"])
    links = link_items(items, bills, sittings, {2: "A Bill to make provision about donations"})
    links = links.set_index("item_id")
    exact, renamed, nothing = links.iloc[0], links.iloc[1], links.iloc[2]
    assert exact["bill_id"] == 1 and exact["decided_by"] == "auto" and exact["similarity"] == 1.0
    assert renamed["bill_id"] == 2 and renamed["decided_by"] in {"auto", "rule"}
    assert not renamed["exact_match"]
    assert nothing["decided_by"] == "none"


def test_link_date_window_excludes_out_of_window_bills():
    bills, sittings = _bills_and_sittings()
    links = link_items(_items(["Housing Bill", "Pensions Bill"]), bills, sittings)
    assert links["decided_by"].tolist() == ["none", "none"]
    assert 3 not in links["bill_id"].tolist() and 4 not in links["bill_id"].tolist()
    # a carried-over item may link to a bill first read before the window
    carried = link_items(_items(["Pensions Bill"], carried=[True]), bills, sittings)
    assert carried.loc[0, "bill_id"] == 4


def test_link_bill_still_alive_at_speech_is_a_candidate():
    bills, sittings = _bills_and_sittings()
    links = link_items(_items(["Domestic Abuse Bill"]), bills, sittings)
    assert links.loc[0, "bill_id"] == 5 and links.loc[0, "decided_by"] == "auto"


def test_link_same_title_twice_prefers_first_after_speech():
    bills, sittings = _bills_and_sittings()
    links = link_items(_items(["Fisheries Bill"]), bills, sittings)
    assert links.loc[0, "bill_id"] == 7
    assert links.loc[0, "second_bill_id"] == 6


def test_suggestion_for_bill_auto_linked_in_same_speech_is_dropped():
    bills, sittings = _bills_and_sittings()
    alone = link_items(_items(["Energy Market Bill"]), bills, sittings)
    assert alone.loc[0, "bill_id"] == 1 and alone.loc[0, "decided_by"] == "rule"
    both = link_items(_items(["Energy Bill", "Energy Market Bill"]), bills, sittings)
    assert both["decided_by"].tolist() == ["auto", "none"]
    assert both.loc[1, "bill_id"] == 1  # best candidate still shown for the hand-check


def test_export_review_csv(tmp_path):
    bills, sittings = _bills_and_sittings()
    items = _items(["Energy Bill", "Votes for Life Bill"])
    links = link_items(items, bills, sittings)
    path = export_review_csv(items, links, tmp_path / "links.csv")
    out = pd.read_csv(path)
    assert out.columns[-2:].tolist() == ["correct", "correct_bill_id"]
    assert out["item_id"].tolist() == items["item_id"].tolist()
    assert out["correct"].isna().all()


# ------------------------------------------------------- carry-over, windows, clashes ----
def test_exact_matches_prefer_the_bill_carried_over_into_the_session():
    """Oct 2019: the Domestic Abuse Bill (2444) was carried over; 2709 is its 2020
    re-introduction. Both match exactly; the carried-over bill wins."""
    bills, sittings = _bills(
        [
            (2444, "Domestic Abuse Bill", "2019-07-16", "2019-10-31"),
            (2709, "Domestic Abuse Act 2021", "2020-03-03"),
        ]
    )
    for carried in (False, True):
        links = link_items(
            _items(["Domestic Abuse Bill"], [carried], speech=date(2019, 10, 14)), bills, sittings
        )
        assert links.loc[0, "bill_id"] == 2444
        assert links.loc[0, "second_bill_id"] == 2709
        assert links.loc[0, "bill_carried_over"]


def test_carry_over_from_included_sessions_without_later_sittings():
    """A bill carried into the speech's session but with no sitting after the speech is still
    carried over (included_session_ids lists the session)."""
    bills, sittings = _bills(
        [
            (2410, "Birmingham Commonwealth Games Bill [HL]", "2019-06-05"),
            (2523, "Birmingham Commonwealth Games Act 2020", "2020-01-07"),
        ]
    )
    bills["introduced_session_id"] = [30, 35]
    bills["included_session_ids"] = [[30, 34], [35]]
    sessions = pd.DataFrame(
        {
            "session_id": [30, 34, 35],
            "start_date": [date(2017, 6, 22), date(2019, 10, 15), date(2019, 12, 19)],
        }
    )
    items = _items(["Birmingham Commonwealth Games Bill"], speech=date(2019, 10, 14))
    links = link_items(items, bills, sittings, sessions=sessions)
    assert links.loc[0, "bill_id"] == 2410 and links.loc[0, "bill_carried_over"]
    # without the session data the bill is not a candidate (first read > 60 days before)
    assert link_items(items, bills, sittings).loc[0, "bill_id"] == 2523
    flagged = apply_carry_over(items, links)
    assert flagged.loc[0, "is_carried_over"] and flagged.loc[0, "carried_over_source"] == "bill"


def test_no_link_to_a_bill_first_read_after_a_change_of_governing_party():
    bills, sittings = _bills(
        [
            (3765, "Terrorism (Protection of Premises) Act", "2024-09-12"),
            (1, "Energy Act", "2024-01-10"),
        ]
    )
    items = _items(
        ["Terrorism (Protection of Premises) Bill", "Energy Bill"], speech=date(2023, 11, 7)
    )
    links = link_items(items, bills, sittings).set_index("item_id")
    first, second = links.iloc[0], links.iloc[1]
    assert first["decided_by"] == "none" and first["bill_id"] != 3765
    assert "censored at 2024-05-30" in first["review_note"] and "3765" in first["review_note"]
    assert second["bill_id"] == 1 and second["decided_by"] == "auto"
    # without the party-change dissolution the later bill would be linked
    free = link_items(items, bills, sittings, regime_ends=()).set_index("item_id")
    assert free.iloc[0]["bill_id"] == 3765


def test_party_change_dissolutions_from_parliaments():
    parliaments = pd.DataFrame(
        {
            "parliament_id": [2, 3, 6, 7],
            "dissolution_date": [date(2010, 4, 12), date(2015, 3, 30), date(2024, 5, 30), None],
            "election_date": [date(2010, 5, 6), date(2015, 5, 7), date(2024, 7, 4), None],
        }
    )
    assert party_change_dissolutions(parliaments) == (date(2010, 4, 12), date(2024, 5, 30))
    assert party_change_dissolutions(None) == (date(2010, 4, 12), date(2024, 5, 30))


def test_draft_items_get_a_longer_window():
    """Draft Patient Safety Bill (2017-06-21) -> HSSI Bill first read 846 days later."""
    bills, sittings = _bills([(2466, "Patient Safety Investigations Bill", "2019-10-15")])
    speech = date(2017, 6, 21)
    draft = link_items(_items(["Draft Patient Safety Bill"], speech=speech), bills, sittings)
    assert draft.loc[0, "bill_id"] == 2466
    plain = link_items(_items(["Patient Safety Bill"], speech=speech), bills, sittings)
    assert pd.isna(plain.loc[0, "bill_id"])


def test_clash_falls_back_to_the_next_free_candidate():
    """An item whose best bill is auto-linked to another item of the speech is suggested its
    next candidate (here >= SUGGEST_MIN_SIMILARITY) instead of 'none'."""
    bills, sittings = _bills(
        [(1, "Energy Act 2018", "2017-09-13"), (2, "Energy Prices Act 2018", "2017-10-22")]
    )
    speech = date(2017, 6, 21)
    alone = link_items(_items(["Energy Price Cap Bill"], speech=speech), bills, sittings)
    assert alone.loc[0, "bill_id"] == 2 and alone.loc[0, "second_bill_id"] == 1
    items = _items(["Energy Prices Bill", "Energy Price Cap Bill"], speech=speech)
    links = link_items(items, bills, sittings)
    assert links["bill_id"].tolist() == [2, 1]
    assert links["decided_by"].tolist() == ["auto", "rule"]
    assert "best candidate 2 is linked to" in links.loc[1, "review_note"]


def test_two_suggestions_never_share_a_bill():
    bills, sittings = _bills([(4140, "National Security (State Threats) Act 2026", "2026-06-09")])
    items = _items(
        ["Tackling State Threats Bill", "National Security Bill"], speech=date(2026, 5, 13)
    )
    links = link_items(items, bills, sittings)
    proposed = links[links["decided_by"].isin(["auto", "rule"])]
    assert len(proposed) == 1 and proposed["bill_id"].tolist() == [4140]


def test_alt_names_give_an_exact_match_and_hybrid_items_are_never_linked():
    bills, sittings = _bills(
        [
            (4083, "Courts and Tribunals Bill", "2026-06-01"),
            (9, "High Speed Rail Act", "2026-06-02"),
        ]
    )
    items = _items(
        ["Courts Modernisation Bill", "Northern Powerhouse Rail Bill"],
        speech=date(2026, 5, 13),
        alt_names=["Courts and Tribunals Bill", "High Speed Rail Bill"],
        exclude_reason=[None, "hybrid_bill"],
    )
    links = link_items(items, bills, sittings)
    assert links.loc[0, "bill_id"] == 4083 and links.loc[0, "decided_by"] == "auto"
    assert links.loc[1, "decided_by"] == "none"
    assert "hybrid" in links.loc[1, "review_note"]


# ---------------------------------------------------------- parsing: tags, flags, depts ----
def test_speech_tags_split_multi_bill_tags():
    text = (
        "My Government will act [Domestic Abuse Bill, Divorce, Dissolution and Separation Bill]."
        " Energy [Energy Bill – carried over from previous session]. [Finance Bill and other "
        "legislation] [Bill of Rights] [Hillsborough Law] [not a tag]"
    )
    assert speech_tags(text) == [
        "Domestic Abuse Bill",
        "Divorce, Dissolution and Separation Bill",
        "Energy Bill",
        "Finance Bill",
        "Bill of Rights",
        "Hillsborough Law",
    ]


SPEECH_TEXT = (
    "My Government will improve connectivity [Telecommunications Infrastructure (Leasehold "
    "Property) Bill] and protect animals [Animal Welfare (Sentience) Bill, Kept Animals Bill] "
    "and fund the NHS [NHS Funding Bill, NHS Long Term Plan Bill] and build a railway "
    "[HS2 2A Bill]. We will also publish a draft Water Bill."
)
TAG_CONTENTS = """CONTENTS
Energy Bill 3
Water Bill 4
Animal Welfare Plan and Legislation 5
NHS Funding Bill and NHS Long Term Plan 6
High Speed Rail (West Midlands - Crewe) Bill 7
Northern Ireland Bill 8
"""
TAG_BODY = [
    "3\nEnergy Bill\nThe Bill was introduced in the last session.",
    "4\nWater Bill\nMore water.",
    "5\nAnimal Welfare Plan and Legislation\nAnimals.",
    "6\nNHS Funding Bill and NHS Long Term Plan\nMoney.",
    "7\nHigh Speed Rail (West Midlands - Crewe) Bill\nTrains.",
    "8\nNorthern Ireland (Stormont House Agreement) Bill\nThis Bill (formally titled the "
    "Northern Ireland (Legacy) Bill) will implement the agreement.",
    "9\nCONTACT DETAILS\nBill Lead Department Contact details\n"
    "Planning and Water Bill (draft) Department for Environment, Food and Rural Affairs "
    "0330 041 6560\nEnergy Bill Office for Equality and Opportunity 0207 276 3999",
]


def test_parse_adds_untagged_bills_and_flags():
    doc = "\f".join([COVER + "\n" + SPEECH_TEXT, TAG_CONTENTS, *TAG_BODY])
    items = parse_items(doc).set_index("announced_name")
    tagged = items[items["item_source"] == "speech_tag"].index.tolist()
    # the combined animal item keeps one tag, the other becomes an item; the NHS tags are
    # both inside the NHS item; the HS2 tag is covered by the hybrid-bill item
    assert tagged == [
        "Telecommunications Infrastructure (Leasehold Property) Bill",
        "Kept Animals Bill",
    ]
    assert not items.loc[tagged, "from_contents"].any()
    assert items["order"].tolist() == list(range(1, 9))  # tags appended after contents
    assert items.loc["High Speed Rail (West Midlands - Crewe) Bill", "exclude_reason"] == (
        "hybrid_bill"
    )
    assert items["exclude_reason"].notna().sum() == 1
    assert items.loc["Energy Bill", "is_carried_over"]  # "introduced in the last session"
    assert items.loc["Water Bill", "is_draft"]  # "publish a draft Water Bill"
    assert not items.loc["Energy Bill", "is_draft"]
    alts = items.loc["Northern Ireland Bill", "alt_names"].split("|")
    assert alts == [
        "Northern Ireland (Legacy) Bill",
        "Northern Ireland (Stormont House Agreement) Bill",
    ]
    # contact-table rows matched without the exact name, and a newer department
    assert items.loc["Water Bill", "department"] == (
        "Department for Environment, Food and Rural Affairs"
    )
    assert items.loc["Energy Bill", "department"] == "Office for Equality and Opportunity"


def test_carried_over_text_variants():
    body = list(BODY)
    body[0] = "3\nEnergy Bill\nThe Bill was introduced in the previous session."
    items = parse_items("\f".join([COVER, CONTENTS, *body])).set_index("announced_name")
    assert items.loc["Energy Bill", "is_carried_over"]
