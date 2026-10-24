"""Offline tests for the take-up data layer: tables, engines and get_takeup_signals(as_of)."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from westminster_weather import db as app_db
from westminster_weather.data import calendar
from westminster_weather.takeup import db as takeup_db
from westminster_weather.takeup import tables
from westminster_weather.takeup.signals import TakeupData, department_slug

NOW = datetime(2026, 10, 24, tzinfo=UTC)
D = date.fromisoformat


def _parliaments() -> pd.DataFrame:
    return pd.DataFrame(
        app_db.DEFAULT_PARLIAMENTS,
        columns=["parliament_id", "first_meeting_date", "dissolution_date", "election_date"],
    )


def _mention(mid, idea_date, source="wms", firmness=1, dept=None, minister=True, commit=True):
    return {
        "mention_id": mid,
        "source": source,
        "doc_id": mid.rsplit("-", 1)[0],
        "event_date": D(idea_date),
        "member_id": 1,
        "is_minister": minister,
        "department": dept,
        "title": "t",
        "sentence": "s",
        "snippet": "s",
        "phrase": "we will legislate",
        "firmness": firmness,
        "is_commitment": commit,
        "topic": "topic",
        "labelled_by": "rules",
        "ingested_at": NOW,
    }


@pytest.fixture
def synthetic() -> dict[str, pd.DataFrame]:
    """Ideas, mentions, events and bills on both sides of the test as_of dates."""
    mentions = [
        # nlp-00001: two topic groups, merged because both link to bill 101 (first read 2016-03-01).
        _mention("wms-1-0", "2015-07-01", firmness=1, dept="Department for Education"),
        _mention("wms-2-0", "2015-09-01", firmness=2),
        _mention("wms-2-1", "2015-09-01", firmness=2),  # same statement: not a repeat
        _mention("han-3-0", "2015-11-01", source="hansard_spoken", firmness=0),
        _mention("wms-4-0", "2016-05-01", firmness=2),
        _mention("wms-5-0", "2015-08-01", minister=False),  # not a minister: invisible
        _mention("wms-6-0", "2015-06-01", commit=False),  # not a commitment: invisible
        _mention("wms-7-0", "2016-02-01"),  # nlp-00002, bill 102 (Lords first)
        _mention("wms-8-0", "2023-01-10", dept="Home Office"),  # nlp-00003, bill after 2024 change
        _mention("wms-9-0", "2022-02-01"),  # nlp-00004, bill before the 2024 dissolution
        _mention("hanw-10-0", "2021-01-05", source="hansard_written"),  # nlp-00005
        _mention("wms-11-0", "2016-01-10", firmness=2),  # nlp-00006, same-party 2017 election
        _mention("wms-12-0", "2008-01-01"),  # nlp-00007, 2010 change of government
        _mention("wms-13-0", "2027-01-01", firmness=2),  # future mention of nlp-00006
    ]
    ks = [
        {
            "item_id": "ks-2016-05-18-01",
            "speech_date": D("2016-05-18"),
            "announced_name": "Prisons Bill",
            "is_draft": False,
            "is_carried_over": False,
            "department": "Ministry of Justice",
            "bill_id": 105,
            "source_url": "u",
            "ingested_at": NOW,
        },
        {
            "item_id": "ks-2016-05-18-02",
            "speech_date": D("2016-05-18"),
            "announced_name": "Topic Six Bill",
            "is_draft": False,
            "is_carried_over": False,
            "department": None,
            "bill_id": None,
            "source_url": "u",
            "ingested_at": NOW,
        },
    ]
    idea_mentions = [
        ("nlp-00001", "wms-1-0", "g1"),
        ("nlp-00001", "wms-2-0", "g1"),
        ("nlp-00001", "wms-2-1", "g1"),
        ("nlp-00001", "wms-5-0", "g1"),
        ("nlp-00001", "wms-6-0", "g1"),
        ("nlp-00001", "han-3-0", "g2"),
        ("nlp-00001", "wms-4-0", "g2"),
        ("nlp-00002", "wms-7-0", "g3"),
        ("nlp-00003", "wms-8-0", "g4"),
        ("nlp-00004", "wms-9-0", "g5"),
        ("nlp-00005", "hanw-10-0", "g6"),
        ("nlp-00006", "wms-11-0", "g7"),
        ("nlp-00006", "wms-13-0", "g7"),
        ("nlp-00007", "wms-12-0", "g8"),
        ("ks-2016-05-18-01", "ks-2016-05-18-01", "ks-2016-05-18-01"),
    ]
    ideas = [
        {
            "idea_id": i,
            "origin": "kings_speech" if i.startswith("ks-") else "nlp",
            "topic": f"topic {i}",
            "trigger_date": D("2015-01-01"),
            "department": None,
            "first_source": "kings_speech" if i.startswith("ks-") else "wms",
            "initial_firmness": 1,
            "n_mentions": 1,
            "ingested_at": NOW,
        }
        for i in sorted({r[0] for r in idea_mentions})
    ]
    links = [
        ("nlp-00001", 101),
        ("nlp-00002", 102),
        ("nlp-00003", 103),
        ("nlp-00004", 104),
        ("nlp-00005", None),
        ("ks-2016-05-18-01", 105),
    ]
    events = [
        ("ks-2016-05-18-01", "consultation", "2015-12-01"),
        ("ks-2016-05-18-01", "white_paper", "2016-02-01"),
        ("ks-2016-05-18-01", "draft_bill", "2017-01-01"),
        ("nlp-00006", "consultation_response", "2019-01-01"),
        # ks-2016-05-18-02 matched by text to nlp-00006's earlier commitment: an event, not
        # an idea of its own (and no bill yet)
        ("nlp-00006", "kings_speech", "2016-05-18"),
    ]
    sittings = [
        (1, 101, 6, "2016-03-01"),
        (2, 101, 7, "2016-04-01"),
        (3, 102, 1, "2016-06-10"),  # Lords first reading
        (4, 102, 6, "2016-11-01"),  # Commons first reading in the second House
        (5, 103, 6, "2025-01-15"),
        (6, 104, 6, "2023-06-01"),
        (7, 105, 6, "2016-06-20"),
        (8, 999, 6, "2014-01-01"),  # not a Government Bill
    ]
    sessions = [
        (20, 2, "2009-11-18"),
        (21, 3, "2010-05-25"),
        (22, 3, "2012-05-09"),
        (23, 4, "2015-05-27"),
        (24, 5, "2017-06-21"),
        (25, 6, "2019-12-19"),
        (26, 6, "2021-05-11"),
        (27, 7, "2024-07-17"),
    ]
    sitting_days = pd.date_range("2010-01-04", "2026-10-22", freq="W-TUE")
    return {
        "signal_mentions": pd.DataFrame(mentions),
        "kings_speech_items": pd.DataFrame(ks).astype({"bill_id": "Int64"}),
        "ideas": pd.DataFrame(ideas),
        "idea_mentions": pd.DataFrame(idea_mentions, columns=["idea_id", "mention_id", "group_id"]),
        "idea_events": pd.DataFrame(
            [
                {
                    "idea_id": i,
                    "event_type": t,
                    "event_date": D(d),
                    "title": t,
                    "source_url": "u",
                    "ingested_at": NOW,
                }
                for i, t, d in events
            ]
        ),
        "idea_bill_links": pd.DataFrame(
            [
                {
                    "idea_id": i,
                    "bill_id": b,
                    "similarity": 0.9,
                    "decided_by": "hand",
                    "checked": True,
                    "ingested_at": NOW,
                }
                for i, b in links
            ]
        ),
        "bills": pd.DataFrame({"bill_id": [101, 102, 103, 104, 105]}),
        "stage_sittings": pd.DataFrame(
            [
                {
                    "sitting_id": s,
                    "bill_id": b,
                    "api_stage_id": st,
                    "house": "Commons" if st == 6 else "Lords",
                    "event_date": D(d),
                }
                for s, b, st, d in sittings
            ]
        ),
        "sessions": pd.DataFrame(
            [
                {"session_id": s, "parliament_id": p, "start_date": D(d), "end_date": None}
                for s, p, d in sessions
            ]
        ),
        "parliaments": _parliaments(),
        "sitting_days": pd.DataFrame(
            {"date": [d.date() for d in sitting_days], "house": "Commons"}
        ),
    }


def _signals(tables_, as_of) -> pd.DataFrame:
    return TakeupData(tables_).get_takeup_signals(as_of, min_positive=0).set_index("signal_id")


def _truncate(t: dict[str, pd.DataFrame], as_of: date) -> dict[str, pd.DataFrame]:
    """The world as it looked on ``as_of``: every row dated after it removed."""
    out = dict(t)

    def keep(name, col):
        df = t[name]
        out[name] = df[pd.to_datetime(df[col]) <= pd.Timestamp(as_of)].reset_index(drop=True)

    keep("signal_mentions", "event_date")
    keep("kings_speech_items", "speech_date")
    keep("idea_events", "event_date")
    keep("stage_sittings", "event_date")
    keep("sessions", "start_date")
    keep("sitting_days", "date")
    # idea_mentions rows have no date of their own: they go with their (removed) mention
    known = set(out["signal_mentions"]["mention_id"].astype(str))
    known |= set(out["kings_speech_items"]["item_id"].astype(str))
    im = t["idea_mentions"]
    out["idea_mentions"] = im[im["mention_id"].astype(str).isin(known)].reset_index(drop=True)
    p = t["parliaments"].copy()
    p = p[p["first_meeting_date"] <= as_of].copy()
    for col in ("dissolution_date", "election_date"):
        p[col] = [d if d is not None and d <= as_of else None for d in p[col]]
    out["parliaments"] = p.reset_index(drop=True)
    return out


AS_OFS = [
    D("2010-04-20"),  # between the 2010 dissolution and the election
    D("2016-01-01"),
    D("2016-05-18"),  # the speech day: kings_speech event, no bill yet
    D("2016-06-01"),
    D("2017-06-15"),
    D("2024-01-01"),
    D("2024-06-15"),  # between the 2024 dissolution and the election
    D("2026-10-24"),
]


# ----------------------------------------------------------------------- leakage


@pytest.mark.parametrize("as_of", AS_OFS)
def test_no_date_after_as_of(synthetic, as_of):
    df = TakeupData(synthetic).get_takeup_signals(as_of, min_positive=0)
    assert len(df)
    dates = [c for c in df.columns if c in ("signal_date", "bill_date", "data_end")]
    dates += [c for c in df.columns if c.startswith("x_")]
    for col in dates:
        assert not (df[col] > pd.Timestamp(as_of)).any(), col


@pytest.mark.parametrize("as_of", AS_OFS)
def test_future_rows_do_not_change_output(synthetic, as_of):
    """Leakage: deleting every row dated after as_of must not change anything."""
    full = TakeupData(synthetic).get_takeup_signals(as_of, min_positive=0)
    past = TakeupData(_truncate(synthetic, as_of)).get_takeup_signals(as_of, min_positive=0)
    pd.testing.assert_frame_equal(full, past)


def _with_events(t: dict[str, pd.DataFrame], events: list[tuple]) -> dict[str, pd.DataFrame]:
    t = dict(t)
    ev = pd.DataFrame(
        [
            {
                "idea_id": i,
                "event_type": e,
                "event_date": D(d),
                "title": e,
                "source_url": "u",
                "ingested_at": NOW,
            }
            for i, e, d in events
        ]
    )
    t["idea_events"] = pd.concat([t["idea_events"], ev], ignore_index=True)
    return t


@pytest.mark.parametrize("as_of", [D("2015-10-20"), *AS_OFS])
def test_future_groups_do_not_change_split_rows(synthetic, as_of):
    """Leakage: which split row gets a multi-group idea's events must not depend on groups
    whose mentions are all after as_of (only ideas that will get a bill have such groups)."""
    t = _with_events(
        synthetic,
        [
            ("nlp-00001", "kings_speech", "2015-10-15"),
            ("nlp-00001", "consultation", "2015-10-01"),
            ("nlp-00001", "green_paper", "2015-12-15"),
        ],
    )
    full = TakeupData(t).get_takeup_signals(as_of, min_positive=0)
    past = TakeupData(_truncate(t, as_of)).get_takeup_signals(as_of, min_positive=0)
    pd.testing.assert_frame_equal(full, past)


def test_kings_speech_event_visible_from_speech_for_idea_that_later_gets_a_bill(synthetic):
    """nlp-00001 gains a second group (g2, 2015-11-01) and bill 101 later; on 2015-10-20 its
    only visible group still gets the kings_speech event from the speech date (§10.9)."""
    t = _with_events(
        synthetic,
        [("nlp-00001", "kings_speech", "2015-10-15"), ("nlp-00001", "consultation", "2015-10-01")],
    )
    row = _signals(t, "2015-10-20").loc["nlp-00001:g1"]
    assert row["x_kings_speech"] == pd.Timestamp("2015-10-15")
    assert row["x_consultation"] == pd.Timestamp("2015-10-01")


def test_invisible_mentions_are_ignored(synthetic):
    df = _signals(synthetic, "2016-01-01")
    # wms-6-0 (2015-06-01, not a commitment) would otherwise be the first mention.
    assert df.loc["nlp-00001:g1", "signal_date"] == pd.Timestamp("2015-07-01")
    assert "nlp-00007:g8" in df.index
    assert "nlp-00006:g7" not in df.index  # first mention 2016-01-10 is after as_of
    later = _signals(synthetic, "2018-01-01").loc["nlp-00006:g7"]
    assert later["z_initial_firmness"] == 2 and pd.isna(later["x_firm_commitment"])
    assert pd.isna(later["x_repeat_mention"])  # the only later mention is in 2027
    assert later["x_kings_speech"] == pd.Timestamp("2016-05-18")  # its kings_speech event
    assert pd.isna(later["x_consultation_response"])  # 2019-01-01 is after as_of
    latest = _signals(synthetic, "2026-10-24").loc["nlp-00006:g7"]
    assert latest["x_consultation_response"] == pd.Timestamp("2019-01-01")


# --------------------------------------------------------------- merge and split


def test_groups_split_before_first_reading(synthetic):
    df = _signals(synthetic, "2016-01-01")
    assert {"nlp-00001:g1", "nlp-00001:g2"} <= set(df.index)
    assert "nlp-00001" not in df.index
    g1, g2 = df.loc["nlp-00001:g1"], df.loc["nlp-00001:g2"]
    assert g1["signal_date"] == pd.Timestamp("2015-07-01")
    assert g1["x_firm_commitment"] == pd.Timestamp("2015-09-01")
    assert g1["x_repeat_mention"] == pd.Timestamp("2015-09-01")
    assert g1["z_src_wms"] == 1 and g1["z_initial_firmness"] == 1
    assert pd.isna(g1["bill_date"]) and pd.isna(g1["bill_id"])
    assert g2["signal_date"] == pd.Timestamp("2015-11-01")
    assert g2["z_src_hansard"] == 1 and g2["z_src_wms"] == 0
    assert pd.isna(g2["x_repeat_mention"])
    assert (df["idea_id"] == "nlp-00001").sum() == 2


def test_groups_merge_after_first_reading(synthetic):
    df = _signals(synthetic, "2016-06-01")
    assert (
        "nlp-00001" in df.index
        and "nlp-00001:g1" not in df.index
        and "nlp-00001:g2" not in df.index
    )
    row = df.loc["nlp-00001"]
    assert row["signal_date"] == pd.Timestamp("2015-07-01")
    assert row["bill_date"] == pd.Timestamp("2016-03-01")
    assert row["bill_id"] == 101
    assert row["data_end"] == pd.Timestamp("2016-06-01")
    assert row["x_repeat_mention"] == pd.Timestamp("2015-09-01")  # same-statement row skipped


def test_first_reading_is_earliest_of_stage_6_or_1(synthetic):
    data = TakeupData(synthetic)
    assert data.first_readings[102] == pd.Timestamp("2016-06-10")  # Lords stage 1
    assert data.first_readings[101] == pd.Timestamp("2016-03-01")  # Commons stage 6
    assert 999 not in data.first_readings.index  # not a Government Bill
    df = _signals(synthetic, "2016-12-31")
    assert df.loc["nlp-00002", "bill_date"] == pd.Timestamp("2016-06-10")


def test_bill_not_visible_before_its_first_reading(synthetic):
    df = _signals(synthetic, "2016-06-01")
    assert pd.isna(df.loc["nlp-00002:g3", "bill_date"]) and pd.isna(
        df.loc["nlp-00002:g3", "bill_id"]
    )


# ------------------------------------------------------- change of government


def test_change_of_government_2024_censors(synthetic):
    df = _signals(synthetic, "2026-10-24")
    b3 = df.loc["nlp-00003"]  # merged: bill first read 2025-01-15, after the 2024 dissolution
    assert pd.isna(b3["bill_date"]) and pd.isna(b3["bill_id"])
    assert b3["data_end"] == pd.Timestamp("2024-05-30")
    b5 = df.loc["nlp-00005:g6"]  # never linked
    assert b5["data_end"] == pd.Timestamp("2024-05-30") and pd.isna(b5["bill_date"])
    b4 = df.loc["nlp-00004"]  # bill before the dissolution is kept
    assert b4["bill_date"] == pd.Timestamp("2023-06-01")
    assert b4["data_end"] == pd.Timestamp("2026-10-24")


def test_change_of_government_not_applied_before_it_happens(synthetic):
    df = _signals(synthetic, "2024-01-01")
    assert df.loc["nlp-00003:g4", "data_end"] == pd.Timestamp("2024-01-01")
    assert df.loc["nlp-00005:g6", "data_end"] == pd.Timestamp("2024-01-01")


@pytest.mark.parametrize(
    ("as_of", "ids"),
    [
        ("2024-06-15", ["nlp-00003:g4", "nlp-00005:g6", "nlp-00006:g7", "nlp-00001"]),
        ("2024-07-04", ["nlp-00003:g4", "nlp-00005:g6"]),
    ],
)
def test_change_of_government_unknown_before_its_election(synthetic, as_of, ids):
    """Between the dissolution and the result, the coming change of government is unknown."""
    df = _signals(synthetic, as_of)
    for sid in ids:
        assert df.loc[sid, "data_end"] == pd.Timestamp(as_of), sid
    early = _signals(synthetic, "2010-04-20")
    assert early.loc["nlp-00007:g8", "data_end"] == pd.Timestamp("2010-04-20")
    after = _signals(synthetic, "2024-07-05")  # the result is known the day after polling
    assert after.loc["nlp-00003:g4", "data_end"] == pd.Timestamp("2024-05-30")
    assert after.loc["nlp-00005:g6", "data_end"] == pd.Timestamp("2024-05-30")


def test_same_and_change_of_party_dissolutions_alike_before_result(synthetic):
    """No difference between a same-party (2017) and a change (2024) dissolution window."""
    for as_of in ("2017-05-20", "2024-06-15"):
        df = _signals(synthetic, as_of)
        df = df[df["signal_date"] > pd.Timestamp("2010-04-12")]  # nlp-00007: the 2010 change
        assert len(df) and (df["data_end"] == pd.Timestamp(as_of)).all(), as_of


def test_change_of_government_2010(synthetic):
    df = _signals(synthetic, "2018-01-01")
    b7 = df.loc["nlp-00007:g8"]
    assert b7["data_end"] == pd.Timestamp("2010-04-12")
    assert pd.isna(b7["x_new_parliament"])  # the 2015 election came after the change


def test_dissolutions_derived_from_parliaments(synthetic):
    t = dict(synthetic)
    p = _parliaments()
    p.loc[p["election_date"] == D("2024-07-04"), "dissolution_date"] = D("2024-05-29")
    t["parliaments"] = p
    data = TakeupData(t)
    assert data.change_of_government_dissolutions[-1] == pd.Timestamp("2024-05-29")
    assert data.changes_of_government[-1] == (
        pd.Timestamp("2024-07-04"),
        pd.Timestamp("2024-05-29"),
    )
    assert data.known_dissolutions(pd.Timestamp("2024-07-04")) == [pd.Timestamp("2010-04-12")]


def test_same_party_election_sets_new_parliament(synthetic):
    later = _signals(synthetic, "2018-01-01").loc["nlp-00006:g7"]
    assert later["x_new_parliament"] == pd.Timestamp("2017-06-21")
    assert later["data_end"] == pd.Timestamp("2018-01-01")  # not censored
    assert later["x_kings_speech"] == pd.Timestamp("2016-05-18")
    before = _signals(synthetic, "2017-06-15").loc["nlp-00006:g7"]
    assert pd.isna(before["x_new_parliament"])  # election held, State Opening not yet


# ---------------------------------------------------------------------- features


def test_kings_speech_idea_signal(synthetic):
    df = _signals(synthetic, "2016-12-31")
    a = df.loc["ks-2016-05-18-01"]
    assert a["origin"] == "kings_speech"
    assert a["signal_date"] == pd.Timestamp("2016-05-18")  # triggered on the speech date
    assert a["z_initial_firmness"] == 2
    assert a["z_src_wms"] == 0 and a["z_src_hansard"] == 0  # King's Speech = reference
    assert pd.isna(a["x_kings_speech"])  # announced at the signal itself
    assert a["x_consultation"] == pd.Timestamp("2015-12-01")
    assert a["x_white_paper"] == pd.Timestamp("2016-02-01")
    assert pd.isna(a["x_draft_bill"])  # 2017-01-01 is after as_of
    assert a["bill_date"] == pd.Timestamp("2016-06-20")


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [("2016-05-17", None), ("2016-05-18", "2016-05-18"), ("2016-12-31", "2016-05-18")],
)
def test_kings_speech_event_visible_from_speech_before_any_bill(synthetic, as_of, expected):
    """x_kings_speech comes from the idea's kings_speech event (text match, dated on the
    speech): visible from the speech date on, although nlp-00006 has no bill at all."""
    df = TakeupData(synthetic).get_takeup_signals(as_of, min_positive=0)
    row = df.set_index("signal_id").loc["nlp-00006:g7"]  # one group, no link: own events
    assert pd.isna(row["bill_id"]) and pd.isna(row["bill_date"])
    if expected is None:
        assert pd.isna(row["x_kings_speech"])
    else:
        assert row["x_kings_speech"] == pd.Timestamp(expected)
    # the matched item is not an idea of its own
    assert not df["idea_id"].str.startswith("ks-2016-05-18-02").any()


def test_kings_speech_item_as_mention_is_not_x_kings_speech(synthetic):
    """A King's Speech item among an idea's mentions (e.g. merged in through a bill) never
    sets x_kings_speech; only kings_speech events do, and only after the first mention."""
    t = dict(synthetic)
    t["idea_events"] = t["idea_events"][t["idea_events"]["event_type"] != "kings_speech"]
    t["idea_mentions"] = pd.concat(
        [
            t["idea_mentions"],
            pd.DataFrame(
                [("nlp-00006", "ks-2016-05-18-02", "g7")],
                columns=["idea_id", "mention_id", "group_id"],
            ),
        ],
        ignore_index=True,
    )
    row = _signals(t, "2018-01-01").loc["nlp-00006:g7"]
    assert pd.isna(row["x_kings_speech"]) and pd.isna(row["x_repeat_mention"])
    # an event on the trigger day (or before it) is not "after the trigger"
    ev = (
        t["idea_events"]
        .iloc[[0]]
        .assign(idea_id="nlp-00006", event_type="kings_speech", event_date=D("2016-01-10"))
    )
    t["idea_events"] = pd.concat([t["idea_events"], ev], ignore_index=True)
    assert pd.isna(_signals(t, "2018-01-01").loc["nlp-00006:g7", "x_kings_speech"])


def test_kings_speech_event_of_multi_group_idea_goes_to_its_first_group(synthetic):
    """While split, only the group of the idea's earliest mention (g1) gets idea events, a
    kings_speech event included; the other group (g2) gets them only through the merge."""
    t = _with_events(synthetic, [("nlp-00001", "kings_speech", "2015-12-01")])
    split = _signals(t, "2016-01-01")
    assert split.loc["nlp-00001:g1", "x_kings_speech"] == pd.Timestamp("2015-12-01")
    assert pd.isna(split.loc["nlp-00001:g2", "x_kings_speech"])
    assert _signals(t, "2016-06-01").loc["nlp-00001", "x_kings_speech"] == pd.Timestamp(
        "2015-12-01"
    )


def test_green_and_white_papers_are_one_feature(synthetic):
    """§10.11 has one `white_paper` feature: white or green paper, whichever came first."""
    t = _with_events(
        synthetic,
        [("nlp-00006", "green_paper", "2016-03-01"), ("nlp-00006", "white_paper", "2016-09-01")],
    )
    df = _signals(t, "2017-01-01")
    assert "x_green_paper" not in df.columns
    assert df.loc["nlp-00006:g7", "x_white_paper"] == pd.Timestamp("2016-03-01")
    assert _signals(t, "2016-06-01").loc["nlp-00006:g7", "x_white_paper"] == pd.Timestamp(
        "2016-03-01"
    )
    assert pd.isna(_signals(t, "2016-02-01").loc["nlp-00006:g7", "x_white_paper"])
    fixed = TakeupData(t).get_takeup_signals("2017-01-01", features=["x_white_paper"])
    assert list(fixed.columns[-1:]) == ["x_white_paper"]


def test_election_and_session_features(synthetic):
    row = _signals(synthetic, "2016-01-01").loc["nlp-00001:g1"]  # 2015-07-01
    assert row["z_years_since_election"] == pytest.approx(55 / 365.25)  # since 2015-05-07
    assert row["z_months_into_session"] == pytest.approx(35 / 30.4375)  # since 2015-05-27


def test_departments_and_sparse_feature_drop(synthetic):
    data = TakeupData(synthetic)
    df = data.get_takeup_signals("2026-10-24", min_positive=0)
    assert {"z_dept_education", "z_dept_home_office", "z_dept_ministry_of_justice"} <= set(
        df.columns
    )
    assert df.set_index("signal_id").loc["nlp-00001", "z_dept_education"] == 1
    sparse = data.get_takeup_signals("2026-10-24")  # default: 15 positive cases
    assert "z_src_wms" in data.dropped_features and "z_src_wms" not in sparse
    assert "z_dept_education" in data.dropped_features
    assert {"z_initial_firmness", "z_years_since_election"} <= set(sparse.columns)


def test_fixed_feature_list(synthetic):
    feats = ["z_src_wms", "z_dept_treasury", "x_draft_bill", "x_manifesto"]
    df = TakeupData(synthetic).get_takeup_signals("2026-10-24", features=feats)
    assert [c for c in df.columns if c[:2] in ("z_", "x_")] == feats
    assert (df["z_dept_treasury"] == 0).all() and df["x_manifesto"].isna().all()
    assert pd.api.types.is_datetime64_any_dtype(df["x_draft_bill"])


def test_department_slug():
    assert department_slug("Department for Education") == "education"
    assert department_slug("Department of Health and Social Care") == "health_and_social_care"
    assert department_slug("HM Treasury") == "hm_treasury"
    assert department_slug(None) is None and department_slug("  ") is None


def test_empty_as_of(synthetic):
    df = TakeupData(synthetic).get_takeup_signals("2000-01-01")
    assert df.empty and "signal_id" in df.columns


# ---------------------------------------------------------------------- calendar


def test_calendar_helpers(synthetic, monkeypatch):
    data = TakeupData(synthetic)
    dates = data.sitting_dates(D("2016-01-01"))
    assert dates.max() <= pd.Timestamp("2016-01-01").to_datetime64()
    monkeypatch.setattr(calendar, "parliaments", _parliaments)
    expected = calendar.week_sitting_probability(synthetic["sitting_days"], D("2020-01-01"))
    pd.testing.assert_series_equal(data.week_probability(D("2020-01-01")), expected)


# ------------------------------------------------------------- tables and engines


TAKEUP_TABLES = {
    "source_documents",
    "member_government_posts",
    "kings_speech_items",
    "signal_mentions",
    "ideas",
    "idea_mentions",
    "idea_events",
    "idea_bill_links",
    "takeup_runs",
    "takeup_forecasts",
}


def test_tables_on_own_metadata():
    import westminster_weather.models  # noqa: F401  (registers the app tables on Base)

    assert tables.takeup_metadata is not app_db.Base.metadata
    assert set(tables.takeup_metadata.tables) == TAKEUP_TABLES
    assert not TAKEUP_TABLES & set(app_db.Base.metadata.tables)
    for name in TAKEUP_TABLES:
        assert app_db.Base.metadata.tables.get(name) is None
        assert tables.TABLES[name].metadata is tables.takeup_metadata
    assert set(tables.SOURCE_TABLES) == {tables.source_documents, tables.member_government_posts}
    assert not set(tables.SOURCE_TABLES) & set(tables.DATA_TABLES)
    # no database-level FKs out of the take-up metadata (bills lives in production)
    for table in tables.takeup_metadata.tables.values():
        for fk in table.foreign_keys:
            assert fk.column.table.metadata is tables.takeup_metadata, fk
    assert not tables.idea_bill_links.c.bill_id.foreign_keys
    assert not tables.kings_speech_items.c.bill_id.foreign_keys
    assert tables.kings_speech_items.c.bill_id.nullable
    assert [c.name for c in tables.idea_mentions.primary_key] == ["idea_id", "mention_id"]


def test_source_tables_and_mentions_columns():
    sd = tables.source_documents
    assert [c.name for c in sd.primary_key] == ["doc_id"]
    assert {"source", "source_id", "event_date", "member_id", "house", "title"} <= set(sd.c.keys())
    assert {"department", "text", "matched_phrases", "fetched_at"} <= set(sd.c.keys())
    posts = tables.member_government_posts
    assert [c.name for c in posts.primary_key] == ["member_id", "post_name", "start_date"]
    assert posts.c.end_date.nullable
    sm = tables.signal_mentions
    assert "source_id" not in sm.c
    assert [fk.target_fullname for fk in sm.c.doc_id.foreign_keys] == ["source_documents.doc_id"]
    assert not sm.c.doc_id.nullable
    assert "origin" in tables.ideas.c and "tier" not in tables.ideas.c


def test_takeup_forecasts_key_and_columns():
    f = tables.takeup_forecasts
    # several split signals of one idea can be forecast at the same as_of
    assert [c.name for c in f.primary_key] == ["run_id", "signal_id", "as_of"]
    assert not f.c.idea_id.primary_key and not f.c.idea_id.nullable
    assert [fk.target_fullname for fk in f.c.idea_id.foreign_keys] == ["ideas.idea_id"]
    assert [fk.target_fullname for fk in f.c.run_id.foreign_keys] == ["takeup_runs.run_id"]
    outputs = ["pi", "pi_lo", "pi_hi", "verdict", "weeks_elapsed", "pi_now", "p26", "p52"]
    quantiles = ["t_10", "t_50", "t_90"]
    assert f.c.keys() == ["run_id", "signal_id", "as_of", "idea_id", *outputs, *quantiles] + [
        "created_at"
    ]
    assert all(f.c[q].nullable for q in quantiles)
    assert not any(f.c[c].nullable for c in outputs)


def _no_takeup_url(monkeypatch):
    monkeypatch.setattr(takeup_db.settings, "database_url_takeup", None)
    monkeypatch.setattr(takeup_db.settings, "database_url_takeup_unpooled", None)


def test_engines_raise_without_takeup_url(monkeypatch):
    _no_takeup_url(monkeypatch)
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured, match="DATABASE_URL_TAKEUP"):
        takeup_db.get_takeup_engine()
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured):
        takeup_db.get_takeup_direct_engine()
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured):
        takeup_db.migrate()
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured):
        takeup_db.upsert("ideas", [{"idea_id": "nlp-00001"}])


def test_takeup_settings_read_from_env_and_never_database_url(monkeypatch):
    """DATABASE_URL_TAKEUP missing: no take-up URL at all, even though DATABASE_URL (and the
    optional DATABASE_URL_TAKEUP_UNPOOLED, or the old TAKEUP_DATABASE_URL name) is set."""
    from westminster_weather.config import Settings

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@production.invalid/db")
    monkeypatch.setenv("DATABASE_URL_TAKEUP_UNPOOLED", "postgresql://u:p@direct.invalid/db")
    monkeypatch.setenv("TAKEUP_DATABASE_URL", "postgresql://u:p@old-name.invalid/db")
    monkeypatch.delenv("DATABASE_URL_TAKEUP", raising=False)
    missing = Settings(_env_file=None)
    assert missing.database_url_takeup is None
    assert missing.sqlalchemy_takeup_pooled_url is None
    assert missing.sqlalchemy_takeup_direct_url is None
    monkeypatch.setattr(takeup_db, "settings", missing)
    for make in (takeup_db.get_takeup_engine, takeup_db.get_takeup_direct_engine):
        with pytest.raises(
            takeup_db.TakeupDatabaseNotConfigured, match="DATABASE_URL_TAKEUP is not set"
        ):
            make()

    monkeypatch.setenv("DATABASE_URL_TAKEUP", "postgres://u:p@takeup-dev.invalid/db")
    found = Settings(_env_file=None)
    assert found.sqlalchemy_takeup_pooled_url == "postgresql+psycopg://u:p@takeup-dev.invalid/db"
    assert found.sqlalchemy_takeup_direct_url == "postgresql+psycopg://u:p@direct.invalid/db"
    monkeypatch.delenv("DATABASE_URL_TAKEUP_UNPOOLED")
    pooled_only = Settings(_env_file=None)  # the unpooled URL is optional
    assert pooled_only.sqlalchemy_takeup_direct_url == pooled_only.sqlalchemy_takeup_pooled_url


def test_engines_refuse_production_url(monkeypatch):
    _no_takeup_url(monkeypatch)
    monkeypatch.setattr(takeup_db.settings, "database_url_takeup", app_db.settings.database_url)
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured):
        takeup_db.get_takeup_engine()


def test_engines_use_takeup_url(monkeypatch):
    url = "postgresql://u:p@takeup-dev.invalid/db"
    _no_takeup_url(monkeypatch)
    monkeypatch.setattr(takeup_db.settings, "database_url_takeup", url)
    engine = takeup_db.get_takeup_engine()  # no connection is made
    assert engine.url.host == "takeup-dev.invalid"
    assert engine.url.drivername == "postgresql+psycopg"
    assert takeup_db.get_takeup_direct_engine().url.host == "takeup-dev.invalid"
    direct = "postgresql://u:p@takeup-dev-direct.invalid/db"
    monkeypatch.setattr(takeup_db.settings, "database_url_takeup_unpooled", direct)
    assert takeup_db.get_takeup_direct_engine().url.host == "takeup-dev-direct.invalid"
    assert takeup_db.get_takeup_engine().url.host == "takeup-dev.invalid"


# ------------------------------------------------------ bills before the signal


def _first_reading(t, bill_id, day):
    t = dict(t)
    ss = t["stage_sittings"].copy()
    ss.loc[ss["bill_id"] == bill_id, "event_date"] = D(day)
    t["stage_sittings"] = ss
    return t


def test_bill_up_to_7_days_before_signal_is_week_one(synthetic):
    """A first reading 1-7 days before the commitment is a take-up in week 1, also for
    prepare(), which counts only bills in week >= 1."""
    from westminster_weather.takeup.model import prepare

    t = _first_reading(synthetic, 104, "2022-01-28")  # nlp-00004's mention is on 2022-02-01
    data = TakeupData(t)
    df = data.get_takeup_signals("2023-01-01", min_positive=0)
    row = df.set_index("signal_id").loc["nlp-00004"]
    assert row["signal_date"] == row["bill_date"] == pd.Timestamp("2022-01-28")
    assert row["bill_id"] == 104 and data.n_signal_date_moved == 1
    assert row["z_initial_firmness"] == 1  # from the mention itself
    d = prepare(df, data.sitting_dates("2023-01-01"), data_end="2023-01-01")
    k = list(d["ids"]).index("nlp-00004")
    assert d["delta"][k] == 1 and d["t"][k] == 1
    assert d["n_bill_before_signal"] == 0


def test_bill_long_before_signal_is_dropped(synthetic):
    """A commitment to a bill that already exists is not a signal (not a silent idea)."""
    t = _first_reading(synthetic, 104, "2021-06-01")
    data = TakeupData(t)
    df = data.get_takeup_signals("2023-01-01", min_positive=0)
    assert "nlp-00004" not in set(df["signal_id"]) and "nlp-00004:g5" not in set(df["signal_id"])
    assert data.dropped_signals == {"nlp-00004": 104}


def _with_ks_item(t, item_id, speech, carried, bill_id=None):
    t = dict(t)
    item = {
        "item_id": item_id,
        "speech_date": D(speech),
        "announced_name": "Carried Bill",
        "is_draft": False,
        "is_carried_over": carried,
        "department": None,
        "bill_id": bill_id,
        "source_url": "u",
        "ingested_at": NOW,
    }
    idea = item_id  # a King's Speech item that is its own idea
    t["kings_speech_items"] = pd.concat(
        [t["kings_speech_items"], pd.DataFrame([item]).astype({"bill_id": "Int64"})],
        ignore_index=True,
    )
    t["idea_mentions"] = pd.concat(
        [
            t["idea_mentions"],
            pd.DataFrame([(idea, item_id, item_id)], columns=["idea_id", "mention_id", "group_id"]),
        ],
        ignore_index=True,
    )
    if bill_id is not None:
        link = t["idea_bill_links"].iloc[[0]].assign(idea_id=idea, bill_id=bill_id)
        t["idea_bill_links"] = pd.concat([t["idea_bill_links"], link], ignore_index=True)
    return t


def test_carried_over_kings_speech_items_are_not_signals(synthetic):
    # carried over, linked to a bill first read in an earlier session
    t = _with_ks_item(synthetic, "ks-2017-06-21-01", "2017-06-21", True, bill_id=102)
    # carried over, not linked: excluded by the flag alone
    t = _with_ks_item(t, "ks-2017-06-21-02", "2017-06-21", True)
    data = TakeupData(t)
    df = data.get_takeup_signals("2020-01-01", min_positive=0)
    assert not any(i.startswith("ks-2017-06-21") for i in df["idea_id"])
    assert data.carried_over_items == ["ks-2017-06-21-01", "ks-2017-06-21-02"]
    assert data.n_carried_over == 2
    assert "ks-2017-06-21-01" not in data.dropped_signals  # excluded, not dropped
    # counted only once the speech has happened, and reset on every call
    data.get_takeup_signals("2017-06-20", min_positive=0)
    assert data.n_carried_over == 0
    # the same item announced as new is a signal
    t = _with_ks_item(synthetic, "ks-2017-06-21-02", "2017-06-21", False)
    data = TakeupData(t)
    df = data.get_takeup_signals("2020-01-01", min_positive=0).set_index("signal_id")
    assert "ks-2017-06-21-02:ks-2017-06-21-02" in df.index and data.n_carried_over == 0
    row = df.loc["ks-2017-06-21-02:ks-2017-06-21-02"]
    assert row["origin"] == "kings_speech" and row["z_initial_firmness"] == 2


# -------------------------------------------------- split rows: no idea-level data


def _own_idea(t, group_id, idea_id):
    """The same world, but with group ``group_id`` as an idea of its own with no link."""
    t = dict(t)
    im = t["idea_mentions"].copy()
    im.loc[im["group_id"] == group_id, "idea_id"] = idea_id
    t["idea_mentions"] = im
    return t


def _comparable(row: pd.Series) -> pd.Series:
    return row.drop(["idea_id"])


def test_split_group_gets_nothing_from_its_later_idea(synthetic):
    """Before the first reading a group's row must be the same whether or not it will be
    merged (through the bill) into a King's Speech idea with events, origin and a title."""
    t = dict(synthetic)
    extra = _mention("wms-20-0", "2016-04-01", firmness=1)
    t["signal_mentions"] = pd.concat(
        [t["signal_mentions"], pd.DataFrame([extra])], ignore_index=True
    )
    im = pd.DataFrame(
        [("ks-2016-05-18-01", "wms-20-0", "g9")], columns=["idea_id", "mention_id", "group_id"]
    )
    t["idea_mentions"] = pd.concat([t["idea_mentions"], im], ignore_index=True)
    as_of = "2016-06-01"  # bill 105 is first read on 2016-06-20
    merged_later = _signals(t, as_of)
    alone = _signals(_own_idea(t, "g9", "nlp-00099"), as_of)
    row = merged_later.loc["ks-2016-05-18-01:g9"]
    assert row["origin"] == "nlp" and row["title"] == "topic"
    assert pd.isna(row["x_consultation"]) and pd.isna(row["x_white_paper"])
    pd.testing.assert_series_equal(
        _comparable(row), _comparable(alone.loc["nlp-00099:g9"]), check_names=False
    )
    # the group holding the King's Speech item keeps the item's own events
    anchor = merged_later.loc["ks-2016-05-18-01:ks-2016-05-18-01"]
    assert anchor["origin"] == "kings_speech" and anchor["title"] == "Prisons Bill"
    assert anchor["x_consultation"] == pd.Timestamp("2015-12-01")
    # after the first reading the idea is one signal with idea-level origin, title and events
    after = _signals(t, "2016-07-01").loc["ks-2016-05-18-01"]
    assert after["origin"] == "kings_speech" and after["title"] == "topic ks-2016-05-18-01"
    assert after["signal_date"] == pd.Timestamp("2016-04-01")
    assert after["x_white_paper"] == pd.Timestamp("2016-02-01")
    # the King's Speech item merged in through the bill is a mention, not an x feature
    assert pd.isna(after["x_kings_speech"]) and pd.isna(after["x_repeat_mention"])


def test_multi_group_nlp_idea_events_go_to_its_first_group(synthetic):
    t = _with_events(synthetic, [("nlp-00001", "consultation", "2015-10-01")])
    split = _signals(t, "2016-01-01")
    assert split.loc["nlp-00001:g1", "x_consultation"] == pd.Timestamp("2015-10-01")
    assert pd.isna(split.loc["nlp-00001:g2", "x_consultation"])
    merged = _signals(t, "2016-06-01").loc["nlp-00001"]
    assert merged["x_consultation"] == pd.Timestamp("2015-10-01")


def test_kings_speech_idea_events_wait_for_its_own_item(synthetic):
    """Before the speech, a group merged into a King's Speech idea through the bill is the
    idea's only visible group, but it still gets none of the idea's events."""
    t = dict(synthetic)
    extra = _mention("wms-20-0", "2016-04-01", firmness=1)
    t["signal_mentions"] = pd.concat(
        [t["signal_mentions"], pd.DataFrame([extra])], ignore_index=True
    )
    im = pd.DataFrame(
        [("ks-2016-05-18-01", "wms-20-0", "g9")], columns=["idea_id", "mention_id", "group_id"]
    )
    t["idea_mentions"] = pd.concat([t["idea_mentions"], im], ignore_index=True)
    row = _signals(t, "2016-05-01").loc["ks-2016-05-18-01:g9"]
    assert pd.isna(row["x_consultation"]) and pd.isna(row["x_white_paper"])


# ---------------------------------------------------------- same-day mentions


def test_firmness_on_signal_day_uses_every_mention(synthetic):
    t = dict(synthetic)
    extra = _mention("han-77-0", "2022-02-01", source="hansard_spoken", firmness=2)
    t["signal_mentions"] = pd.concat(
        [t["signal_mentions"], pd.DataFrame([extra])], ignore_index=True
    )
    im = pd.DataFrame(
        [("nlp-00004", "han-77-0", "g5")], columns=["idea_id", "mention_id", "group_id"]
    )
    t["idea_mentions"] = pd.concat([t["idea_mentions"], im], ignore_index=True)
    row = _signals(t, "2023-01-01").loc["nlp-00004:g5"]
    assert row["z_initial_firmness"] == 2
    assert pd.isna(row["x_firm_commitment"])  # already firm at the signal
    assert pd.isna(row["x_repeat_mention"])  # a same-day document is not a later repeat
    assert row["z_src_wms"] == 1  # the first source is still the statement


# ------------------------------------------------------- read-only reference load


def test_load_reference_tables_is_read_only(tmp_path, monkeypatch):
    """Plain SELECTs: no seeding (create_all / merges) on the production database."""
    import sqlalchemy as sa

    import westminster_weather.models  # noqa: F401  (registers the app tables on Base)

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'prod.db'}")
    app_db.Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            sa.text("INSERT INTO sitting_days (date, house) VALUES ('2020-01-07', 'Commons')")
        )

    def _no_seed():
        raise AssertionError("load_reference_tables must not seed production")

    monkeypatch.setattr(app_db, "ensure_reference_tables_seeded_sync", _no_seed)
    monkeypatch.setattr(app_db, "get_sync_pooled_engine", lambda: engine)
    out = takeup_db.load_reference_tables()
    assert set(out) == {"bills", "stage_sittings", "sessions", "parliaments", "sitting_days"}
    assert len(out["sitting_days"]) == 1 and out["bills"].empty
    assert {"bill_id", "short_title"} <= set(out["bills"].columns)
    assert {"bill_id", "api_stage_id", "event_date"} <= set(out["stage_sittings"].columns)
    # empty parliaments: the defaults in memory, nothing written
    assert len(out["parliaments"]) == len(app_db.DEFAULT_PARLIAMENTS)
    with sa.create_engine(f"sqlite:///{tmp_path / 'prod.db'}").connect() as conn:
        assert conn.execute(sa.text("SELECT COUNT(*) FROM parliaments")).scalar_one() == 0
