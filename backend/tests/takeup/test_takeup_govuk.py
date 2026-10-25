"""Offline tests for GOV.UK events (takeup.govuk_events): parsing, classification, matching,
no bill information, idempotent upsert on in-memory SQLite."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select

from westminster_weather.takeup import govuk_events as ge
from westminster_weather.takeup.tables import idea_events, ideas, takeup_metadata


def _rec(title, doc_type, ts, start=None, link=None, desc="", orgs=()):
    r = {
        "title": title,
        "link": link or "/government/x/" + title.lower().replace(" ", "-"),
        "public_timestamp": ts,
        "content_store_document_type": doc_type,
        "organisations": [{"title": o} for o in orgs],
        "description": desc,
    }
    if start:
        r["start_date"] = start
    return r


# --- classification --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record, expected",
    [
        (
            _rec("Online Harms White Paper", "policy_paper", "2019-04-08T10:00:00Z"),
            [("white_paper", date(2019, 4, 8))],
        ),
        (
            _rec("Schools that work for everyone: green paper", "guidance", "2016-09-12T09:00Z"),
            [("green_paper", date(2016, 9, 12))],
        ),
        (
            _rec("Draft Online Safety Bill", "policy_paper", "2021-05-12T12:00:00Z"),
            [("draft_bill", date(2021, 5, 12))],
        ),
        (
            _rec("CMA response to Industrial Strategy Green Paper", "policy_paper", "2017-04-01"),
            [],
        ),
        (_rec("Annual report 2019", "policy_paper", "2019-07-01T00:00:00Z"), []),
        (
            _rec("Tenant fees", "open_consultation", "2016-11-03T00:00:00Z", "2016-11-02T09:00"),
            [("consultation", date(2016, 11, 2))],
        ),
        (
            _rec("Tenant fees", "closed_consultation", "2017-01-10T00:00:00Z"),
            [("consultation", date(2017, 1, 10))],
        ),
        (
            _rec("Tenant fees", "consultation_outcome", "2017-04-20T00:00:00Z", "2016-11-02T09"),
            [("consultation", date(2016, 11, 2)), ("consultation_response", date(2017, 4, 20))],
        ),
        (
            # outcome without an opening date: only the response is known
            _rec("Tenant fees", "consultation_outcome", "2017-04-20T00:00:00Z"),
            [("consultation_response", date(2017, 4, 20))],
        ),
        (
            # a consultation on a green paper is the green paper
            _rec(
                "Building our industrial strategy: green paper",
                "consultation_outcome",
                "2017-11-27T00:00Z",
                "2017-01-23T00:00",
            ),
            [("green_paper", date(2017, 1, 23)), ("consultation_response", date(2017, 11, 27))],
        ),
        (_rec("Old white paper", "policy_paper", "2004-01-01T00:00:00Z"), []),
    ],
)
def test_classify(record, expected):
    assert ge.classify(record) == expected


def test_parse_results_builds_urls_orgs_and_dedupes():
    recs = [
        _rec(
            "Draft Online Safety Bill",
            "policy_paper",
            "2021-05-12T12:00:00Z",
            link="/government/publications/draft-online-safety-bill",
            desc="The draft Bill",
            orgs=("Department for Digital, Culture, Media & Sport",),
        ),
        # same document returned by a second query
        _rec(
            "Draft Online Safety Bill",
            "policy_paper",
            "2021-05-12T12:00:00Z",
            link="/government/publications/draft-online-safety-bill",
        ),
        {
            "title": "no link",
            "public_timestamp": "2020-01-01",
            "content_store_document_type": "policy_paper",
        },
    ]
    df = ge.parse_results(recs)
    assert len(df) == 1
    row = df.iloc[0]
    assert row.event_type == "draft_bill"
    assert row.event_date == date(2021, 5, 12)
    assert row.source_url == "https://www.gov.uk/government/publications/draft-online-safety-bill"
    assert row.organisations == "Department for Digital, Culture, Media & Sport"
    assert row.description == "The draft Bill"
    assert set(df.event_type) <= set(ge.EVENT_TYPES)


def test_fetch_reads_paginated_cache_offline(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(ge, "PAGE_SIZE", 2)
    monkeypatch.setattr(
        ge, "QUERIES", [ge.Query("consultations", {"filter_content_store_document_type": []})]
    )
    pages = [
        {"results": [_rec("A", "closed_consultation", "2015-01-01")] * 2, "total": 3},
        {"results": [_rec("B", "closed_consultation", "2015-02-01")], "total": 3},
    ]
    for i, page in enumerate(pages):
        p = tmp_path / "consultations" / f"{i * 2:06d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(page))
    results = ge.fetch_govuk(cache=tmp_path, offline=True)
    assert [r["title"] for r in results] == ["A", "A", "B"]
    assert "first record keys" in capsys.readouterr().out


# --- matching fixture ------------------------------------------------------------------------

IDEAS = pd.DataFrame(
    [
        {
            "idea_id": "nlp-00001",
            "topic": "legislation to tackle online harms",
            "trigger_date": date(2018, 5, 20),
            "department": "Department for Digital, Culture, Media and Sport",
        },
        {
            "idea_id": "nlp-00002",
            "topic": "ban letting agents' fees charged to tenants",
            "trigger_date": date(2016, 11, 23),
            "department": "Ministry of Housing, Communities and Local Government",
        },
        {
            "idea_id": "nlp-00003",
            "topic": "introduce a new deposit return scheme for drinks containers",
            "trigger_date": date(2018, 3, 28),
            "department": None,
        },
        {
            "idea_id": "nlp-00004",
            "topic": "We will introduce the Bill",
            "trigger_date": date(2019, 1, 10),
            "department": None,
        },
    ]
)
MENTIONS = pd.DataFrame(
    [
        {"idea_id": "nlp-00001", "title": "Internet Safety Strategy"},
        {"idea_id": "nlp-00001", "title": "Online Harms"},
        {"idea_id": "nlp-00002", "title": "Letting Agents' Fees"},
        {"idea_id": "nlp-00003", "title": "Plastic Bottles: Deposit Return Scheme"},
        {"idea_id": "nlp-00004", "title": "Business of the House"},
    ]
)
EVENTS = pd.DataFrame(
    [
        # true matches
        {
            "event_type": "white_paper",
            "event_date": date(2019, 4, 8),
            "title": "Online Harms White Paper",
            "description": "Proposals for tackling online harms and making the internet safer.",
            "source_url": "u1",
            "organisations": "Department for Digital, Culture, Media & Sport; Home Office",
        },
        {
            "event_type": "consultation",
            "event_date": date(2017, 4, 7),
            "title": "Banning letting agent fees paid by tenants",
            "description": "Seeking views on banning letting agent fees paid by tenants.",
            "source_url": "u2",
            "organisations": "Ministry of Housing, Communities and Local Government",
        },
        {
            "event_type": "consultation",
            "event_date": date(2019, 2, 18),
            "title": "Introducing a deposit return scheme for drinks containers",
            "description": "Consultation on a deposit return scheme for drinks containers "
            "in England, Wales and Northern Ireland.",
            "source_url": "u3",
            "organisations": "Department for Environment, Food & Rural Affairs",
        },
        # decoys: share generic or near words only
        {
            "event_type": "consultation",
            "event_date": date(2018, 6, 1),
            "title": "Harbour safety: draft guidance consultation",
            "description": "Proposed changes to port marine safety guidance.",
            "source_url": "d1",
            "organisations": "Department for Transport",
        },
        {
            "event_type": "white_paper",
            "event_date": date(2018, 7, 12),
            "title": "The future relationship between the UK and the EU",
            "description": "White paper setting out the government's proposals.",
            "source_url": "d2",
            "organisations": "Department for Exiting the European Union",
        },
        {
            "event_type": "draft_bill",
            "event_date": date(2019, 3, 1),
            "title": "Draft Domestic Abuse Bill",
            "description": "The government will introduce a bill to tackle domestic abuse.",
            "source_url": "d3",
            "organisations": "Home Office",
        },
        # true topic but out of window (more than 2 years before / 5 years after the trigger)
        {
            "event_type": "consultation",
            "event_date": date(2009, 1, 1),
            "title": "Online harms to children",
            "description": "Online harms consultation",
            "source_url": "o1",
            "organisations": "",
        },
        {
            "event_type": "consultation",
            "event_date": date(2024, 6, 1),
            "title": "Letting agent fees paid by tenants: review",
            "description": "Letting agents fees tenants",
            "source_url": "o2",
            "organisations": "",
        },
    ]
)


def _pairs(df):
    return set(zip(df.idea_id, df.source_url, strict=True))


def test_fixture_threshold_separates_true_matches_from_decoys():
    texts = ge.idea_texts(IDEAS, MENTIONS)
    w, c = ge.score_matrix(
        [texts[i] for i in IDEAS.idea_id], (EVENTS.title + " . " + EVENTS.description).tolist()
    )
    score = ge.WORD_WEIGHT * w + ge.CHAR_WEIGHT * c
    true = {(0, 0), (1, 1), (2, 2)}
    in_window_decoys = [
        score[i, j] for i in range(len(IDEAS)) for j in range(6) if (i, j) not in true
    ]
    assert min(score[i, j] for i, j in true) > ge.THRESHOLD
    assert max(in_window_decoys) < ge.THRESHOLD


def test_match_events_true_match_decoy_and_window():
    out = ge.match_events(IDEAS, MENTIONS, EVENTS)
    assert _pairs(out) == {("nlp-00001", "u1"), ("nlp-00002", "u2"), ("nlp-00003", "u3")}
    assert set(out.columns) >= {"idea_id", "event_type", "event_date", "title", "source_url"}
    assert not out.duplicated(["idea_id", "event_type", "event_date"]).any()


def test_window_bounds():
    ideas_ = IDEAS.iloc[[1]]
    near = EVENTS.iloc[[1]].copy()
    for d, keep in [
        (date(2014, 11, 24), True),  # 730 days before
        (date(2014, 11, 22), False),
        (date(2021, 11, 22), True),  # ~5 years after
        (date(2021, 11, 25), False),
    ]:
        near["event_date"] = d
        assert (len(ge.match_events(ideas_, MENTIONS, near)) == 1) is keep, d


def test_top_k_per_idea():
    many = pd.concat([EVENTS.iloc[[0]]] * 6, ignore_index=True)
    many["event_date"] = [date(2019, 1, d) for d in range(1, 7)]
    out = ge.match_events(IDEAS, MENTIONS, many)
    assert len(out) == ge.TOP_K


def test_no_bill_information_used(monkeypatch):
    # match_events takes no bill input; extra bill columns on ideas change nothing.
    with_bills = IDEAS.assign(bill_id=[1, 2, 3, 4], bill_title=["Online Safety Bill"] * 4)
    a = ge.match_events(IDEAS, MENTIONS, EVENTS)
    b = ge.match_events(with_bills, MENTIONS, EVENTS)
    pd.testing.assert_frame_equal(a, b)

    # run_govuk_events never reads idea_bill_links or production reference tables.
    from westminster_weather.takeup import db

    def fail(*a, **k):
        raise AssertionError("bill / reference data must not be read")

    links = pd.DataFrame({"idea_id": ["nlp-00004"], "bill_id": [99]})

    class Tables(dict):
        def __getitem__(self, key):
            if key == "idea_bill_links":
                fail()
            return super().__getitem__(key)

    tables = Tables(
        ideas=IDEAS,
        idea_mentions=MENTIONS.assign(mention_id=[f"m{i}" for i in range(len(MENTIONS))])[
            ["idea_id", "mention_id"]
        ].assign(group_id="g"),
        signal_mentions=MENTIONS.assign(
            mention_id=[f"m{i}" for i in range(len(MENTIONS))], event_date=date(2018, 1, 1)
        )[["mention_id", "title", "event_date"]],
        idea_bill_links=links,
    )
    monkeypatch.setattr(db, "load_takeup_tables", lambda: tables)
    monkeypatch.setattr(db, "load_reference_tables", fail)
    monkeypatch.setattr(db, "get_takeup_engine", fail)
    monkeypatch.setattr(ge, "fetch_govuk", lambda **k: [])
    monkeypatch.setattr(ge, "parse_results", lambda r: EVENTS)
    res = ge.run_govuk_events(dry_run=True)
    assert res["written"] == 0
    assert res["ideas_with_events"] == 3


# --- loading ---------------------------------------------------------------------------------


def test_load_idea_events_is_idempotent_on_sqlite():
    engine = create_engine("sqlite://")
    takeup_metadata.create_all(engine)
    now = pd.Timestamp("2026-10-24", tz="UTC")
    with engine.begin() as conn:
        conn.execute(
            ideas.insert(),
            [
                {
                    "idea_id": i,
                    "origin": "nlp",
                    "topic": t,
                    "trigger_date": d,
                    "first_source": "wms",
                    "initial_firmness": 2,
                    "n_mentions": 1,
                    "ingested_at": now,
                }
                for i, t, d in IDEAS[["idea_id", "topic", "trigger_date"]].itertuples(index=False)
            ],
        )
    rows = ge.match_events(IDEAS, MENTIONS, EVENTS)
    assert ge.load_idea_events(engine, rows) == 3
    assert ge.load_idea_events(engine, rows) == 3
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(idea_events)).scalar_one() == 3
        got = conn.execute(select(idea_events.c.idea_id, idea_events.c.source_url)).all()
    assert set(got) == _pairs(rows)
    assert ge.load_idea_events(engine, rows.iloc[0:0]) == 0
