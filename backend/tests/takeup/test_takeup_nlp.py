"""Offline tests for the take-up NLP pipeline (Tier B2-B4, B6): ``takeup/nlp.py``.

The synthetic self-test is ported from the prototype ``nlp_signals.py test``: speeches with
known answers (which sentences are commitments, their firmness, which idea and which bill).
Everything runs in memory or on in-memory SQLite; no test touches a real database."""

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score
from sqlalchemy import create_engine, func, insert, select

from westminster_weather import db as app_db
from westminster_weather.takeup import db as takeup_db
from westminster_weather.takeup import nlp
from westminster_weather.takeup.signals import TakeupData
from westminster_weather.takeup.tables import (
    idea_events,
    ideas,
    member_government_posts,
    signal_mentions,
    source_documents,
    takeup_metadata,
)

NOW = datetime(2026, 10, 24, tzinfo=UTC)
PRODUCTION = "postgresql://u:p@production.invalid/db"


@pytest.fixture(autouse=True)
def _offline_settings(monkeypatch):
    """No real database URL is ever visible to these tests."""
    monkeypatch.setattr(takeup_db.settings, "database_url", PRODUCTION)
    monkeypatch.setattr(takeup_db.settings, "database_url_unpooled", None)
    monkeypatch.setattr(takeup_db.settings, "database_url_takeup", None)
    monkeypatch.setattr(takeup_db.settings, "database_url_takeup_unpooled", None)


# ------------------------------------------------------------- synthetic world ----
TOPICS = [
    ("ban no-fault evictions", ["abolish no-fault evictions", "end no-fault evictions for renters"],
     "Renters' Rights Bill",
     "A Bill to abolish no-fault evictions and assured shorthold tenancies for renters"),
    ("regulate water company bonuses",
     ["stop bonuses for water company bosses", "water company bonuses and sewage"],
     "Water (Special Measures) Bill",
     "A Bill to make provision about water company bonuses and sewage pollution"),
    ("raise the smoking age every year",
     ["create a smoke-free generation", "phased ban on tobacco sales smoking age"],
     "Tobacco and Vapes Bill",
     "A Bill to prohibit the sale of tobacco to anyone born after 2009 and raise the smoking age"),
    ("set up a football regulator",
     ["independent regulator for English football", "football club regulation"],
     "Football Governance Bill",
     "A Bill to establish an independent football regulator for football clubs"),
    ("protect online children from harmful content",
     ["online safety duties for platforms", "harmful content online children"],
     "Online Safety Bill",
     "A Bill to impose duties on online platforms about harmful content and children"),
    ("nationalise passenger railways",
     ["bring passenger rail services into public ownership", "public ownership of railways"],
     "Passenger Railway Services (Public Ownership) Bill",
     "A Bill to bring passenger railway services into public ownership"),
    ("create a national energy company",
     ["publicly owned clean energy company", "great british energy company"],
     "Great British Energy Bill",
     "A Bill to establish Great British Energy, a publicly owned clean energy company"),
    ("strengthen employment rights for zero hours workers",
     ["ban exploitative zero hours contracts", "employment rights day one"],
     "Employment Rights Bill",
     "A Bill to strengthen employment rights and restrict zero hours contracts"),
    ("reform leasehold and ground rents",
     ["abolish ground rents for leaseholders", "leasehold reform for homeowners"], None, None),
    ("ban conversion practices",
     ["outlaw conversion therapy practices", "ending conversion practices"], None, None),
    ("introduce digital identity cards",
     ["national digital id scheme", "digital identity verification for citizens"], None, None),
    ("regulate artificial intelligence developers",
     ["binding rules for frontier ai models", "ai safety legislation for developers"], None, None),
]  # fmt: skip
TEMPLATES = {
    2: [
        "We will legislate to {t}.",
        "The Government will bring forward legislation to {t} this session.",
        "We are introducing legislation to {t}.",
        "A bill will be introduced to {t}.",
    ],
    1: [
        "We intend to legislate to {t}.",
        "The Government is committed to legislating to {t}.",
        "We will publish a draft bill to {t}.",
    ],
    0: [
        "We will legislate to {t} when parliamentary time allows.",
        "We will keep the need for legislation to {t} under review.",
        "The Government will consider legislation to {t} in due course.",
    ],
}
NOISE = [
    "Will the Minister bring forward legislation to {t}?",
    "I urge the Government to legislate to {t}.",
    "We have no plans to legislate to {t}.",
    "The Government should introduce a bill to {t}.",
    "The Minister spoke about {t} at length in Committee.",
]
MINISTER, BACKBENCHER = 1, 2


def _doc(doc_id, source, day, member, title, text, dept=None) -> dict:
    return {
        "doc_id": doc_id,
        "source": source,
        "source_id": doc_id.split("-", 1)[1],
        "event_date": day,
        "member_id": member,
        "house": "Commons",
        "title": title,
        "department": dept,
        "text": text,
        "matched_phrases": None,
        "fetched_at": NOW,
    }


def _posts(rows: list[tuple[int, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "member_id": m,
                "post_name": "Minister of State",
                "start_date": date.fromisoformat(s),
                "end_date": date.fromisoformat(e) if e else None,
                "fetched_at": NOW,
            }
            for m, s, e in rows
        ],
        columns=[c.name for c in member_government_posts.columns],
    )


def _bills(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """bills, stage_sittings (Commons first reading = intro date), long titles."""
    bills = pd.DataFrame({"bill_id": [r["bill_id"] for r in rows],
                          "short_title": [r["title"] for r in rows]})  # fmt: skip
    ss = pd.DataFrame(
        {
            "bill_id": [r["bill_id"] for r in rows],
            "api_stage_id": 6,
            "event_date": [r["intro_date"] for r in rows],
        }
    )
    return bills, ss, {r["bill_id"]: r["long_title"] for r in rows}


def synthetic_world(seed: int = 1):
    """The prototype's synthetic speeches with known answers."""
    rng = np.random.default_rng(seed)
    docs, truth, bills = [], [], []
    for k, (topic, variants, btitle, blong) in enumerate(TOPICS):
        start = pd.Timestamp("2016-01-01") + pd.Timedelta(days=int(rng.integers(0, 2500)))
        for j in range(int(rng.integers(2, 6))):
            day = (start + pd.Timedelta(days=int(j * rng.integers(10, 120)))).date()
            phr = str(rng.choice([topic, *variants]))
            level = (
                int(rng.choice([2, 1, 0], p=[0.5, 0.3, 0.2])) if j else int(rng.choice([2, 1, 0]))
            )
            sent = str(rng.choice(TEMPLATES[level])).format(t=phr)
            text = f"Today I am updating the House. {sent} Officials are working with stakeholders."
            source = str(rng.choice(["wms", "hansard_spoken"]))
            prefix = "wms" if source == "wms" else "han"
            doc_id = f"{prefix}-T{k}x{j}"
            docs.append(_doc(doc_id, source, day, MINISTER, phr.title(), text, "Department X"))
            truth.append({"mention_id": f"{doc_id}-1", "idea": k, "firmness": level})
        for j in range(int(rng.integers(1, 3))):  # noise: questions, demands, negations
            day = (start + pd.Timedelta(days=int(rng.integers(0, 400)))).date()
            text = str(rng.choice(NOISE)).format(t=topic)
            docs.append(_doc(f"han-N{k}x{j}", "hansard_spoken", day, MINISTER, topic.title(), text))
        if btitle:
            intro = (start + pd.Timedelta(days=int(rng.integers(60, 500)))).date()
            bills.append({"bill_id": 1000 + k, "title": btitle, "long_title": blong,
                          "intro_date": intro, "true_idea": k})  # fmt: skip
    for j in range(15):  # unrelated government bills
        intro = (pd.Timestamp("2016-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)))).date()
        bills.append({"bill_id": 2000 + j, "title": f"Finance (No. {j}) Bill",
                      "long_title": "A Bill to grant certain duties and to alter other duties",
                      "intro_date": intro, "true_idea": None})  # fmt: skip
    return pd.DataFrame(docs), pd.DataFrame(truth), bills


@pytest.fixture(scope="module")
def world():
    docs, truth, bill_rows = synthetic_world()
    bills, ss, long_titles = _bills(bill_rows)
    posts = _posts([(MINISTER, "2010-01-01", None)])
    result = nlp.run_pipeline(docs, posts, bills, ss, long_titles, now=NOW)
    return docs, truth, bill_rows, result


def test_selftest_commitments_and_firmness(world):
    _, truth, _, result = world
    kept = result.mentions[result.mentions["kept"]]
    found, want = set(kept["mention_id"]), set(truth["mention_id"])
    tp, fp, fn = len(found & want), len(found - want), len(want - found)
    assert tp / (tp + fp) == 1.0  # precision
    assert tp / (tp + fn) == 1.0  # recall
    fm = kept.merge(truth, on="mention_id", suffixes=("", "_true"))
    assert (fm["firmness"] == fm["firmness_true"]).mean() == 1.0


def test_selftest_grouping_and_linking(world):
    _, truth, bill_rows, result = world
    # topic groups (before merging) already agree well with the true ideas
    g = result.grouped.merge(truth, on="mention_id")
    assert adjusted_rand_score(g["idea"], g["group_id"]) >= 0.8
    # after merging groups that share a bill
    im = result.idea_mentions.merge(truth, on="mention_id")
    assert adjusted_rand_score(im["idea"], im["idea_id"]) >= 0.85
    true_bill = {r["true_idea"]: r["bill_id"] for r in bill_rows if r["true_idea"] is not None}
    links = result.idea_bill_links.set_index("idea_id")["bill_id"]
    correct = 0
    for idea_id, rows in im.groupby("idea_id"):
        want = true_bill.get(int(rows["idea"].mode().iloc[0]))
        got = links.get(idea_id)
        correct += (None if pd.isna(got) else int(got)) == want
    n_ideas = im["idea_id"].nunique()
    assert correct / n_ideas >= 0.85
    assert len(result.ideas) <= len(TOPICS) + 3


def test_selftest_tables_match_contract(world):
    _, _, _, result = world
    tables = result.tables()
    for name, df in tables.items():
        assert list(df.columns) == [c.name for c in takeup_metadata.tables[name].columns]
    ideas_df = tables["ideas"]
    assert ideas_df["idea_id"].str.fullmatch(r"nlp-\d{5}").all()
    assert ideas_df["idea_id"].is_unique and (ideas_df["origin"] == "nlp").all()
    # ideas numbered by trigger date
    assert list(ideas_df.sort_values("idea_id")["trigger_date"]) == sorted(ideas_df["trigger_date"])
    # every idea has exactly one link row; every idea mention is a kept signal mention
    assert set(tables["idea_bill_links"]["idea_id"]) == set(ideas_df["idea_id"])
    kept = set(result.mentions.loc[result.mentions["kept"], "mention_id"])
    assert set(tables["idea_mentions"]["mention_id"]) == kept
    counts = tables["idea_mentions"].groupby("idea_id").size()
    assert (ideas_df.set_index("idea_id")["n_mentions"] == counts).all()
    # mention ids: <doc_id>-<sentence index>
    m = tables["signal_mentions"]
    assert (m["mention_id"] == m["doc_id"] + "-1").all()


def test_pipeline_is_deterministic(world):
    docs, _, bill_rows, result = world
    bills, ss, long_titles = _bills(bill_rows)
    posts = _posts([(MINISTER, "2010-01-01", None)])
    again = nlp.run_pipeline(docs.sample(frac=1, random_state=3), posts, bills, ss, long_titles,
                             now=NOW)  # fmt: skip
    for name, df in result.tables().items():
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True), again.tables()[name].reset_index(drop=True)
        )


# ------------------------------------------------------------- unit behaviour ----
@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("We will legislate to ban no-fault evictions.", True),
        ("I will update the House in due course.", None),  # no trigger at all
        ("I will provide further details on the legislation in due course.", True),
        ("Officials will report back at the earliest opportunity.", None),
        ("Legislation will be introduced in due course.", True),
        ("The draft Bill lets smaller platforms off the hook.", False),
        ("We published a draft Bill last year.", False),
        ("We intend to publish a draft Bill shortly.", True),
        ("We are introducing the national living wage.", False),
        ("We will lay regulations in due course.", None),
    ],
)
def test_commitment_check(sentence, expected):
    level, phrase = nlp.classify_sentence(sentence)
    if expected is None:
        assert level is None or not nlp.commitment_check(sentence, phrase)[0]
        return
    assert level is not None
    assert nlp.commitment_check(sentence, phrase)[0] is expected


def test_rule_topic_strips_filler_and_falls_back_to_title():
    assert nlp.rule_topic("We will legislate to ban the sale of ivory.", "We will legislate") == (
        "ban the sale of ivory"
    )
    s = (
        "I have confirmed that we intend, when parliamentary time allows, to introduce "
        "legislation that comprehensively tackles all SLAPPs."
    )
    assert "SLAPPs" in nlp.rule_topic(s, "when parliamentary time allows")
    s = "When parliamentary time allows, we will introduce a water Bill creating new laws."
    assert nlp.rule_topic(s, "we will introduce a water Bill").startswith("water")
    # generic topic -> the statement title
    docs = pd.DataFrame(
        [
            _doc(
                "wms-1",
                "wms",
                date(2020, 1, 1),
                5,
                "Firearms Licensing: Sound Moderators",
                "We will legislate to ensure that this happens.",
            )
        ]
    )
    m = nlp.label_documents(docs, _posts([]), now=NOW)
    assert m.loc[0, "topic"] == "Firearms Licensing: Sound Moderators"


def test_hansard_minister_check_by_date():
    posts = _posts([(7, "2016-07-14", "2018-01-08"), (7, "2019-07-24", None)])
    docs = pd.DataFrame(
        [
            _doc("han-A", "hansard_spoken", date(2017, 3, 1), 7, "T", "We will legislate on fish."),
            _doc("han-B", "hansard_spoken", date(2018, 6, 1), 7, "T", "We will legislate on fish."),
            _doc("han-C", "hansard_spoken", date(2020, 6, 1), 7, "T", "We will legislate on fish."),
            _doc("han-D", "hansard_spoken", date(2017, 3, 1), 8, "T", "We will legislate on fish."),
            _doc("han-E", "hansard_spoken", date(2017, 3, 1), 9, "T", "We will legislate on fish."),
            _doc("han-F", "hansard_spoken", date(2017, 3, 1), None, "T", "We will legislate."),
            _doc("wms-G", "wms", date(2018, 6, 1), 7, "T", "We will legislate on fish."),
        ]
    )
    m = nlp.label_documents(docs, posts, fetched_members={7, 8}, now=NOW).set_index("doc_id")
    assert m.loc["han-A", "is_minister"] is True  # in post
    assert m.loc["han-B", "is_minister"] is False  # between posts
    assert m.loc["han-C", "is_minister"] is True  # open-ended post
    assert m.loc["han-D", "is_minister"] is False  # fetched, never held a post
    assert m.loc["han-E", "is_minister"] is None  # not fetched: unknown
    assert m.loc["han-F", "is_minister"] is None  # no member id
    assert m.loc["wms-G", "is_minister"] is True  # written statements: always ministers
    assert list(m.loc[m["kept"]].index) == ["han-A", "han-C", "wms-G"]
    assert m.loc["han-B", "drop_reason"] == "not_minister"
    assert m.loc["han-E", "drop_reason"] == "minister_unknown"
    assert m.loc["han-A", "department"] is None and m.loc["wms-G", "department"] is None


def test_cluster_vectors_matches_global_agglomerative():
    from sklearn.cluster import AgglomerativeClustering

    texts = [
        f"{a} {b}"
        for a in ["water bonuses", "football regulator", "rail ownership", "smoking age", "ivory"]
        for b in ["", "now", "law", "uk"]
    ]
    emb = nlp.Embedder().fit(texts)
    X = emb.encode(texts)
    ours = nlp.cluster_vectors(X, 0.3, block=3)
    ref = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.7
    ).fit_predict(X.toarray())
    assert adjusted_rand_score(ref, ours) == 1.0


# ------------------------------------------------------- merge record / leakage ----
def _merge_world():
    """Two differently worded groups (2018 and 2021) that both link to one bill (2022)."""
    docs = pd.DataFrame(
        [
            _doc("wms-1", "wms", date(2018, 3, 1), 1, "Ivory Trade",
                 "We will legislate to ban the ivory trade.", "Defra"),
            _doc("wms-2", "wms", date(2018, 5, 1), 1, "Ivory Trade",
                 "We will legislate to ban the ivory trade this session.", "Defra"),
            _doc("wms-3", "wms", date(2021, 6, 1), 1, "Elephant Tusks",
                 "We will legislate to outlaw elephant tusk sales.", "Defra"),
            _doc("wms-4", "wms", date(2019, 1, 1), 1, "Digital Identity",
                 "We intend to legislate for national digital identity cards.", "Home Office"),
        ]
    )  # fmt: skip
    bills, ss, long_titles = _bills(
        [{"bill_id": 501, "title": "Ivory Trade and Elephant Tusk Sales Bill",
          "intro_date": date(2022, 1, 10),
          "long_title": "A Bill to ban the ivory trade and elephant tusk sales"}]
    )  # fmt: skip
    return docs, bills, ss, long_titles


def test_merge_keeps_pre_merge_groups_and_earliest_trigger():
    docs, bills, ss, long_titles = _merge_world()
    r = nlp.run_pipeline(docs, _posts([]), bills, ss, long_titles, now=NOW)
    t = r.tables()
    assert len(t["ideas"]) == 2
    ivory = t["ideas"].iloc[0]
    assert ivory["idea_id"] == "nlp-00001" and ivory["trigger_date"] == date(2018, 3, 1)
    assert ivory["n_mentions"] == 3 and ivory["department"] == "Defra"
    im = t["idea_mentions"].set_index("mention_id")
    assert im.loc["wms-1-0", "group_id"] == im.loc["wms-2-0", "group_id"] == "g-wms-1-0"
    assert im.loc["wms-3-0", "group_id"] == "g-wms-3-0"  # separate group, same idea
    assert im.loc["wms-3-0", "idea_id"] == "nlp-00001"
    links = t["idea_bill_links"].set_index("idea_id")
    assert links.loc["nlp-00001", "bill_id"] == 501
    assert pd.isna(links.loc["nlp-00002", "bill_id"])
    assert links.loc["nlp-00002", "decided_by"] in {"auto", "rule"}

    # get_takeup_signals splits the groups until the bill's first reading is visible
    tables = {
        **t,
        "idea_events": pd.DataFrame(columns=[c.name for c in idea_events.columns]),
        "bills": bills,
        "stage_sittings": ss,
        "sessions": pd.DataFrame(
            {"session_id": [1], "start_date": [date(2017, 6, 21)], "parliament_id": [57]}
        ),  # fmt: skip
        "parliaments": pd.DataFrame(
            app_db.DEFAULT_PARLIAMENTS,
            columns=["parliament_id", "first_meeting_date", "dissolution_date", "election_date"],
        ),
        "sitting_days": pd.DataFrame({"date": [date(2018, 1, 10)], "house": ["Commons"]}),
    }
    data = TakeupData(tables)
    before = data.get_takeup_signals("2021-12-31", features=[])
    assert set(before["signal_id"]) == {
        "nlp-00001:g-wms-1-0", "nlp-00001:g-wms-3-0", "nlp-00002:g-wms-4-0"
    }  # fmt: skip
    after = data.get_takeup_signals("2022-02-01", features=[])
    assert set(after["signal_id"]) == {"nlp-00001", "nlp-00002:g-wms-4-0"}


def test_gap_over_two_years_splits_a_group():
    docs = pd.DataFrame(
        [
            _doc("wms-1", "wms", date(2015, 1, 1), 1, "Ivory", "We will legislate to ban ivory."),
            _doc("wms-2", "wms", date(2016, 1, 1), 1, "Ivory", "We will legislate to ban ivory."),
            _doc("wms-3", "wms", date(2019, 1, 1), 1, "Ivory", "We will legislate to ban ivory."),
        ]
    )
    bills, ss, lt = _bills([])
    r = nlp.run_pipeline(docs, _posts([]), bills, ss, lt, now=NOW)
    groups = r.idea_mentions.set_index("mention_id")["group_id"]
    assert groups["wms-1-0"] == groups["wms-2-0"] == "g-wms-1-0"
    assert groups["wms-3-0"] == "g-wms-3-0"
    assert len(r.ideas) == 2


# ------------------------------------------------------------------------ write ----
@pytest.fixture
def engine():
    eng = create_engine("sqlite://")
    takeup_metadata.create_all(eng)
    yield eng
    eng.dispose()


def _count(eng, table) -> int:
    with eng.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


def test_write_is_idempotent_and_leaves_sources_alone(engine):
    docs, bills, ss, long_titles = _merge_world()
    r = nlp.run_pipeline(docs, _posts([]), bills, ss, long_titles, now=NOW)
    takeup_db.upsert_rows(engine, source_documents, docs)
    takeup_db.upsert_rows(engine, member_government_posts, _posts([(1, "2010-01-01", None)]))
    first = nlp.write_results(engine, r)
    second = nlp.write_results(engine, r)
    assert first == second == {"signal_mentions": 4, "ideas": 2, "idea_mentions": 4,
                               "idea_bill_links": 2}  # fmt: skip
    assert _count(engine, signal_mentions) == 4 and _count(engine, ideas) == 2
    assert _count(engine, source_documents) == 4 and _count(engine, member_government_posts) == 1
    # a smaller rerun leaves no stale rows behind
    small = nlp.run_pipeline(docs.iloc[:1], _posts([]), bills, ss, long_titles, now=NOW)
    nlp.write_results(engine, small)
    assert _count(engine, signal_mentions) == 1 and _count(engine, ideas) == 1


def test_write_refuses_when_idea_events_exist(engine):
    docs, bills, ss, long_titles = _merge_world()
    r = nlp.run_pipeline(docs, _posts([]), bills, ss, long_titles, now=NOW)
    nlp.write_results(engine, r)
    with engine.begin() as conn:
        conn.execute(insert(idea_events).values(
            idea_id="nlp-00001", event_type="consultation", event_date=date(2018, 4, 1),
            title="c", source_url="u", ingested_at=NOW))  # fmt: skip
    with pytest.raises(RuntimeError, match="idea_events"):
        nlp.write_results(engine, r)
    assert _count(engine, ideas) == 2  # nothing changed
    nlp.write_results(engine, r, clear_events=True)
    assert _count(engine, idea_events) == 0


def test_clear_refuses_source_tables(engine):
    with engine.begin() as conn, pytest.raises(ValueError, match="raw source"):
        takeup_db.clear_derived_tables(conn, [source_documents])


# ----------------------------------------------------------------- review / score ----
def test_review_csv_guard_and_score(tmp_path: Path, world):
    _, _, bill_rows, result = world
    bills, ss, long_titles = _bills(bill_rows)
    tables = result.tables()
    backbench = (
        tables["signal_mentions"]
        .iloc[[0]]
        .assign(mention_id="han-X-0", is_minister=False, member_id=BACKBENCHER)
    )
    tables["signal_mentions"] = pd.concat([tables["signal_mentions"], backbench])
    status = nlp.export_review(tables, nlp.prepare_bills(bills, ss, long_titles), tmp_path)
    assert set(status.values()) == {"written"}
    mentions = pd.read_csv(tmp_path / "review_mentions.csv")
    assert list(mentions.columns) == ["id", "date", "source", "title", "sentence", "firmness",
                                      "topic", "kept", "correct"]  # fmt: skip
    assert (~mentions["kept"]).any() and mentions["kept"].any()  # includes dropped ones
    links = pd.read_csv(tmp_path / "review_links.csv")
    assert {"topic", "example_sentence", "bill_title", "bill_long_title", "similarity",
            "linked", "correct", "correct_bill_id"} <= set(links.columns)  # fmt: skip

    # same seed, same sample; an unlabelled file may be overwritten
    again = nlp.export_review(tables, nlp.prepare_bills(bills, ss, long_titles), tmp_path)
    assert set(again.values()) == {"written"}
    pd.testing.assert_frame_equal(mentions, pd.read_csv(tmp_path / "review_mentions.csv"))

    # label some rows: the file is then never overwritten
    labelled = mentions.copy()
    labelled["correct"] = ""
    labelled.loc[labelled["kept"], "correct"] = "y"
    labelled.loc[labelled.index[labelled["kept"]][0], "correct"] = "n"
    labelled.to_csv(tmp_path / "review_mentions.csv", index=False)
    with pytest.raises(nlp.ReviewAlreadyLabelled):
        nlp.write_review_csv(mentions, tmp_path / "review_mentions.csv")
    status = nlp.export_review(tables, nlp.prepare_bills(bills, ss, long_titles), tmp_path)
    assert status[str(tmp_path / "review_mentions.csv")].startswith("skipped")
    assert pd.read_csv(tmp_path / "review_mentions.csv", dtype=str)["correct"].notna().any()

    scores = nlp.score(tmp_path)
    n_kept = int(labelled["kept"].sum())
    assert scores["review_mentions"]["positives_checked"] == n_kept
    assert scores["review_mentions"]["precision"] == pytest.approx((n_kept - 1) / n_kept)
    assert scores["review_links"] == {"labelled": 0}


def _link_one(topic, bill_title, gap_days, long_title="", text=None):
    """Link a single group to a single candidate bill (plus a decoy) and return the row."""
    t = pd.Timestamp("2015-01-05")
    groups = pd.DataFrame(
        [{"group_id": "g1", "trigger_date": t, "topic": topic, "text": text or topic, "names": ""}]
    )
    bills = pd.DataFrame(
        [
            {"bill_id": 1, "title": bill_title, "long_title": long_title,
             "first_reading": t + pd.Timedelta(days=gap_days)},
            {"bill_id": 2, "title": "Zoo Licensing Bill", "long_title": "A Bill about zoos",
             "first_reading": t + pd.Timedelta(days=30)},
        ]
    )  # fmt: skip
    return nlp.link_groups(groups, bills).iloc[0]


def test_routine_money_bills_are_never_linked():
    row = _link_one("supply and appropriation of money", "Supply and Appropriation Act 2016", 300)
    assert pd.isna(row["bill_id"]) or int(row["bill_id"]) != 1


def test_finance_bill_only_within_a_year():
    near = _link_one("finance bill tax relief changes", "Finance Act 2015", 120)
    far = _link_one("finance bill tax relief changes", "Finance Act 2017", 800)
    assert int(near["bill_id"]) == 1
    assert pd.isna(far["bill_id"]) or int(far["bill_id"]) != 1


def test_procedural_topic_is_not_linked():
    row = _link_one("Madam Deputy Speaker", "Deputy Speaker Arrangements Bill", 200)
    assert pd.isna(row["bill_id"])


def test_long_gap_needs_a_strong_match():
    weak = _link_one(
        "modernise consumer rules", "Consumer and Markets Reform Bill", 4 * 365,
        long_title="A Bill to make provision about competition, markets and digital platforms",
    )  # fmt: skip
    assert pd.isna(weak["bill_id"])
    strong = _link_one("recall of MPs petitions", "Recall of MPs Act 2019", 4 * 365)
    assert int(strong["bill_id"]) == 1
