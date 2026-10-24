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
        "source_id": mid.rsplit("-", 1)[0],
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
        # B-1: two topic groups, merged because both link to bill 101 (first read 2016-03-01).
        _mention("wms-1-0", "2015-07-01", firmness=1, dept="Department for Education"),
        _mention("wms-2-0", "2015-09-01", firmness=2),
        _mention("wms-2-1", "2015-09-01", firmness=2),  # same statement: not a repeat
        _mention("hansard_spoken-3-0", "2015-11-01", source="hansard_spoken", firmness=0),
        _mention("wms-4-0", "2016-05-01", firmness=2),
        _mention("wms-5-0", "2015-08-01", minister=False),  # not a minister: invisible
        _mention("wms-6-0", "2015-06-01", commit=False),  # not a commitment: invisible
        _mention("wms-7-0", "2016-02-01"),  # B-2, bill 102 (Lords first)
        _mention("wms-8-0", "2023-01-10", dept="Home Office"),  # B-3, bill after 2024 change
        _mention("wms-9-0", "2022-02-01"),  # B-4, bill before the 2024 dissolution
        _mention("hansard_written-10-0", "2021-01-05", source="hansard_written"),  # B-5
        _mention("wms-11-0", "2016-01-10", firmness=2),  # B-6, same-party 2017 election
        _mention("wms-12-0", "2008-01-01"),  # B-7, 2010 change of government
        _mention("wms-13-0", "2027-01-01", firmness=2),  # future mention of B-6
    ]
    ks = [
        {
            "item_id": "ks-2016-05-18-01",
            "speech_date": D("2016-05-18"),
            "announced_name": "Prisons Bill",
            "is_draft": False,
            "is_carried_over": False,
            "department": "Ministry of Justice",
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
            "source_url": "u",
            "ingested_at": NOW,
        },
    ]
    idea_mentions = [
        ("B-1", "wms-1-0", "g1"),
        ("B-1", "wms-2-0", "g1"),
        ("B-1", "wms-2-1", "g1"),
        ("B-1", "wms-5-0", "g1"),
        ("B-1", "wms-6-0", "g1"),
        ("B-1", "hansard_spoken-3-0", "g2"),
        ("B-1", "wms-4-0", "g2"),
        ("B-2", "wms-7-0", "g3"),
        ("B-3", "wms-8-0", "g4"),
        ("B-4", "wms-9-0", "g5"),
        ("B-5", "hansard_written-10-0", "g6"),
        ("B-6", "wms-11-0", "g7"),
        ("B-6", "ks-2016-05-18-02", "g7"),
        ("B-6", "wms-13-0", "g7"),
        ("B-7", "wms-12-0", "g8"),
        ("A-ks-2016-05-18-01", "ks-2016-05-18-01", "ks-2016-05-18-01"),
    ]
    ideas = [
        {
            "idea_id": i,
            "tier": "A" if i.startswith("A") else "B",
            "topic": f"topic {i}",
            "trigger_date": D("2015-01-01"),
            "department": None,
            "first_source": "wms",
            "initial_firmness": 1,
            "n_mentions": 1,
            "ingested_at": NOW,
        }
        for i in sorted({r[0] for r in idea_mentions})
    ]
    links = [
        ("B-1", 101),
        ("B-2", 102),
        ("B-3", 103),
        ("B-4", 104),
        ("B-5", None),
        ("A-ks-2016-05-18-01", 105),
    ]
    events = [
        ("A-ks-2016-05-18-01", "consultation", "2015-12-01"),
        ("A-ks-2016-05-18-01", "white_paper", "2016-02-01"),
        ("A-ks-2016-05-18-01", "draft_bill", "2017-01-01"),
        ("B-6", "consultation_response", "2019-01-01"),
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
        "kings_speech_items": pd.DataFrame(ks),
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
    p = t["parliaments"].copy()
    p = p[p["first_meeting_date"] <= as_of].copy()
    for col in ("dissolution_date", "election_date"):
        p[col] = [d if d is not None and d <= as_of else None for d in p[col]]
    out["parliaments"] = p.reset_index(drop=True)
    return out


AS_OFS = [
    D("2010-04-20"),  # between the 2010 dissolution and the election
    D("2016-01-01"),
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


def test_invisible_mentions_are_ignored(synthetic):
    df = _signals(synthetic, "2016-01-01")
    # wms-6-0 (2015-06-01, not a commitment) would otherwise be the first mention.
    assert df.loc["B-1:g1", "signal_date"] == pd.Timestamp("2015-07-01")
    assert "B-7:g8" in df.index
    assert "B-6:g7" not in df.index  # first mention 2016-01-10 is after as_of
    later = _signals(synthetic, "2018-01-01").loc["B-6:g7"]
    assert later["z_initial_firmness"] == 2 and pd.isna(later["x_firm_commitment"])
    assert later["x_repeat_mention"] == pd.Timestamp("2016-05-18")  # King's Speech item
    assert pd.isna(later["x_consultation_response"])  # 2019-01-01 is after as_of
    latest = _signals(synthetic, "2026-10-24").loc["B-6:g7"]
    assert latest["x_consultation_response"] == pd.Timestamp("2019-01-01")


# --------------------------------------------------------------- merge and split


def test_groups_split_before_first_reading(synthetic):
    df = _signals(synthetic, "2016-01-01")
    assert {"B-1:g1", "B-1:g2"} <= set(df.index)
    assert "B-1" not in df.index
    g1, g2 = df.loc["B-1:g1"], df.loc["B-1:g2"]
    assert g1["signal_date"] == pd.Timestamp("2015-07-01")
    assert g1["x_firm_commitment"] == pd.Timestamp("2015-09-01")
    assert g1["x_repeat_mention"] == pd.Timestamp("2015-09-01")
    assert g1["z_src_wms"] == 1 and g1["z_initial_firmness"] == 1
    assert pd.isna(g1["bill_date"]) and pd.isna(g1["bill_id"])
    assert g2["signal_date"] == pd.Timestamp("2015-11-01")
    assert g2["z_src_hansard"] == 1 and g2["z_src_wms"] == 0
    assert pd.isna(g2["x_repeat_mention"])
    assert (df["idea_id"] == "B-1").sum() == 2


def test_groups_merge_after_first_reading(synthetic):
    df = _signals(synthetic, "2016-06-01")
    assert "B-1" in df.index and "B-1:g1" not in df.index and "B-1:g2" not in df.index
    row = df.loc["B-1"]
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
    assert df.loc["B-2", "bill_date"] == pd.Timestamp("2016-06-10")


def test_bill_not_visible_before_its_first_reading(synthetic):
    df = _signals(synthetic, "2016-06-01")
    assert pd.isna(df.loc["B-2:g3", "bill_date"]) and pd.isna(df.loc["B-2:g3", "bill_id"])


# ------------------------------------------------------- change of government


def test_change_of_government_2024_censors(synthetic):
    df = _signals(synthetic, "2026-10-24")
    b3 = df.loc["B-3"]  # merged: bill first read 2025-01-15, after the 2024 dissolution
    assert pd.isna(b3["bill_date"]) and pd.isna(b3["bill_id"])
    assert b3["data_end"] == pd.Timestamp("2024-05-30")
    b5 = df.loc["B-5:g6"]  # never linked
    assert b5["data_end"] == pd.Timestamp("2024-05-30") and pd.isna(b5["bill_date"])
    b4 = df.loc["B-4"]  # bill before the dissolution is kept
    assert b4["bill_date"] == pd.Timestamp("2023-06-01")
    assert b4["data_end"] == pd.Timestamp("2026-10-24")


def test_change_of_government_not_applied_before_it_happens(synthetic):
    df = _signals(synthetic, "2024-01-01")
    assert df.loc["B-3:g4", "data_end"] == pd.Timestamp("2024-01-01")
    assert df.loc["B-5:g6", "data_end"] == pd.Timestamp("2024-01-01")


@pytest.mark.parametrize(
    ("as_of", "ids"),
    [("2024-06-15", ["B-3:g4", "B-5:g6", "B-6:g7", "B-1"]), ("2024-07-04", ["B-3:g4", "B-5:g6"])],
)
def test_change_of_government_unknown_before_its_election(synthetic, as_of, ids):
    """Between the dissolution and the result, the coming change of government is unknown."""
    df = _signals(synthetic, as_of)
    for sid in ids:
        assert df.loc[sid, "data_end"] == pd.Timestamp(as_of), sid
    early = _signals(synthetic, "2010-04-20")
    assert early.loc["B-7:g8", "data_end"] == pd.Timestamp("2010-04-20")
    after = _signals(synthetic, "2024-07-05")  # the result is known the day after polling
    assert after.loc["B-3:g4", "data_end"] == pd.Timestamp("2024-05-30")
    assert after.loc["B-5:g6", "data_end"] == pd.Timestamp("2024-05-30")


def test_same_and_change_of_party_dissolutions_alike_before_result(synthetic):
    """No difference between a same-party (2017) and a change (2024) dissolution window."""
    for as_of in ("2017-05-20", "2024-06-15"):
        df = _signals(synthetic, as_of)
        df = df[df["signal_date"] > pd.Timestamp("2010-04-12")]  # B-7: the 2010 change
        assert len(df) and (df["data_end"] == pd.Timestamp(as_of)).all(), as_of


def test_change_of_government_2010(synthetic):
    df = _signals(synthetic, "2018-01-01")
    b7 = df.loc["B-7:g8"]
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
    later = _signals(synthetic, "2018-01-01").loc["B-6:g7"]
    assert later["x_new_parliament"] == pd.Timestamp("2017-06-21")
    assert later["data_end"] == pd.Timestamp("2018-01-01")  # not censored
    assert later["x_kings_speech"] == pd.Timestamp("2016-05-18")
    before = _signals(synthetic, "2017-06-15").loc["B-6:g7"]
    assert pd.isna(before["x_new_parliament"])  # election held, State Opening not yet


# ---------------------------------------------------------------------- features


def test_tier_a_signal(synthetic):
    df = _signals(synthetic, "2016-12-31")
    a = df.loc["A-ks-2016-05-18-01"]
    assert a["tier"] == "A"
    assert a["z_initial_firmness"] == 2
    assert a["z_src_wms"] == 0 and a["z_src_hansard"] == 0  # King's Speech = reference
    assert pd.isna(a["x_kings_speech"])  # announced at the signal itself
    assert a["x_consultation"] == pd.Timestamp("2015-12-01")
    assert a["x_white_paper"] == pd.Timestamp("2016-02-01")
    assert pd.isna(a["x_draft_bill"])  # 2017-01-01 is after as_of
    assert a["bill_date"] == pd.Timestamp("2016-06-20")


def test_election_and_session_features(synthetic):
    row = _signals(synthetic, "2016-01-01").loc["B-1:g1"]  # 2015-07-01
    assert row["z_years_since_election"] == pytest.approx(55 / 365.25)  # since 2015-05-07
    assert row["z_months_into_session"] == pytest.approx(35 / 30.4375)  # since 2015-05-27


def test_departments_and_sparse_feature_drop(synthetic):
    data = TakeupData(synthetic)
    df = data.get_takeup_signals("2026-10-24", min_positive=0)
    assert {"z_dept_education", "z_dept_home_office", "z_dept_ministry_of_justice"} <= set(
        df.columns
    )
    assert df.set_index("signal_id").loc["B-1", "z_dept_education"] == 1
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


def test_tables_on_own_metadata():
    import westminster_weather.models  # noqa: F401  (registers the app tables on Base)

    assert tables.takeup_metadata is not app_db.Base.metadata
    names = {
        "kings_speech_items",
        "signal_mentions",
        "ideas",
        "idea_mentions",
        "idea_events",
        "idea_bill_links",
        "takeup_runs",
        "takeup_forecasts",
    }
    assert set(tables.takeup_metadata.tables) == names
    assert not names & set(app_db.Base.metadata.tables)
    assert not tables.idea_bill_links.c.bill_id.foreign_keys  # bills lives in production
    assert [c.name for c in tables.idea_mentions.primary_key] == ["idea_id", "mention_id"]
    # several split signals of one idea can be forecast at the same as_of
    assert [c.name for c in tables.takeup_forecasts.primary_key] == [
        "run_id",
        "signal_id",
        "as_of",
    ]
    assert not tables.takeup_forecasts.c.idea_id.nullable


def test_engines_raise_without_takeup_url(monkeypatch):
    monkeypatch.setattr(takeup_db.settings, "takeup_database_url", None)
    monkeypatch.setattr(takeup_db.settings, "takeup_database_url_unpooled", None)
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured):
        takeup_db.get_takeup_engine()
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured):
        takeup_db.get_takeup_direct_engine()
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured):
        takeup_db.migrate()
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured):
        takeup_db.upsert("ideas", [{"idea_id": "B-1"}])


def test_engines_refuse_production_url(monkeypatch):
    monkeypatch.setattr(takeup_db.settings, "takeup_database_url", app_db.settings.database_url)
    monkeypatch.setattr(takeup_db.settings, "takeup_database_url_unpooled", None)
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured):
        takeup_db.get_takeup_engine()


def test_engines_use_takeup_url(monkeypatch):
    url = "postgresql://u:p@takeup-dev.invalid/db"
    monkeypatch.setattr(takeup_db.settings, "takeup_database_url", url)
    monkeypatch.setattr(takeup_db.settings, "takeup_database_url_unpooled", None)
    engine = takeup_db.get_takeup_engine()  # no connection is made
    assert engine.url.host == "takeup-dev.invalid"
    assert engine.url.drivername == "postgresql+psycopg"
    assert takeup_db.get_takeup_direct_engine().url.host == "takeup-dev.invalid"


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

    t = _first_reading(synthetic, 104, "2022-01-28")  # B-4's mention is on 2022-02-01
    data = TakeupData(t)
    df = data.get_takeup_signals("2023-01-01", min_positive=0)
    row = df.set_index("signal_id").loc["B-4"]
    assert row["signal_date"] == row["bill_date"] == pd.Timestamp("2022-01-28")
    assert row["bill_id"] == 104 and data.n_signal_date_moved == 1
    assert row["z_initial_firmness"] == 1  # from the mention itself
    d = prepare(df, data.sitting_dates("2023-01-01"), data_end="2023-01-01")
    k = list(d["ids"]).index("B-4")
    assert d["delta"][k] == 1 and d["t"][k] == 1
    assert d["n_bill_before_signal"] == 0


def test_bill_long_before_signal_is_dropped(synthetic):
    """A commitment to a bill that already exists is not a signal (not a silent idea)."""
    t = _first_reading(synthetic, 104, "2021-06-01")
    data = TakeupData(t)
    df = data.get_takeup_signals("2023-01-01", min_positive=0)
    assert "B-4" not in set(df["signal_id"]) and "B-4:g5" not in set(df["signal_id"])
    assert data.dropped_signals == {"B-4": 104}


def _with_ks_item(t, item_id, speech, carried, bill_id=None):
    t = dict(t)
    item = {
        "item_id": item_id,
        "speech_date": D(speech),
        "announced_name": "Carried Bill",
        "is_draft": False,
        "is_carried_over": carried,
        "department": None,
        "source_url": "u",
        "ingested_at": NOW,
    }
    idea = f"A-{item_id}"
    t["kings_speech_items"] = pd.concat(
        [t["kings_speech_items"], pd.DataFrame([item])], ignore_index=True
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
    df = _signals(t, "2020-01-01")
    assert not any(i.startswith("A-ks-2017-06-21") for i in df["idea_id"])
    # the same items announced as new are signals
    t = _with_ks_item(synthetic, "ks-2017-06-21-02", "2017-06-21", False)
    assert "A-ks-2017-06-21-02:ks-2017-06-21-02" in _signals(t, "2020-01-01").index


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
    merged (through the bill) into a Tier A idea with events, tier A and a title."""
    t = dict(synthetic)
    extra = _mention("wms-20-0", "2016-04-01", firmness=1)
    t["signal_mentions"] = pd.concat(
        [t["signal_mentions"], pd.DataFrame([extra])], ignore_index=True
    )
    im = pd.DataFrame(
        [("A-ks-2016-05-18-01", "wms-20-0", "g9")], columns=["idea_id", "mention_id", "group_id"]
    )
    t["idea_mentions"] = pd.concat([t["idea_mentions"], im], ignore_index=True)
    as_of = "2016-06-01"  # bill 105 is first read on 2016-06-20
    merged_later = _signals(t, as_of)
    alone = _signals(_own_idea(t, "g9", "B-99"), as_of)
    row = merged_later.loc["A-ks-2016-05-18-01:g9"]
    assert row["tier"] == "B" and row["title"] == "topic"
    assert pd.isna(row["x_consultation"]) and pd.isna(row["x_white_paper"])
    pd.testing.assert_series_equal(
        _comparable(row), _comparable(alone.loc["B-99:g9"]), check_names=False
    )
    # the group holding the Tier A item keeps the item's own events
    anchor = merged_later.loc["A-ks-2016-05-18-01:ks-2016-05-18-01"]
    assert anchor["tier"] == "A" and anchor["title"] == "Prisons Bill"
    assert anchor["x_consultation"] == pd.Timestamp("2015-12-01")
    # after the first reading the idea is one signal with idea-level tier, title and events
    after = _signals(t, "2016-07-01").loc["A-ks-2016-05-18-01"]
    assert after["tier"] == "A" and after["title"] == "topic A-ks-2016-05-18-01"
    assert after["signal_date"] == pd.Timestamp("2016-04-01")
    assert after["x_white_paper"] == pd.Timestamp("2016-02-01")


def test_multi_group_tier_b_idea_events_wait_for_the_merge(synthetic):
    t = dict(synthetic)
    ev = t["idea_events"].iloc[[0]].assign(idea_id="B-1", event_date=D("2015-10-01"))
    t["idea_events"] = pd.concat([t["idea_events"], ev], ignore_index=True)
    split = _signals(t, "2016-01-01")
    assert split.loc[["B-1:g1", "B-1:g2"], "x_consultation"].isna().all()
    merged = _signals(t, "2016-06-01").loc["B-1"]
    assert merged["x_consultation"] == pd.Timestamp("2015-10-01")


# ---------------------------------------------------------- same-day mentions


def test_firmness_on_signal_day_uses_every_mention(synthetic):
    t = dict(synthetic)
    extra = _mention("hansard_spoken-77-0", "2022-02-01", source="hansard_spoken", firmness=2)
    t["signal_mentions"] = pd.concat(
        [t["signal_mentions"], pd.DataFrame([extra])], ignore_index=True
    )
    im = pd.DataFrame(
        [("B-4", "hansard_spoken-77-0", "g5")], columns=["idea_id", "mention_id", "group_id"]
    )
    t["idea_mentions"] = pd.concat([t["idea_mentions"], im], ignore_index=True)
    row = _signals(t, "2023-01-01").loc["B-4:g5"]
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
