"""Offline tests for the take-up loaders (DATA_CONTRACT §10.0a, §10.0b) and CLI.

Everything runs on temporary cache directories and an in-memory SQLite engine created from
the take-up metadata, with the database settings monkeypatched; no test connects to a real
database."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select

from westminster_weather.takeup import cli, loaders
from westminster_weather.takeup import db as takeup_db
from westminster_weather.takeup.tables import (
    member_government_posts,
    source_documents,
    takeup_metadata,
)

EXT_A = "BD5241B0-2BF4-4DE6-94A3-AD9D95C024B7"
PRODUCTION = "postgresql://u:p@production.invalid/db"


@pytest.fixture(autouse=True)
def _offline_settings(monkeypatch):
    """No real database URL is ever visible to these tests."""
    monkeypatch.setattr(takeup_db.settings, "database_url", PRODUCTION)
    monkeypatch.setattr(takeup_db.settings, "database_url_unpooled", None)
    monkeypatch.setattr(takeup_db.settings, "database_url_takeup", None)
    monkeypatch.setattr(takeup_db.settings, "database_url_takeup_unpooled", None)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data))


def _contribution(ext_id: str | None, item_id: str = "1", **extra) -> dict:
    return {
        "MemberId": "3610",
        "ItemId": item_id,
        "ContributionExtId": ext_id,
        "ContributionTextFull": "We will legislate <b>this session</b>.",
        "DebateSection": "Digital Economy Bill",
        "SittingDate": "2016-12-13T00:00:00",
        "House": "Lords",
        **extra,
    }


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    c = tmp_path / "cache"
    _write(
        c / "wms" / "full" / "1001.json",
        {
            "id": "1001",
            "memberId": "1405",
            "dateMade": "2018-11-05T00:00:00",
            "title": "Tailored Review",
            "text": "<p>First&nbsp;para &amp; more.</p><p>Second <em>para</em>.</p>",
            "house": "Commons",
            "answeringBodyName": "Foreign and Commonwealth Office",
        },
    )
    _write(c / "wms" / "full" / "1002.json", {"id": "1002", "dateMade": None, "text": "x"})
    _write(c / "wms" / "full" / "1003.json", "{not json")
    spoken = c / "hansard" / "spoken"
    _write(spoken / "we-will-legislate" / "2016-01-01_2016-12-31.json", [_contribution(EXT_A)])
    _write(
        spoken / "draft-bill" / "2016-01-01_2016-12-31.json",
        [
            _contribution(EXT_A),
            _contribution(None, item_id="777"),
            _contribution("NO-DATE", SittingDate=None),
            "not a record",
        ],
    )
    _write(
        c / "hansard" / "written" / "we-will-legislate" / "2010-01-01_2010-12-31.json",
        [_contribution("W-1", MemberId="0", House="Unknown")],
    )
    members = c / "members"
    post = {"name": "Minister for X", "start": "2010-05-12T00:00:00", "end": "2012-01-01"}
    _write(members / "1.json", [post, post, {"name": "Whip", "start": "2015-05-08", "end": None}])
    _write(members / "2.json", [])
    _write(members / "3.json", [{"name": None, "start": "2010-01-01", "end": None}])
    _write(members / "bad.json", [])
    return c


def test_strip_html_keeps_paragraphs_and_decodes_entities():
    text = loaders.strip_html("<p>First&nbsp;para &amp; more.</p><p>Second <em>para</em>.</p>")
    assert text == "First para & more.\n\nSecond para ."
    assert loaders.strip_html("a<br/>b<ul><li>c</li><li>d</li></ul>") == "a\n\nb c\n\nd"
    assert loaders.strip_html(None) == ""


def test_source_documents_dedupe_prefixes_and_skips(cache: Path):
    df = loaders.build_source_documents(cache)
    assert list(df.columns) == [c.name for c in source_documents.columns]
    by_id = df.set_index("doc_id")
    assert sorted(by_id.index) == sorted(["wms-1001", f"han-{EXT_A}", "han-777", "hanw-W-1"])

    wms = by_id.loc["wms-1001"]
    assert wms["source"] == "wms" and wms["source_id"] == "1001"
    assert wms["event_date"] == date(2018, 11, 5) and wms["member_id"] == 1405
    assert wms["department"] == "Foreign and Commonwealth Office"
    assert wms["text"] == "First para & more.\n\nSecond para ."
    assert pd.isna(wms["matched_phrases"])
    assert str(wms["fetched_at"].tz) == "UTC"

    han = by_id.loc[f"han-{EXT_A}"]
    assert han["source"] == "hansard_spoken" and han["source_id"] == EXT_A
    assert han["matched_phrases"] == "we will legislate,draft bill"  # SEARCH_PHRASES order
    assert han["text"] == "We will legislate this session ."
    assert han["title"] == "Digital Economy Bill" and han["house"] == "Lords"
    assert by_id.loc["han-777", "source_id"] == "777"  # ItemId fallback

    written = by_id.loc["hanw-W-1"]
    assert written["source"] == "hansard_written"
    assert pd.isna(written["member_id"]) and written["house"] is None

    skipped = df.attrs["report"]["skipped"]
    assert skipped == {
        "unreadable_file": 1,
        "wms_missing_id_or_date": 1,
        "hansard_missing_id_or_date": 1,
        "hansard_not_a_record": 1,
    }


def test_member_posts_dedupe_open_posts_and_fetched_set(cache: Path):
    df, fetched = loaders.build_member_government_posts(cache)
    assert list(df.columns) == [c.name for c in member_government_posts.columns]
    assert fetched == {1, 2, 3}  # 2 = fetched with no posts; bad.json is not a member
    assert len(df) == 2 and set(df["member_id"]) == {1}
    whip = df[df["post_name"] == "Whip"].iloc[0]
    assert whip["start_date"] == date(2015, 5, 8) and pd.isna(whip["end_date"])
    report = df.attrs["report"]
    assert report["members_without_posts"] == 2 and report["open_posts"] == 1
    assert report["skipped"] == {
        "malformed_member_file": 1,
        "post_missing_name_or_start": 1,
        "duplicate_pk": 1,
    }


def _count(engine, table) -> int:
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


def test_load_functions_are_idempotent_on_sqlite(cache: Path):
    engine = create_engine("sqlite://")
    takeup_metadata.create_all(engine)
    docs = loaders.build_source_documents(cache)
    posts, _ = loaders.build_member_government_posts(cache)
    for _ in range(2):
        assert loaders.load_source_documents(engine, docs, batch_size=2) == 4
        assert loaders.load_member_government_posts(engine, posts) == 2
    assert _count(engine, source_documents) == 4
    assert _count(engine, member_government_posts) == 2

    docs.loc[docs["doc_id"] == "wms-1001", "title"] = "Updated"
    with engine.begin() as conn:
        loaders.load_source_documents(conn, docs)
        title = conn.execute(
            select(source_documents.c.title).where(source_documents.c.doc_id == "wms-1001")
        ).scalar_one()
        member = conn.execute(
            select(source_documents.c.member_id).where(source_documents.c.doc_id == "wms-1001")
        ).scalar_one()
    assert title == "Updated" and member == 1405
    assert _count(engine, source_documents) == 4


@pytest.mark.parametrize("command", ["migrate", "load-raw", "db-status"])
def test_cli_refuses_without_takeup_url(cache: Path, command):
    extra = ["--cache-dir", str(cache)] if command == "load-raw" else []
    with pytest.raises(SystemExit, match="DATABASE_URL_TAKEUP"):
        cli.main([command, *extra])


def test_cli_has_no_kings_speech_command():
    with pytest.raises(SystemExit):
        cli.main(["load-kings-speech", "--dry-run"])


def test_cli_dry_run_never_creates_an_engine(monkeypatch, cache: Path, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("dry run created an engine")

    for name in ("get_takeup_engine", "get_takeup_direct_engine", "migrate", "create_engine"):
        monkeypatch.setattr(takeup_db, name, boom)
    cli.main(["load-raw", "--dry-run", "--cache-dir", str(cache)])
    cli.main(["migrate", "--dry-run"])
    cli.main(["db-status", "--dry-run"])
    out = capsys.readouterr().out
    assert "source_documents: 4 rows" in out
    assert "member_government_posts: 2 rows" in out
    assert "text_bytes" in out and "members_fetched: 3" in out
    assert "kings_speech" not in out
    assert "postgres" not in out.lower() and "invalid" not in out


def test_upsert_rows_refuses_production_url(monkeypatch):
    engine = create_engine("sqlite://")
    takeup_metadata.create_all(engine)
    monkeypatch.setattr(
        takeup_db.settings, "database_url", engine.url.render_as_string(hide_password=False)
    )
    with pytest.raises(takeup_db.TakeupDatabaseNotConfigured, match="production"):
        takeup_db.upsert_rows(engine, "ideas", [{"idea_id": "nlp-00001"}])
