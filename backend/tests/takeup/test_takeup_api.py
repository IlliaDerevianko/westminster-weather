"""Offline tests for the take-up API (``takeup/api.py``) with a fake TakeupData and fixed model."""

import json
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from westminster_weather import db as app_db
from westminster_weather.main import app
from westminster_weather.takeup import api
from westminster_weather.takeup import model as tm
from westminster_weather.takeup.signals import TakeupData

NOW = datetime(2026, 10, 24, tzinfo=UTC)
D = date.fromisoformat
ZCOLS = ["z_src_hansard", "z_initial_firmness"]
XCOLS = ["x_consultation", "x_white_paper"]


def _mention(mid: str, day: str, source: str = "wms", firmness: int = 1) -> dict:
    return {
        "mention_id": mid,
        "source": source,
        "doc_id": mid.rsplit("-", 1)[0],
        "event_date": D(day),
        "member_id": 1,
        "is_minister": True,
        "department": None,
        "title": f"title {mid}",
        "sentence": f"We will legislate ({mid}).",
        "snippet": "s",
        "phrase": "we will legislate",
        "firmness": firmness,
        "is_commitment": True,
        "topic": f"topic {mid}",
        "labelled_by": "rules",
        "ingested_at": NOW,
    }


def tables() -> dict[str, pd.DataFrame]:
    """nlp-00001: groups g1 + g2, merged through bill 101 (first read 2016-03-01).
    nlp-00002: no bill; a 2017 mention. nlp-00003: first mentioned in 2018."""
    mentions = [
        _mention("wms-1-0", "2015-07-01", firmness=2),
        _mention("han-2-0", "2015-11-01", source="hansard_spoken"),
        _mention("wms-3-0", "2015-09-01"),
        _mention("wms-4-0", "2017-01-01", firmness=2),
        _mention("wms-5-0", "2018-01-01"),
    ]
    im = [
        ("nlp-00001", "wms-1-0", "g1"),
        ("nlp-00001", "han-2-0", "g2"),
        ("nlp-00002", "wms-3-0", "g3"),
        ("nlp-00002", "wms-4-0", "g3"),
        ("nlp-00003", "wms-5-0", "g4"),
    ]
    ideas = [
        {
            "idea_id": i,
            "origin": "nlp",
            "topic": f"idea {i}",
            "trigger_date": D("2015-01-01"),
            "department": None,
            "first_source": "wms",
            "initial_firmness": 1,
            "n_mentions": 1,
            "ingested_at": NOW,
        }
        for i in ("nlp-00001", "nlp-00002", "nlp-00003")
    ]
    events = [
        ("nlp-00001", "consultation", "2015-12-01"),
        ("nlp-00002", "white_paper", "2016-01-15"),
        ("nlp-00002", "consultation", "2016-09-01"),  # after the 2016-06-01 as_of
    ]
    sitting_days = pd.date_range("2010-01-05", "2026-10-20", freq="W-TUE")
    return {
        "signal_mentions": pd.DataFrame(mentions),
        "ideas": pd.DataFrame(ideas),
        "idea_mentions": pd.DataFrame(im, columns=["idea_id", "mention_id", "group_id"]),
        "idea_events": pd.DataFrame(
            [
                {
                    "idea_id": i,
                    "event_type": t,
                    "event_date": D(d),
                    "title": f"{t} for {i}",
                    "source_url": "https://www.gov.uk/x",
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
                for i, b in [("nlp-00001", 101), ("nlp-00002", None)]
            ]
        ),
        "bills": pd.DataFrame({"bill_id": [101], "short_title": ["Example Bill"]}),
        "stage_sittings": pd.DataFrame(
            [
                {
                    "sitting_id": 1,
                    "bill_id": 101,
                    "api_stage_id": 6,
                    "house": "Commons",
                    "event_date": D("2016-03-01"),
                }
            ]
        ),
        "sessions": pd.DataFrame(
            [
                {"session_id": 23, "parliament_id": 4, "start_date": D("2015-05-27")},
                {"session_id": 24, "parliament_id": 5, "start_date": D("2017-06-21")},
            ]
        ),
        "parliaments": pd.DataFrame(
            app_db.DEFAULT_PARLIAMENTS,
            columns=["parliament_id", "first_meeting_date", "dissolution_date", "election_date"],
        ),
        "sitting_days": pd.DataFrame(
            {"date": [d.date() for d in sitting_days], "house": "Commons"}
        ),
    }


def make_model() -> tm.TakeUpModel:
    theta = [0.2, -0.5, 0.4, -3.0, -3.2, -3.5, -4.0, -4.5, -5.0, 0.8, 1.0]
    return tm.TakeUpModel.from_params(theta, ZCOLS, XCOLS, Sigma=np.eye(11) * 0.01)


@pytest.fixture
def service():
    svc = api.TakeupService(TakeupData(tables()), make_model(), run_id=7)
    app.dependency_overrides[api.get_takeup_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(api.get_takeup_service, None)


async def _get(path: str, **params) -> tuple[int, dict]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(path, params=params)
    return r.status_code, r.json()


def _dates_in(obj) -> list[date]:
    """Every ISO date string value in a JSON object (keys ending in 'date' or 'as_of')."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and (k.endswith("date") or k == "date"):
                out.append(D(v))
            else:
                out += _dates_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _dates_in(v)
    return out


async def test_ideas_after_merge(service):
    code, data = await _get("/api/v1/takeup/ideas", as_of="2016-06-01")
    assert code == 200
    assert data["run_id"] == 7 and data["as_of"] == "2016-06-01"
    by_id = {i["signal_id"]: i for i in data["ideas"]}
    assert set(by_id) == {"nlp-00001", "nlp-00002:g3"}  # nlp-00003 not yet mentioned
    a, b = by_id["nlp-00001"], by_id["nlp-00002:g3"]
    assert a["taken_up"] and a["linked_bill"] == {
        "bill_id": 101,
        "title": "Example Bill",
        "first_reading_date": "2016-03-01",
    }
    assert a["topic"] == "idea nlp-00001" and a["first_source"] == "wms"
    assert not b["taken_up"] and b["linked_bill"] is None
    for key in ("pi", "pi_now", "p26", "p52"):
        assert 0 <= b[key] <= 1
    assert b["verdict"] in {"likely", "unlikely", "uncertain"}
    assert b["t_50"] > b["weeks_elapsed"]
    assert D(b["t_50_date"]) > D("2016-05-25")
    assert b["forecast_source"] == "computed"


async def test_ideas_before_merge_are_split_and_hide_the_bill(service):
    code, data = await _get("/api/v1/takeup/ideas", as_of="2016-01-01")
    assert code == 200
    ids = {i["signal_id"] for i in data["ideas"]}
    assert ids == {"nlp-00001:g1", "nlp-00001:g2", "nlp-00002:g3"}
    assert all(i["linked_bill"] is None and not i["taken_up"] for i in data["ideas"])
    srcs = {i["signal_id"]: i["first_source"] for i in data["ideas"]}
    assert srcs["nlp-00001:g2"] == "hansard_spoken"


async def test_ideas_before_any_mention_is_empty(service):
    code, data = await _get("/api/v1/takeup/ideas", as_of="2010-01-01")
    assert code == 200 and data["count"] == 0 and data["ideas"] == []


async def test_forecast_fields_and_as_of_filter(service):
    code, f = await _get("/api/v1/takeup/forecast", idea_id="nlp-00002", as_of="2016-06-01")
    assert code == 200
    for key in (
        "pi",
        "pi_lo",
        "pi_hi",
        "verdict",
        "weeks_elapsed",
        "pi_now",
        "p26",
        "p52",
        "t_10",
        "t_50",
        "t_90",
        "t_10_date",
        "t_50_date",
        "t_90_date",
        "t_10_sitting",
        "t_50_sitting",
        "t_90_sitting",
        "events",
        "mentions",
        "linked_bill",
        "run_id",
    ):
        assert key in f
    assert f["idea_id"] == "nlp-00002" and f["signal_id"] == "nlp-00002:g3"
    assert f["pi_lo"] <= f["pi"] <= f["pi_hi"]
    assert f["weeks_elapsed"] < f["t_10"] <= f["t_50"] <= f["t_90"]
    assert f["t_10_sitting"] <= f["t_10"]
    # nothing dated after as_of: the 2017 mention and the 2016-09 consultation are hidden
    assert [m["mention_id"] for m in f["mentions"]] == ["wms-3-0"]
    assert [e["event_type"] for e in f["events"]] == ["white_paper"]
    assert f["mentions"][0]["sentence"].startswith("We will legislate")
    timing = {"t_10_date", "t_50_date", "t_90_date"}
    others = {k: v for k, v in f.items() if k not in timing}
    assert all(d <= D("2016-06-01") for d in _dates_in(others))


async def test_forecast_split_idea_uses_anchor_group_only(service):
    code, f = await _get("/api/v1/takeup/forecast", idea_id="nlp-00001", as_of="2016-01-01")
    assert code == 200
    assert f["signal_id"] == "nlp-00001:g1"
    assert sorted(f["signal_ids"]) == ["nlp-00001:g1", "nlp-00001:g2"]
    assert [m["mention_id"] for m in f["mentions"]] == ["wms-1-0"]
    assert f["linked_bill"] is None
    # a signal_id can be asked for directly
    code, g = await _get("/api/v1/takeup/forecast", idea_id="nlp-00001:g2", as_of="2016-01-01")
    assert code == 200 and [m["mention_id"] for m in g["mentions"]] == ["han-2-0"]


async def test_forecast_linked_bill_once_first_read(service):
    code, f = await _get("/api/v1/takeup/forecast", idea_id="nlp-00001", as_of="2016-06-01")
    assert code == 200
    assert f["taken_up"] and f["linked_bill"]["bill_id"] == 101
    assert {m["mention_id"] for m in f["mentions"]} == {"wms-1-0", "han-2-0"}


async def test_unknown_or_invisible_idea_is_404(service):
    code, _ = await _get("/api/v1/takeup/forecast", idea_id="nlp-99999", as_of="2016-06-01")
    assert code == 404
    code, _ = await _get("/api/v1/takeup/forecast", idea_id="nlp-00003", as_of="2016-06-01")
    assert code == 404
    code, _ = await _get("/api/v1/takeup/history", idea_id="nlp-00003", as_of="2016-06-01")
    assert code == 404
    code, f = await _get("/api/v1/takeup/forecast", idea_id="nlp-00003", as_of="2018-06-01")
    assert code == 200 and f["signal_id"] == "nlp-00003:g4"


async def test_bad_as_of_is_422(service):
    code, _ = await _get("/api/v1/takeup/ideas", as_of="not-a-date")
    assert code == 422


async def test_history_weekly_sequence(service):
    code, h = await _get("/api/v1/takeup/history", idea_id="nlp-00002", as_of="2016-06-01")
    assert code == 200
    pts = h["points"]
    assert [p["week"] for p in pts] == list(range(len(pts)))
    assert pts[0]["date"] == h["trigger_date"] == "2015-09-01"
    assert all(D(p["date"]) <= D("2016-06-01") for p in pts)
    assert len(pts) - 1 == (D("2016-06-01") - D("2015-09-01")).days // 7
    assert all(0 <= p["pi_now"] <= 1 and 0 <= p["p52"] <= 1 for p in pts)
    # pi_now falls in silent weeks without new events
    assert pts[5]["pi_now"] <= pts[0]["pi_now"]
    kinds = [(m["kind"], m["date"]) for m in h["markers"]]
    assert kinds == [("mention", "2015-09-01"), ("white_paper", "2016-01-15")]
    # the last point matches the forecast at as_of
    _, f = await _get("/api/v1/takeup/forecast", idea_id="nlp-00002", as_of="2016-06-01")
    assert pts[-1]["week"] == f["weeks_elapsed"]
    assert pts[-1]["pi_now"] == pytest.approx(f["pi_now"])
    assert pts[-1]["p52"] == pytest.approx(f["p52"])


async def test_history_stops_before_the_bill(service):
    code, h = await _get("/api/v1/takeup/history", idea_id="nlp-00001", as_of="2016-06-01")
    assert code == 200 and h["taken_up"]
    bill = [m for m in h["markers"] if m["kind"] == "first_reading"]
    assert len(bill) == 1 and bill[0]["date"] == "2016-03-01"
    assert h["points"][-1]["week"] < bill[0]["week"]


async def test_stored_forecast_preferred(service):
    def lookup(signal_ids, as_of):
        assert as_of == D("2016-06-01")
        return {"nlp-00002:g3": {"pi_now": 0.123, "verdict": "unlikely"}}

    service.stored_lookup = lookup
    code, f = await _get("/api/v1/takeup/forecast", idea_id="nlp-00002", as_of="2016-06-01")
    assert code == 200
    assert f["forecast_source"] == "stored"
    assert f["pi_now"] == 0.123 and f["verdict"] == "unlikely"


def test_model_from_run_round_trip():
    model = make_model()
    tab = model.parameter_table()
    run = {
        "run_id": 1,
        "features": ",".join(ZCOLS + XCOLS),
        "parameters": json.dumps(
            {r.parameter: {"estimate": r.estimate, "std_err": r.std_err} for r in tab.itertuples()}
        ),
    }
    rebuilt = api.model_from_run(run)
    assert rebuilt.zcols == ZCOLS and rebuilt.xcols == XCOLS
    np.testing.assert_allclose(rebuilt.theta, model.theta)
    z, x = np.array([0.0, 1.0]), np.array([10.0, tm.T_MAX + 1])
    a, b = model.predict(z, x, weeks_elapsed=5), rebuilt.predict(z, x, weeks_elapsed=5)
    assert a["pi_now"] == pytest.approx(b["pi_now"]) and a["t_50"] == b["t_50"]
