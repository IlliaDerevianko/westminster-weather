"""Offline tests of stage 3 (Royal Assent -> commencement orders), journey/commencement.py,
on synthetic Acts: no leakage past as_of, censoring, conditional quantiles, CQR coverage."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from westminster_weather.journey import commencement as cm

DATA_END = pd.Timestamp("2026-06-30")
DEPTS = ["HM Treasury", "Home Office", "Ministry of Justice", "Department for Transport"]


def make_tables(n: int = 260, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Acts with Royal Assent 2008-2025. Treasury Acts never need orders; others need them
    with probability 0.75 (0.9 if Lords-first); the first order comes after a geometric wait;
    each later week brings another order with probability 0.08 until a stop (prob 0.3 after
    each order)."""
    rng = np.random.default_rng(seed)
    acts, events, bills, sittings = [], [], [], []
    end_day = DATA_END
    for i in range(n):
        bid = 1000 + i
        ra = pd.Timestamp("2008-01-07") + pd.Timedelta(int(rng.integers(0, 365 * 17.5)), unit="D")
        dept = DEPTS[int(rng.integers(0, len(DEPTS)))]
        lords = bool(rng.random() < 0.3)
        first_reading = ra - pd.Timedelta(int(rng.integers(60, 400)), unit="D")
        needs = dept != "HM Treasury" and rng.random() < (0.9 if lords else 0.75)
        orders = []
        if needs:
            t = 1 + int(rng.geometric(0.08))
            while True:
                day = ra + pd.Timedelta(7 * (t - 1) + int(rng.integers(0, 7)), unit="D")
                if day > end_day:
                    break
                orders.append(day)
                if rng.random() < 0.3:
                    break
                t += 1 + int(rng.geometric(0.08))
        acts.append(
            {
                "bill_id": bid,
                "act_title": f"Synthetic Act {i}",
                "royal_assent_date": ra.date(),
                "first_commencement_order_date": orders[0].date() if orders else None,
                "latest_commencement_order_date": orders[-1].date() if orders else None,
                "commencement_orders_count": len(orders),
            }
        )
        events += [{"bill_id": bid, "stage": "royal_assent", "event_date": ra.date()}]
        events += [
            {"bill_id": bid, "stage": "commencement_orders", "event_date": d.date()} for d in orders
        ]
        bills.append(
            {
                "bill_id": bid,
                "originating_house": "Lords" if lords else "Commons",
                "introduced_date": first_reading.date(),
                "department": dept,
            }
        )
        stage = 1 if lords else 6
        sittings.append({"bill_id": bid, "api_stage_id": stage, "event_date": first_reading.date()})
        sittings.append({"bill_id": bid, "api_stage_id": 11, "event_date": ra.date()})
        if rng.random() < 0.4:
            sittings.append({"bill_id": bid, "api_stage_id": 12, "event_date": ra.date()})
    events.append({"bill_id": 1000, "stage": "amended_or_repealed", "event_date": end_day.date()})
    return {
        "act_commencements": pd.DataFrame(acts),
        "post_royal_assent_events": pd.DataFrame(events),
        "bills": pd.DataFrame(bills),
        "stage_sittings": pd.DataFrame(sittings),
        "sessions": pd.DataFrame(
            {"session_id": range(18), "start_date": [date(2007 + k, 11, 6) for k in range(18)]}
        ),
        "parliaments": pd.DataFrame(
            {
                "parliament_id": [2, 3, 4, 5, 6, 7],
                "election_date": [
                    date(2005, 5, 5),
                    date(2010, 5, 6),
                    date(2015, 5, 7),
                    date(2017, 6, 8),
                    date(2019, 12, 12),
                    date(2024, 7, 4),
                ],
            }
        ),
    }


@pytest.fixture(scope="module")
def tables():
    return make_tables()


@pytest.fixture(scope="module")
def data(tables):
    return cm.build_data(tables)


@pytest.fixture(scope="module")
def fc(data):
    return cm.CommencementForecaster(data, "2019-01-01")


@pytest.fixture(autouse=True)
def fewer_sims(monkeypatch):
    monkeypatch.setattr(cm, "N_SIMS", 300)


def _act_with(data, pred):
    for _, a in data.acts.iterrows():
        if pred(a, data.orders.get(int(a["bill_id"]), np.array([]))):
            return a
    raise AssertionError("no such synthetic Act")


def test_build_data_features(data, tables):
    assert data.data_end == DATA_END
    assert len(data.acts) == len(tables["act_commencements"])
    a = data.acts.iloc[0]
    row = tables["bills"].set_index("bill_id").loc[a["bill_id"]]
    assert a["lords_first"] == float(row["originating_house"] == "Lords")
    assert 0 <= a["months_into_session"] < 13 and 0 <= a["years_since_election"] < 5.5
    assert a["days_1r_to_ra"] > 0 and a["n_stage_sittings"] >= 2
    assert data.acts["draft_regs_by_ra"].sum() == 0  # none in the synthetic data
    for days in data.orders.values():
        assert np.all(np.diff(days) >= 0)


def test_fit_uses_nothing_after_the_cut(tables, data):
    """Changing only orders dated after the cut leaves the fitted models unchanged."""
    cut = pd.Timestamp("2016-01-01")
    t2 = {k: v.copy() for k, v in tables.items()}
    ev = t2["post_royal_assent_events"]
    late = pd.to_datetime(ev["event_date"]) > cut
    ev = ev[~late | (ev["stage"] != "commencement_orders")]
    extra = [
        {"bill_id": int(b), "stage": "commencement_orders", "event_date": date(2018, 3, 1)}
        for b in data.acts["bill_id"].iloc[:40]
    ]
    t2["post_royal_assent_events"] = pd.concat([ev, pd.DataFrame(extra)], ignore_index=True)
    d2 = cm.build_data(t2)
    a = cm.CommencementForecaster(data, cut, calibrate=False)
    b = cm.CommencementForecaster(d2, cut, calibrate=False)
    np.testing.assert_allclose(a.cure.theta, b.cure.theta, atol=1e-8)
    np.testing.assert_allclose(a.orders.coef, b.orders.coef, atol=1e-8)
    assert a.orders.n_rows == b.orders.n_rows


def test_forecast_ignores_events_after_as_of(data, fc):
    act = _act_with(data, lambda a, o: len(o) >= 3 and a["ra_day"] >= cm._day("2019-01-01"))
    bid = int(act["bill_id"])
    as_of = cm._date(data.orders[bid][1])  # two orders seen
    f1 = cm.forecast_act(bid, as_of, data=data, forecaster=fc)
    saved = data.orders[bid]
    try:
        data.orders[bid] = np.append(saved[:2], [saved[1] + 3, saved[1] + 900.0])
        f2 = cm.forecast_act(bid, as_of, data=data, forecaster=fc)
    finally:
        data.orders[bid] = saved
    assert f1 == f2
    assert f1["orders_seen"] == 2


def test_censoring_rules(data, fc):
    end = cm._day(data.data_end)
    recent = _act_with(data, lambda a, o: len(o) and o[-1] > end - 7 * 50)
    out = cm._outcomes(data, recent, end)
    assert out["ever"] == 1 and out["T"] is None  # more orders may come
    young = _act_with(data, lambda a, o: not len(o) and a["ra_day"] > end - 7 * 100)
    assert cm._outcomes(data, young, end)["ever"] is None
    # the cure model sees a first order after the cut as no order (censored at the cut)
    train = data.acts[data.acts["ra_day"] < fc.cut_day]
    Z = fc.static.transform(train)
    df = cm._cure_frame(data, train, Z, fc.cut_day)
    assert (pd.to_datetime(df["bill_date"]).dropna() <= fc.cut).all()
    later = [
        k
        for k, b in enumerate(train["bill_id"])
        if len(data.orders.get(int(b), [])) and data.orders[int(b)][0] > fc.cut_day
    ]
    assert later and df["bill_date"].iloc[later].isna().all()


def test_quantiles_conditional_on_elapsed(data, fc):
    act = _act_with(
        data,
        lambda a, o: (
            a["ra_day"] >= cm._day("2019-01-01") and (not len(o) or o[0] > a["ra_day"] + 7 * 40)
        ),
    )
    bid, ra = int(act["bill_id"]), act["royal_assent_date"]
    f0 = cm.forecast_act(bid, ra, data=data, forecaster=fc)
    f30 = cm.forecast_act(bid, ra + pd.Timedelta(7 * 30, unit="D"), data=data, forecaster=fc)
    assert f30["weeks_since_royal_assent"] == 30
    for block in ("raw", "calibrated"):
        fo = f30["first_order"][block]
        assert fo["t_10"] > 30 and fo["t_10"] <= fo["t_50"]
        assert fo["t_90"] is None or fo["t_50"] <= fo["t_90"]
        fin = f30["orders_finished"][block]
        assert fin["t_10"] >= 32 and fin["t_10"] <= fin["t_50"]
    # long silence lowers the chance that orders are needed at all
    f150 = cm.forecast_act(bid, ra + pd.Timedelta(7 * 150, unit="D"), data=data, forecaster=fc)
    if not f150["first_order"]["observed"]:
        assert f150["needs_orders"]["p"] < f30["needs_orders"]["p"] < f0["needs_orders"]["p"]


def test_after_first_order(data, fc):
    act = _act_with(data, lambda a, o: len(o) >= 2 and a["ra_day"] >= cm._day("2019-01-01"))
    bid = int(act["bill_id"])
    first = data.orders[bid][0]
    f = cm.forecast_act(bid, cm._date(first + 7), data=data, forecaster=fc)
    L0 = f["orders_finished"]["latest_order_week"]
    assert f["needs_orders"]["p"] == 1.0 and f["first_order"]["observed"]
    assert f["first_order"]["week"] == L0 == cm._week(first, act["ra_day"])
    for block in ("raw", "calibrated"):
        q = f["orders_finished"][block]
        assert q["t_10"] >= L0 and q["t_10"] <= q["t_50"]
    assert 0 < f["orders_finished"]["p_finished_already"] < 1
    # with a long silence since the latest order, 'finished' becomes more likely
    g = cm.forecast_act(bid, cm._date(first + 7 * 120), data=data, forecaster=fc)
    if g["orders_seen"] == 1:
        assert (
            g["orders_finished"]["p_finished_already"] > f["orders_finished"]["p_finished_already"]
        )


def test_forecast_act_json_and_errors(data, fc):
    act = data.acts.iloc[-1]
    f = cm.forecast_act(int(act["bill_id"]), DATA_END.date(), data=data, forecaster=fc)
    json.dumps(f, default=str)
    for key in ("needs_orders", "first_order", "orders_finished", "definitions", "model_cut"):
        assert key in f
    assert set(f["orders_finished"]["p_by_week"]) == {"52", "104", "208"}
    with pytest.raises(ValueError):
        cm.forecast_act(int(act["bill_id"]), date(2000, 1, 1), data=data, forecaster=fc)
    with pytest.raises(KeyError):
        cm.forecast_act(1, DATA_END.date(), data=data, forecaster=fc)


def test_forecaster_for_quarter_cut_capped(data, monkeypatch):
    made = []

    class Fake:
        def __init__(self, d, cut):
            made.append(cut)

    monkeypatch.setattr(cm, "CommencementForecaster", Fake)
    monkeypatch.setattr(cm, "_FORECASTERS", {})
    cm.forecaster_for(data, date(2020, 8, 17))
    cm.forecaster_for(data, date(2020, 9, 30))  # same quarter: cached
    cm.forecaster_for(data, date(2030, 1, 1))  # after the data end
    assert made == [date(2020, 7, 1), DATA_END.date()]


def test_window_calibrator_coverage():
    """Exchangeable synthetic windows: CQR brings 80% coverage from ~50% back to ~80%."""
    rng = np.random.default_rng(0)

    def draw(n):
        mid = rng.integers(20, 200, n)
        t = mid + rng.normal(0, 30, n)
        return np.clip(np.round(t), 1, None), mid - 20, mid + 20

    t, lo, hi = draw(400)
    cal = cm.WindowCalibrator(cm.FINISH_HORIZON).fit(t, lo, hi)
    assert cal.usable and cal.q > 0 and cal.raw_coverage < 0.7
    t2, lo2, hi2 = draw(2000)
    raw = np.mean((lo2 <= t2) & (t2 <= hi2))
    win = [cal.apply(a, b, 1) for a, b in zip(lo2, hi2, strict=True)]
    cov = np.mean(
        [lo <= x <= (np.inf if hi is None else hi) for x, (lo, hi) in zip(t2, win, strict=True)]
    )
    assert raw < 0.7 and 0.75 <= cov <= 0.87
    few = cm.WindowCalibrator(cm.FINISH_HORIZON).fit(t[:5], lo[:5], hi[:5])
    assert not few.usable and few.apply(10, 20, 1) == (10, 20)


def test_calibration_fitted_and_flagged(fc):
    assert fc.first_cal.n >= cm.MIN_CALIBRATION
    s = fc.summary()
    assert s["first_order_cqr"]["n_calibration"] == fc.first_cal.n
    assert s["n_train_acts"] > 100


def test_backtest_writes_reports(data, tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "BACKTEST_OFFSETS", (0, 26))
    table, metrics = cm.backtest(data, offsets=(0, 26))
    assert set(table["target"]) >= {"needs_orders", "first_order_week"}
    assert (pd.to_datetime(table["as_of"]) >= pd.to_datetime(table["royal_assent_date"])).all()
    assert (pd.to_datetime(table["royal_assent_date"]) >= pd.Timestamp(cm.TRAIN_CUT)).all()
    m0 = metrics["by_offset"]["0"]
    assert m0["needs_orders"]["model"]["auc"] > 0.6
    assert {"coverage_80_raw", "coverage_80_cqr", "mae_model_p50"} <= set(m0["first_order_week"])
    monkeypatch.setattr(cm, "backtest", lambda d: (table, metrics))
    cm.write_backtest(data, tmp_path)
    assert (tmp_path / "stage3_backtest.csv").exists()
    assert json.loads((tmp_path / "stage3_metrics.json").read_text())["train_cut"] == "2019-01-01"
