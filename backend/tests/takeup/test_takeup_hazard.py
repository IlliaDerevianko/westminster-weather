"""Offline tests of the hazard model (takeup/hazard.py), CQR windows (takeup/conformal.py)
and the three-model backtest in takeup/evaluate.py, on synthetic data."""

import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from westminster_weather.takeup import conformal as cf
from westminster_weather.takeup import evaluate as ev
from westminster_weather.takeup import hazard as hzm
from westminster_weather.takeup import model as tm

AS_OF = pd.Timestamp("2025-12-31")
TRUE_A = -2.5  # hazard logit while a mention is recent (0-4 weeks)
TRUE_GAP = -1.5  # added once the last mention is 5+ weeks old
MENTION_RATE = 0.06  # weekly chance of a new mention


def all_sitting(as_of=AS_OF) -> np.ndarray:
    return pd.date_range("2005-01-01", as_of).to_numpy().astype("datetime64[D]")


def make_recency(n: int = 700, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signals whose weekly hazard depends only on the age of the last mention known at the
    start of the week; new mentions arrive at random days within weeks."""
    rng = np.random.default_rng(seed)
    sigs, mentions = [], []
    for i in range(n):
        start = pd.Timestamp("2010-01-04") + pd.Timedelta(days=int(rng.integers(0, 365 * 12)))
        sid = f"s{i:04d}"
        mdays = [0]  # day offsets from the start
        bill = None
        for t in range(1, tm.T_MAX + 1):
            ws = 7 * (t - 1)
            if start + pd.Timedelta(days=int(ws)) > AS_OF:
                break
            gap_weeks = np.floor((ws - max(d for d in mdays if d <= ws)) / 7)
            h = TRUE_A + (TRUE_GAP if gap_weeks >= 5 else 0.0)
            if rng.random() < 1 / (1 + np.exp(-h)):
                bill = start + pd.Timedelta(days=int(ws) + int(rng.integers(0, 7)))
                break
            if rng.random() < MENTION_RATE:
                mdays.append(ws + int(rng.integers(1, 7)))
        if bill is not None and bill > AS_OF:
            bill = None
        sigs.append(
            {
                "signal_id": sid,
                "idea_id": sid,
                "title": sid,
                "signal_date": start,
                "bill_id": i if bill is not None else None,
                "bill_date": bill if bill is not None else pd.NaT,
                "data_end": AS_OF,
                "z_src_hansard": float(rng.random() < 0.5),
            }
        )
        for d in mdays:
            md = start + pd.Timedelta(days=int(d))
            if md <= AS_OF:
                mentions.append({"signal_id": sid, "event_date": md, "firmness": 1})
    sig = pd.DataFrame(sigs)
    sig["bill_date"] = pd.to_datetime(sig["bill_date"])
    return sig, pd.DataFrame(mentions)


@pytest.fixture(scope="module")
def recency():
    sig, tl = make_recency()
    model = hzm.HazardModel().fit(sig, tl, all_sitting(), AS_OF)
    return sig, tl, model


# ----------------------------------------------------------------------------- rows


def test_rows_use_nothing_after_week_start(recency):
    sig, tl, model = recency
    rows = model.build_rows(sig.head(80), tl, all_sitting(), AS_OF)
    recs = {r["signal_id"]: r for r in sig.head(80).to_dict("records")}
    rng = np.random.default_rng(0)
    for k in rng.choice(len(rows["y"]), 300, replace=False):
        sid, week = rows["signal_id"][k], int(rows["week"][k])
        r = recs[sid]
        ws = r["signal_date"] + pd.Timedelta(weeks=int(week) - 1)
        own = tl[(tl["signal_id"] == sid) & (tl["event_date"] <= ws)]
        days = (own["event_date"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy(float)
        x = model.week_features(r, days, own["firmness"].to_numpy(float), np.array([week]), np.inf)
        np.testing.assert_allclose(rows["X"][k], x[0])


def test_future_mentions_do_not_change_rows():
    sig, tl = make_recency(60, seed=9)
    m = hzm.HazardModel(c_grid=(1.0,)).fit(sig, tl, all_sitting(), AS_OF)
    cut = pd.Timestamp("2016-01-01")
    early = sig[sig["signal_date"] < cut].copy()
    early["data_end"] = cut
    early.loc[early["bill_date"] > cut, "bill_date"] = pd.NaT
    a = m.build_rows(early, tl, all_sitting(cut), cut)
    b = m.build_rows(early, tl[tl["event_date"] <= cut], all_sitting(cut), cut)
    np.testing.assert_array_equal(a["X"], b["X"])
    np.testing.assert_array_equal(a["y"], b["y"])


def test_non_sitting_weeks_have_no_rows():
    start = pd.Timestamp("2015-01-05")  # a Monday
    sig = pd.DataFrame(
        {
            "signal_id": ["a"],
            "title": ["a"],
            "signal_date": [start],
            "bill_date": [start + pd.Timedelta(weeks=20, days=2)],  # week 21
            "data_end": [AS_OF],
            "z_src_hansard": [0.0],
        }
    )
    days = pd.bdate_range("2014-01-01", AS_OF)
    recess = (days >= start + pd.Timedelta(weeks=5)) & (days < start + pd.Timedelta(weeks=9))
    sitting = days[~recess].to_numpy().astype("datetime64[D]")
    m = hzm.HazardModel()
    m.elections = m.session_starts = np.array([])
    m._set_columns(sig)
    rows = m.build_rows(sig, None, sitting, AS_OF)
    assert set(rows["week"]) == set(range(1, 22)) - {6, 7, 8, 9}
    assert (rows["w"] == 1).all()
    assert rows["y"].sum() == 1 and rows["y"][rows["week"] == 21][0] == 1


def test_censoring_respected():
    start = pd.Timestamp("2023-01-02")
    dissolution = pd.Timestamp("2024-05-30")
    sig = pd.DataFrame(
        {
            "signal_id": ["a", "b"],
            "title": ["a", "b"],
            "signal_date": [start, start],
            "bill_date": [pd.NaT, pd.NaT],  # a bill after the dissolution is not visible
            "data_end": [dissolution, AS_OF],
            "z_src_hansard": [0.0, 1.0],
        }
    )
    tl = pd.DataFrame(
        {
            "signal_id": ["a", "a", "b"],
            "event_date": [start, pd.Timestamp("2024-09-01"), start],
            "firmness": [1, 2, 1],
        }
    )
    m = hzm.HazardModel()
    m._set_columns(sig)
    rows = m.build_rows(sig, tl, all_sitting(), AS_OF)
    wa = rows["week"][rows["signal_id"] == "a"]
    assert wa.max() == (dissolution - start).days // 7
    assert rows["y"].sum() == 0
    # the post-dissolution firm mention is never seen by signal a's rows
    j = m.feature_names.index("any_firm")
    assert rows["X"][rows["signal_id"] == "a", j].max() == 0
    wb = rows["week"][rows["signal_id"] == "b"]
    assert wb.max() == min((AS_OF - start).days // 7, tm.T_MAX)


# ------------------------------------------------------------------------- predict


@pytest.mark.parametrize("weeks_ago", [0, 10, 60, 200])
def test_predict_outputs(recency, weeks_ago):
    sig, tl, model = recency
    r = sig.iloc[3].to_dict()  # its bill (if any) is after as_of or ignored
    as_of = r["signal_date"] + pd.Timedelta(weeks=weeks_ago, days=3)
    p = model.predict(r, tl[tl["signal_id"] == r["signal_id"]], as_of, all_sitting(as_of))
    s = p["weeks_elapsed"]
    assert s == weeks_ago
    for k in ("pi", "pi_lo", "pi_hi", "pi_now", "pi_now_lo", "pi_now_hi", "p13", "p26", "p52"):
        assert 0 <= p[k] <= 1, k
    assert p["p13"] <= p["p26"] <= p["p52"] <= p["pi_now"] + 1e-12
    assert p["pi_now_lo"] <= p["pi_now"] <= p["pi_now_hi"]
    assert p["verdict"] in ("likely", "unlikely", "uncertain")
    qs = [p["t_10"], p["t_50"], p["t_90"]]
    assert all(q is None or q > s for q in qs)
    known = [q for q in qs if q is not None]
    assert known == sorted(known)


def test_silence_lowers_pi_now(recency):
    sig, tl, model = recency
    r = sig.iloc[0].to_dict()
    only_trigger = tl[(tl["signal_id"] == r["signal_id"]) & (tl["event_date"] == r["signal_date"])]
    vals = [
        model.predict(r, only_trigger, r["signal_date"] + pd.Timedelta(weeks=w), all_sitting())[
            "pi_now"
        ]
        for w in (0, 26, 104)
    ]
    assert vals[0] > vals[1] > vals[2]


def test_recency_effect_recovered(recency):
    _, _, model = recency
    tab = model.parameter_table().set_index("parameter")["estimate"]
    gaps = tab[[f"gap_{e}+" for e in hzm.GAP_EDGES[1:]]]
    # all gap buckets share the true -1.5 effect (relative to 0-4 weeks)
    assert -2.4 < gaps.iloc[:3].mean() < -0.8
    assert (gaps.iloc[:3] < 0).all()
    assert abs(tab["z_src_hansard"]) < 0.4  # no true effect


def test_no_new_mentions_assumed_in_future(recency):
    """A later mention (after as_of) must not change the forecast at as_of."""
    sig, tl, model = recency
    r = sig.iloc[1].to_dict()
    as_of = r["signal_date"] + pd.Timedelta(weeks=8)
    own = tl[(tl["signal_id"] == r["signal_id"]) & (tl["event_date"] <= as_of)]
    extra = pd.DataFrame(
        {
            "signal_id": [r["signal_id"]],
            "event_date": [as_of + pd.Timedelta(days=5)],
            "firmness": [2],
        }
    )
    a = model.predict(r, own, as_of, all_sitting(as_of))
    b = model.predict(r, pd.concat([own, extra]), as_of, all_sitting(as_of))
    assert a == b


# --------------------------------------------------------------------------- CQR


def test_conformal_quantile_by_hand():
    assert cf.conformal_quantile([0.1, 0.2, 0.3, 0.4], alpha=0.2) == pytest.approx(0.4)  # k=4
    assert cf.conformal_quantile([0.1, 0.2, 0.3], alpha=0.2) == np.inf  # k = 4 > n
    scores = np.arange(1, 20) / 10  # n = 19, k = ceil(20 * 0.8) = 16
    assert cf.conformal_quantile(scores, alpha=0.2) == pytest.approx(1.6)
    np.testing.assert_allclose(
        cf.conformity_scores([5, 10, 30], [8, 8, 8], [12, 12, 12]), [0.75, -0.5, 4.5]
    )


def test_widen_keeps_lower_end_after_s():
    lo, hi = cf.widen([10, 3], [20, None], q=0.5, s=[12, 0])
    assert lo == [13, 1] and hi[0] == 25 and hi[1] is None


def test_cqr_reaches_target_coverage():
    rng = np.random.default_rng(1)

    def draw(n):
        centre = rng.uniform(20, 120, n)
        t = np.maximum(1, np.round(centre + rng.normal(0, 0.4 * centre)))
        lo = np.round(centre - 0.1 * centre)  # far too narrow
        hi = np.round(centre + 0.1 * centre)
        return t, lo, hi

    t_cal, lo_cal, hi_cal = draw(400)
    cal = cf.CQRCalibrator().fit(t_cal, lo_cal, hi_cal)
    t, lo, hi = draw(5000)
    raw = ev.window_coverage(t, lo, hi)["coverage"]
    new_lo, new_hi = cal.apply(lo, hi)
    cov = ev.window_coverage(t, new_lo, new_hi)["coverage"]
    assert raw < 0.4
    assert 0.76 <= cov <= 0.86
    assert cal.summary()["n_calibration"] == 400


# ---------------------------------------------------------------- three-model backtest


def _evaluate_module():
    path = Path(__file__).with_name("test_takeup_evaluate.py")
    spec = importlib.util.spec_from_file_location("_takeup_eval_fixtures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fake():
    te = _evaluate_module()
    return te.FakeTakeupData(te.make_truth(400, seed=5)), te.END


def test_backtest_runs_all_models(fake):
    td, end = fake
    bt = ev.backtest(td, date(2019, 1, 1), end, later_points=True)
    m = bt["metrics"]
    for H in ("26", "52"):
        c = m["comparison"][H]
        assert set(c) == {"cure", "hazard", "b0", "b1"}
        assert c["cure"]["n"] == c["hazard"]["n"] == c["b1"]["n"]
        assert c["hazard"]["auc"] is not None
    assert {"raw", "cqr", "calibration"} <= set(m["timing_windows"]["hazard"])
    assert {"raw", "cqr"} <= set(m["timing_windows"]["cure"])
    assert "later_as_of" in m["hazard"]
    t = bt["test"]
    assert ((t["hz_p26"] <= t["hz_p52"] + 1e-12) & (t["hz_p52"] <= 1)).all()


def test_run_evaluation_three_models(fake, tmp_path):
    td, end = fake
    m = ev.run_evaluation(
        td=td, end=end, out_dir=tmp_path, doc_path=tmp_path / "EVAL.md", splits=[date(2019, 1, 1)]
    )
    assert m["model_comparison"]["rows"] and m["model_comparison"]["recommendation"]
    assert "hazard_full_fit" in m
    doc = (tmp_path / "EVAL.md").read_text()
    assert "Model comparison" in doc and "CQR" in doc
    assert (tmp_path / "calibration.png").exists()
