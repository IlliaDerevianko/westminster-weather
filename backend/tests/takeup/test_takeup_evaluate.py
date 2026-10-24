"""Offline tests of takeup/evaluate.py on synthetic signals with known parameters."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select

from westminster_weather.takeup import evaluate as ev
from westminster_weather.takeup import model as tm
from westminster_weather.takeup.tables import takeup_forecasts, takeup_metadata, takeup_runs

TRUE_GAMMA0 = -0.5
TRUE_GAMMA = {"z_src_hansard": -1.2, "z_initial_firmness": 0.8}
TRUE_ALPHA = -3.0
TRUE_BETA = {"x_consultation_response": 1.2, "x_draft_bill": 1.5}
END = pd.Timestamp("2025-12-31")


def make_truth(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Ideas with their full true histories (ported from make_synthetic_signals.py)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        start = pd.Timestamp("2012-01-02") + pd.Timedelta(days=int(rng.integers(0, 365 * 13)))
        z = {
            "z_src_hansard": float(rng.random() < 0.5),
            "z_initial_firmness": float(rng.integers(0, 3)),
        }
        eta = TRUE_GAMMA0 + sum(TRUE_GAMMA[k] * v for k, v in z.items())
        taken_up = rng.random() < 1 / (1 + np.exp(-eta))
        ev_week = {k: (int(rng.integers(1, 60)) if rng.random() < 0.5 else None) for k in TRUE_BETA}
        bill_week = None
        if taken_up:
            for t in range(1, tm.T_MAX + 1):
                h = TRUE_ALPHA + sum(
                    TRUE_BETA[k] for k, w in ev_week.items() if w is not None and t >= w
                )
                if rng.random() < 1 / (1 + np.exp(-h)):
                    bill_week = t
                    break

        def to_date(w, start=start):
            return start + pd.Timedelta(weeks=w - 1) if w else pd.NaT

        rows.append(
            {
                "signal_id": f"nlp-{i:05d}",
                "idea_id": f"nlp-{i:05d}",
                "title": f"Synthetic idea {i}",
                "origin": "nlp",
                "signal_date": start,
                "bill_id": i if bill_week else None,
                "bill_date": to_date(bill_week),
                **z,
                **{k: to_date(w) for k, w in ev_week.items()},
            }
        )
    return pd.DataFrame(rows)


class FakeTakeupData:
    """TakeupData-like view of ``truth`` as of a date; every week is a sitting week."""

    def __init__(self, truth: pd.DataFrame) -> None:
        self.truth = truth
        self.calls: list[pd.Timestamp] = []
        self.dropped_features: dict[str, int] = {}

    def get_takeup_signals(self, as_of, features=None) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of).normalize()
        self.calls.append(as_of)
        df = self.truth[self.truth["signal_date"] <= as_of].copy()
        df.loc[df["bill_date"] > as_of, ["bill_date", "bill_id"]] = [pd.NaT, None]
        for c in TRUE_BETA:
            df.loc[df[c] > as_of, c] = pd.NaT
        df["data_end"] = as_of
        if features is not None:
            df = df[[c for c in df if not c.startswith(("z_", "x_"))] + list(features)]
        return df.reset_index(drop=True)

    def sitting_dates(self, as_of) -> np.ndarray:
        days = pd.bdate_range("2010-01-01", pd.Timestamp(as_of))
        return days.to_numpy().astype("datetime64[D]")

    def week_probability(self, as_of) -> pd.Series:
        return pd.Series(1.0, index=pd.RangeIndex(1, 54, name="iso_week"))


@pytest.fixture(scope="module")
def truth() -> pd.DataFrame:
    return make_truth()


@pytest.fixture(scope="module")
def bt(truth):
    return ev.backtest(FakeTakeupData(truth), date(2019, 1, 1), END, later_points=False)


# ------------------------------------------------------------------ hand-checked metrics


def test_binary_metrics_by_hand():
    m = ev.binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert m["n"] == 4 and m["n_bill"] == 2
    assert m["auc"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)
    assert ev.binary_metrics([1, 1], [0.2, 0.3])["auc"] is None


def test_calibration_table_by_hand():
    tab = ev.calibration_table([0, 1, 1, 0, 1], [0.05, 0.15, 0.5, 0.55, 1.0])
    assert [r["n"] for r in tab] == [2, 0, 2, 0, 1]
    assert tab[0]["mean_predicted"] == pytest.approx(0.10)
    assert tab[0]["observed"] == pytest.approx(0.5)
    assert tab[1]["observed"] is None
    assert tab[4]["observed"] == 1.0


def test_window_coverage_by_hand():
    out = ev.window_coverage([5, 10, 30, 300], [3, 11, 20, 50], [8, 20, None, 200])
    assert out == {"n": 4, "coverage": 0.5}


def test_verdict_accuracy_by_hand():
    out = ev.verdict_accuracy(
        ["likely", "likely", "unlikely", "uncertain", "unlikely"], [1, 0, 0, 1, 0]
    )
    assert out["likely_accuracy"] == 0.5 and out["n_likely"] == 2
    assert out["unlikely_accuracy"] == 1.0 and out["n_unlikely"] == 2
    assert out["share_uncertain"] == pytest.approx(0.2)


def test_concordance_perfect_and_reversed():
    times, events = [1, 2, 3, 4], [1, 1, 1, 0]
    assert ev.concordance(times, [1, 2, 3, 4], events)["c_index"] == pytest.approx(1.0)
    assert ev.concordance(times, [4, 3, 2, 1], events)["c_index"] == pytest.approx(0.0)


def test_outcome_labels_and_baselines_by_hand():
    d = {
        "delta": np.array([1, 1, 0, 0, 0]),
        "t": np.array([10, 40, 60, 20, 30]),
        "weeks_observed": np.array([10, 40, 60, 20, 30]),
    }
    known, y = ev.outcome_labels(d, 26)
    assert known.tolist() == [True, True, True, False, True]
    assert y.tolist() == [True, False, False, False, False]
    train = pd.DataFrame({"z_src_hansard": [0.0, 1.0, 1.0, 0.0, 0.0]})
    rates = ev.baseline_rates(train, d, 26)
    assert rates["b0"] == pytest.approx(1 / 4)
    assert rates["b1"]["written_statement"] == {"n": 2, "rate": 0.5}
    assert rates["b1"]["hansard"] == {"n": 2, "rate": 0.0}
    b0, b1 = ev.apply_baselines(pd.DataFrame({"z_src_hansard": [1.0, 0.0]}), rates)
    assert b0.tolist() == [0.25, 0.25] and b1.tolist() == [0.0, 0.5]


# -------------------------------------------------------------------------- backtest


def test_no_leakage_training_sees_only_the_cut(truth):
    """Changing anything after the cut leaves the training fit unchanged."""
    cut = pd.Timestamp("2019-01-01")
    td = FakeTakeupData(truth)
    a = ev.backtest(td, cut, END, later_points=False)
    assert td.calls[0] == cut  # first read is the training frame at the cut
    assert (a["train"]["signal_date"] < cut).all()
    for col in ["bill_date", "data_end", *TRUE_BETA]:
        assert not (pd.to_datetime(a["train"][col]) > cut).any()

    future = truth.copy()
    late = future["signal_date"] < cut
    future.loc[late, "bill_date"] = future.loc[late, "bill_date"].where(
        future.loc[late, "bill_date"] <= cut, cut + pd.Timedelta(days=3)
    )
    future.loc[future["signal_date"] >= cut, "z_initial_firmness"] = 2.0
    b = ev.backtest(FakeTakeupData(future), cut, END, later_points=False)
    np.testing.assert_allclose(a["model"].theta, b["model"].theta)


def test_trigger_forecast_ignores_the_outcome(truth, bt):
    """A test idea's forecast at its trigger does not depend on its bill or later events."""
    model, test = bt["model"], bt["test"]
    row = FakeTakeupData(truth).get_takeup_signals(END)
    row = row[row["signal_id"] == test["signal_id"].iloc[0]].reset_index(drop=True)
    altered = row.copy()
    altered["bill_date"] = altered["signal_date"] + pd.Timedelta(weeks=3)
    for col in TRUE_BETA:
        altered[col] = altered["signal_date"] + pd.Timedelta(weeks=1)
    sit = FakeTakeupData(truth).sitting_dates(END)
    p1 = ev.predict_at_trigger(model, row, sit, None)
    p2 = ev.predict_at_trigger(model, altered, sit, None)
    assert p1["p52"].iloc[0] == pytest.approx(p2["p52"].iloc[0])
    assert p1["weeks_elapsed"].iloc[0] == 0


def test_backtest_beats_baselines_on_synthetic(bt):
    m = bt["metrics"]
    assert m["n_test"] == len(bt["test"]) > 50
    for H in ("26", "52"):
        h = m["horizons"][H]
        assert h["model"]["auc"] > max(0.55, h["b1"]["auc"])
        assert h["model"]["brier"] <= h["b0"]["brier"] + 0.005
        assert h["b0"]["auc"] in (None, 0.5)
    assert 0.6 <= m["timing_window"]["coverage"] <= 0.95
    assert m["concordance"]["all_ideas_p52"]["c_index"] > 0.55
    names = [p["parameter"] for p in m["parameters"]]
    assert names[:3] == ["intercept", "z_src_hansard", "z_initial_firmness"]
    assert m["horizons"]["52"]["model"]["auc"] > 0.6
    # the training fit recovers the true parameters (within 2.5 standard errors)
    truth = {"intercept": TRUE_GAMMA0, **TRUE_GAMMA, **TRUE_BETA}
    for p in m["parameters"]:
        if p["parameter"] in truth:
            assert abs(p["estimate"] - truth[p["parameter"]]) < 2.5 * p["std_err"], p


def test_later_points_use_signals_as_of_each_point(truth):
    td = FakeTakeupData(truth)
    out = ev.backtest(td, date(2019, 1, 1), END)
    later = out["metrics"]["later_as_of"]
    assert [p["as_of"] for p in later["points"]][0] == "2020-01-01"
    assert pd.Timestamp("2020-01-01") in td.calls
    assert later["n"] > 0 and later["brier"] is not None


# ---------------------------------------------------------------- charts, report, runs


@pytest.fixture(scope="module")
def evaluation(truth, tmp_path_factory):
    out = tmp_path_factory.mktemp("reports")
    doc = out / "TAKEUP_EVALUATION.md"
    metrics = ev.run_evaluation(
        write=False, td=FakeTakeupData(truth), end=END, out_dir=out, doc_path=doc
    )
    return metrics, out, doc


def test_run_evaluation_writes_charts_metrics_and_report(evaluation):
    metrics, out, doc = evaluation
    for name in ("calibration", "pi_now_curves", "time_to_bill", "feature_effects"):
        path = out / f"{name}.png"
        assert path.exists() and path.stat().st_size > 1000
    saved = json.loads((out / "metrics.json").read_text())
    assert [b["cut"] for b in saved["backtests"]] == ["2019-01-01", "2022-01-01"]
    text = doc.read_text()
    assert "King's Speech" in text and "B1" in text and "80% timing window" in text


def test_save_run_on_sqlite(truth):
    engine = create_engine("sqlite://")
    takeup_metadata.create_all(engine)
    full = ev.fit_all(FakeTakeupData(truth), END)
    assert len(full["forecasts"]) > 0
    run1 = ev.save_run(engine, full["model"], END, {"a": 1}, full["forecasts"], version="test")
    run2 = ev.save_run(engine, full["model"], END, {"a": 2}, full["forecasts"].head(3), "test")
    assert (run1, run2) == (1, 2)
    with engine.connect() as conn:
        runs = conn.execute(select(takeup_runs)).mappings().all()
        fc = conn.execute(select(takeup_forecasts).where(takeup_forecasts.c.run_id == 1)).all()
    assert len(runs) == 2 and json.loads(runs[0]["metrics"]) == {"a": 1}
    assert runs[0]["features"] == ",".join([*full["model"].zcols, *full["model"].xcols])
    assert "z_src_hansard" in json.loads(runs[0]["parameters"])
    assert len(fc) == len(full["forecasts"])
