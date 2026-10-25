"""Take-up cure model with the sitting calendar (TAKEUP_PLAN §2.2 change 1). Offline."""

import inspect
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import expit

from westminster_weather.takeup import model as tm

# ------------------------------------------------------------------------ helpers

RECESS_ISO_WEEKS = {1, 14, 31, 32, 33, 34, 35, 36, 52, 53}  # Easter, summer, Christmas


def recess_calendar(start: str = "2005-01-01", end: str = "2031-12-31") -> pd.DatetimeIndex:
    """Monday to Thursday sittings, except recess weeks each year."""
    days = pd.date_range(start, end)
    week = days.isocalendar().week.to_numpy()
    return days[(days.weekday < 4) & ~np.isin(week, list(RECESS_ISO_WEEKS))]


def calendar_without(start: str, end: str, gap: tuple[str, str]) -> pd.DatetimeIndex:
    """Monday to Thursday sittings, with no sittings from gap[0] to gap[1]."""
    days = pd.date_range(start, end)
    days = days[days.weekday < 4]
    return days[(days < gap[0]) | (days > gap[1])]


def one_idea(**kw) -> pd.DataFrame:
    row = {"signal_id": "I1", "title": "Idea", "signal_date": "2020-06-01", "bill_date": None}
    row.update(kw)
    return pd.DataFrame([row])


def model_for(d, alpha=(-2.5, -2.8, -3.0, -3.0, -3.2, -3.5), g0=0.3, gamma=None, beta=None):
    gamma = [0.5] * len(d["zcols"]) if gamma is None else gamma
    beta = [0.8] * len(d["xcols"]) if beta is None else beta
    return tm.TakeUpModel.from_params([g0, *gamma, *alpha, *beta], d["zcols"], d["xcols"])


# ------------------------------------------------------------- mask construction


def test_sitting_mask_matches_brute_force():
    cal = recess_calendar("2018-01-01", "2024-12-31")
    sat = set(cal.date)
    signals = ["2019-07-17", "2020-12-20", "2021-03-30"]
    M = tm.sitting_mask(signals, [tm.T_MAX] * 3, cal)
    for i, s in enumerate(signals):
        start = date.fromisoformat(s)
        for t in range(1, 120):
            week = [start + timedelta(days=7 * (t - 1) + k) for k in range(7)]
            assert M[i, t - 1] == float(any(d in sat for d in week)), (s, t)
    assert (M == 0).any() and (M == 1).any()


def test_weeks_after_data_end_use_sitting_probability():
    cal = recess_calendar()
    prob = pd.Series(np.linspace(0.05, 0.95, 53), index=pd.RangeIndex(1, 54))
    d = tm.prepare(one_idea(data_end="2020-09-01"), cal, week_probability=prob)
    c = d["weeks_observed"][0]  # floor(92 / 7) = 13
    assert c == 13
    assert set(np.unique(d["M"][0, :c])) <= {0.0, 1.0}
    mid = pd.Timestamp("2020-06-01") + pd.to_timedelta(7 * np.arange(c, tm.T_MAX) + 3, unit="D")
    expected = prob.loc[mid.isocalendar().week.to_numpy()].to_numpy()
    np.testing.assert_allclose(d["M"][0, c:], expected)
    # weeks starting after the last known sitting date are not observed either
    d2 = tm.prepare(one_idea(data_end="2026-10-24"), cal[cal < "2021-01-01"], week_probability=prob)
    unknown = pd.Timestamp("2020-06-01") + pd.to_timedelta(7 * np.arange(tm.T_MAX), unit="D")
    unknown = unknown > pd.Timestamp(cal[cal < "2021-01-01"][-1])
    assert np.all((d2["M"][0][unknown] > 0) & (d2["M"][0][unknown] < 1))


# -------------------------------------------------------- non-sitting weeks (b)


def test_silent_recess_weeks_change_neither_likelihood_nor_pi_now():
    # No sittings 27 Jul - 30 Aug 2020: weeks 9-13 after a 1 June signal.
    cal = calendar_without("2020-01-01", "2022-12-31", ("2020-07-27", "2020-08-30"))
    df = one_idea(z_a=1.0, x_b="2020-08-10")
    before = tm.prepare(df.assign(data_end="2020-07-27"), cal, pd.Series(0.7, range(1, 54)))
    after = tm.prepare(df.assign(data_end="2020-08-31"), cal, pd.Series(0.7, range(1, 54)))
    assert (before["t"][0], after["t"][0]) == (8, 13)
    assert np.all(after["M"][0, 8:13] == 0)

    m = model_for(before)  # baseline differs by bin; x_b switches on inside the recess
    np.testing.assert_allclose(m.loglik_contributions(before), m.loglik_contributions(after))
    p_before = m.predict(**tm.row_inputs(before, 0))
    p_after = m.predict(**tm.row_inputs(after, 0))
    assert p_before["pi_now"] == pytest.approx(p_after["pi_now"], abs=1e-12)
    assert p_after["pi_now"] < p_after["pi"]


def test_signal_moved_back_through_recess_keeps_likelihood_with_flat_baseline():
    cal = calendar_without("2020-01-01", "2022-12-31", ("2020-07-27", "2020-08-30"))
    base = {"z_a": 1.0, "x_b": "2020-09-14", "data_end": "2021-06-01"}
    for bill in ("2020-10-05", None):
        late = tm.prepare(one_idea(signal_date="2020-08-31", bill_date=bill, **base), cal)
        early = tm.prepare(one_idea(signal_date="2020-07-27", bill_date=bill, **base), cal)
        assert early["t"][0] == late["t"][0] + 5
        m = model_for(late, alpha=[-3.0] * len(tm.BIN_EDGES))
        np.testing.assert_allclose(m.loglik_contributions(late), m.loglik_contributions(early))


# ------------------------------------------------- bill in a non-sitting week (c)


def test_bill_in_non_sitting_week_is_forced_to_sitting_and_counted():
    cal = calendar_without("2020-01-01", "2022-12-31", ("2020-07-27", "2020-08-30"))
    df = pd.concat(
        [
            one_idea(signal_id="gap", bill_date="2020-08-12", data_end="2021-01-01"),
            one_idea(signal_id="ok", bill_date="2020-10-07", data_end="2021-01-01"),
        ]
    )
    d = tm.prepare(df, cal)
    assert d["n_forced_sitting"] == 1
    assert list(d["forced_ids"]) == ["gap"]
    assert d["delta"].tolist() == [1, 1]
    assert d["M"][0, d["t"][0] - 1] == 1.0
    assert d["M"][0, d["t"][0] - 2] == 0.0  # the rest of the recess stays non-sitting
    ll = model_for(d).loglik_contributions(d)
    assert np.all(np.isfinite(ll))
    fitted = tm.TakeUpModel().fit(d)
    assert np.isfinite(fitted._negloglik(fitted.theta, d))


# ------------------------------------------------------- per-row data_end (d)


def test_row_data_end_censors_later_bill_and_hides_later_events():
    cal = recess_calendar()
    df = pd.concat(
        [
            # change of government: censored at the 2010 dissolution although a bill came later
            one_idea(
                signal_id="cog",
                signal_date="2009-06-01",
                bill_date="2010-11-03",
                data_end="2010-04-12",
                x_a="2010-06-01",
            ),
            one_idea(
                signal_id="same",
                signal_date="2009-06-01",
                bill_date="2010-11-03",
                x_a="2010-06-01",
            ),
        ]
    )
    d = tm.prepare(df, cal, data_end="2026-10-24")
    assert d["delta"].tolist() == [0, 1]
    assert d["t"][0] == (date(2010, 4, 12) - date(2009, 6, 1)).days // 7
    assert d["t"][1] == (date(2010, 11, 3) - date(2009, 6, 1)).days // 7 + 1
    assert d["xweek"][0, 0] == tm.T_MAX + 1  # after its data_end: not known
    assert d["xweek"][1, 0] == (date(2010, 6, 1) - date(2009, 6, 1)).days // 7 + 1
    assert str(d["data_end"][0]) == "2010-04-12"


def test_global_cut_clips_row_data_end():
    d = tm.prepare(
        one_idea(bill_date="2021-03-03", data_end="2026-10-24"),
        recess_calendar(),
        data_end="2021-01-01",
    )
    assert d["delta"][0] == 0
    assert d["t"][0] == (date(2021, 1, 1) - date(2020, 6, 1)).days // 7


def test_bill_shortly_before_signal_is_a_week_one_bill():
    d = tm.prepare(
        pd.concat(
            [
                one_idea(signal_id="before", signal_date="2020-06-10", bill_date="2020-06-07"),
                one_idea(signal_id="same", signal_date="2020-06-10", bill_date="2020-06-10"),
            ]
        ),
        recess_calendar(),
        data_end="2026-10-24",
    )
    assert d["delta"].tolist() == [1, 1]
    assert d["t"].tolist() == [1, 1]
    assert d["n_bill_before_signal"] == 1
    assert tm.row_inputs(d, 0)["weeks_elapsed"] == 0


def test_bill_in_last_partial_week_is_an_event():
    # bill on day 50 after the signal; data_end on day 53: only 7 whole weeks observed
    cal = recess_calendar()
    df = pd.concat(
        [
            one_idea(signal_id="bill", signal_date="2026-09-01", bill_date="2026-10-21"),
            one_idea(signal_id="none", signal_date="2026-09-01"),
            one_idea(signal_id="late", signal_date="2026-09-01", bill_date="2026-10-26"),
        ]
    )
    d = tm.prepare(df, cal, pd.Series(0.5, range(1, 54)), data_end="2026-10-24")
    assert d["delta"].tolist() == [1, 0, 0]
    assert d["t"].tolist() == [8, 7, 7]
    assert d["M"][0, 7] == 1.0  # the bill week is a sitting week
    assert tm.row_inputs(d, 0)["weeks_elapsed"] == 0
    assert np.all(np.isfinite(model_for(d).loglik_contributions(d)))


def test_calendar_end_keeps_recess_before_as_of_as_non_sitting():
    cal = recess_calendar()
    as_of = "2021-09-08"  # in recess; last sitting 2021-07-29
    prob = pd.Series(0.5, range(1, 54))
    df = one_idea(signal_date="2021-03-01", data_end=as_of)
    full = tm.prepare(df, cal, prob)
    cut = tm.prepare(df, cal[cal <= as_of], prob, calendar_end=as_of)
    stale = tm.prepare(df, cal[cal <= as_of], prob)  # default: unknown after the last sitting
    t = full["t"][0]
    np.testing.assert_array_equal(cut["M"][0, :t], full["M"][0, :t])
    assert np.all(cut["M"][0, t - 5 : t] == 0)
    assert np.all(stale["M"][0, t - 5 : t] == 0.5)
    m = model_for(full)
    np.testing.assert_allclose(m.loglik_contributions(cut), m.loglik_contributions(full))


def test_weeks_before_calendar_start_are_not_treated_as_recess():
    cal = recess_calendar("2010-05-18", "2031-12-31")
    prob = pd.Series(0.6, range(1, 54))
    d = tm.prepare(one_idea(signal_date="2006-03-01"), cal, prob, data_end="2026-10-24")
    before = pd.Timestamp("2006-03-01") + pd.to_timedelta(7 * np.arange(tm.T_MAX) + 6, unit="D")
    before = before < pd.Timestamp("2010-05-18")
    assert before.sum() > 200
    np.testing.assert_allclose(d["M"][0][before], 0.6)
    # an explicit earlier calendar start makes those weeks known non-sitting weeks
    d2 = tm.prepare(
        one_idea(signal_date="2006-03-01"),
        cal,
        prob,
        data_end="2026-10-24",
        calendar_start="2006-01-01",
    )
    assert np.all(d2["M"][0][before] == 0)


def test_resample_keeps_row_arrays_aligned():
    cal = calendar_without("2018-01-01", "2022-12-31", ("2019-07-22", "2019-09-01"))
    df = pd.concat(
        [
            one_idea(signal_id="a", signal_date="2019-01-01"),
            one_idea(signal_id="b", signal_date="2019-06-01", bill_date="2019-08-07"),
            one_idea(signal_id="c", signal_date="2020-01-01", bill_date="2019-12-30"),
        ]
    )
    d = tm.prepare(df, cal, data_end="2021-01-01")
    assert d["n_forced_sitting"] == 1 and d["n_bill_before_signal"] == 1
    r = tm.resample(d, np.array([2, 2]))
    assert r["ids"].tolist() == ["c", "c"]
    assert [str(x) for x in r["signal_date"]] == ["2020-01-01"] * 2
    assert [str(x) for x in r["data_end"]] == ["2021-01-01"] * 2
    assert r["n_forced_sitting"] == 0 and list(r["forced_ids"]) == []
    assert r["n_bill_before_signal"] == 2
    assert r["bin_events"].sum() == 2


def test_non_positive_variance_gives_nan_standard_error():
    theta = [0.0, *([-3.0] * len(tm.BIN_EDGES))]
    Sigma = np.diag([0.04, -1e-10, *([0.01] * (len(tm.BIN_EDGES) - 1))])
    se = tm.TakeUpModel.from_params(theta, [], [], Sigma=Sigma).standard_errors()
    assert se[0] == pytest.approx(0.2)
    assert np.isnan(se[1]) and np.all(np.isfinite(se[2:]))


# --------------------------------------------------------------- verdicts (e)


@pytest.mark.parametrize(
    ("g0", "se", "verdict"),
    [
        (1.0, 0.1, "likely"),
        (-1.0, 0.1, "unlikely"),
        (0.3, 0.5, "uncertain"),
        (0.0, 0.0, "uncertain"),
    ],
)
def test_verdict_thresholds(g0, se, verdict):
    theta = [g0, *([-3.0] * len(tm.BIN_EDGES))]
    Sigma = np.diag([se**2] + [0.01] * len(tm.BIN_EDGES))
    m = tm.TakeUpModel.from_params(theta, [], [], Sigma=Sigma)
    pi, lo, hi, v = m.pi_with_ci(np.array([]))
    assert v == verdict
    assert pi == pytest.approx(expit(g0))
    assert (lo, hi) == pytest.approx((expit(g0 - 1.96 * se), expit(g0 + 1.96 * se)))


# ------------------------------------------------------------- predictions (f)


def test_predict_outputs_are_coherent():
    cal = recess_calendar()
    prob = pd.Series(0.6, index=pd.RangeIndex(1, 54))
    df = one_idea(signal_date="2025-01-15", z_a=1.0, x_b="2025-06-01", data_end="2026-10-24")
    d = tm.prepare(df, cal, week_probability=prob)
    m = model_for(d)
    p0 = m.predict(**{**tm.row_inputs(d, 0), "weeks_elapsed": 0})
    p = m.predict(**tm.row_inputs(d, 0))
    assert p["weeks_elapsed"] == d["t"][0] > 0
    for out in (p0, p):
        for key in ("pi", "pi_lo", "pi_hi", "pi_now", *(f"p{h}" for h in tm.HORIZONS)):
            assert 0.0 <= out[key] <= 1.0
        assert out["p13"] <= out["p26"] <= out["p52"]
        assert out["p52"] <= out["pi_now"]
    assert p0["pi_now"] == pytest.approx(p0["pi"])
    assert p["pi_now"] < p["pi"]
    # sitting-week timing never exceeds calendar timing; equal with an all-ones mask
    for key in ("t_10", "t_50", "t_90"):
        assert p[f"{key}_sitting"] <= p[key]
    ones = m.predict(d["Z"][0], d["xweek"][0], weeks_elapsed=10)
    for key in ("t_10", "t_50", "t_90"):
        assert ones[f"{key}_sitting"] == ones[key]
    assert "sitting weeks" in tm.describe("I1", "Idea", p)


# ------------------------------------------- conditional timing quantiles (g)

T_KEYS = ("t_10", "t_50", "t_90")
CONST_LOGIT = -2.0  # lambda = 0.1192 per sitting week
# ceil(log(1 - q) / log(1 - lambda)) = ceil(0.83), ceil(5.46), ceil(18.14): no ties at q
GEOMETRIC_K = {"t_10": 1, "t_50": 6, "t_90": 19}


def constant_hazard_model(g0: float = 0.5) -> tm.TakeUpModel:
    return tm.TakeUpModel.from_params([g0, *([CONST_LOGIT] * len(tm.BIN_EDGES))], [], [])


def predict_const(m: tm.TakeUpModel, s: int, mask: np.ndarray | None = None) -> dict:
    return m.predict(np.array([]), np.array([]), weeks_elapsed=s, mask=mask)


def test_pi_now_is_finite_when_pi_rounds_to_one_and_survival_underflows():
    # eta = 40 makes pi == 1.0 in floats; hazard logit 8 makes log S_u(100) < -745, so
    # S_u(100) == 0.0 and the direct formula pi S / (1 - pi + pi S) would be 0 / 0
    m = tm.TakeUpModel.from_params([40.0, *([8.0] * len(tm.BIN_EDGES))], [], [])
    assert m.pi_with_ci(np.array([]))[0] == 1.0
    assert np.exp(m.log_survival_u(np.array([]))[99]) == 0.0
    out = predict_const(m, 100)
    assert out["pi_now"] == pytest.approx(expit(40.0 + m.log_survival_u(np.array([]))[99]))
    for key in ("pi_now", *(f"p{h}" for h in tm.HORIZONS)):
        assert np.isfinite(out[key]) and 0.0 <= out[key] <= 1.0


@pytest.mark.parametrize("s", [0, 1, 7, 40, tm.T_MAX])
def test_pi_now_equals_direct_formula_in_ordinary_range(s):
    m = constant_hazard_model(g0=0.5)
    pi = expit(0.5)
    Su_s = 1.0 if s == 0 else m.survival_u(np.array([]))[s - 1]
    expected = pi * Su_s / (1 - pi + pi * Su_s)
    assert predict_const(m, s)["pi_now"] == pytest.approx(expected, rel=1e-12)


def brute_force_quantile(Su: np.ndarray, s: int, q: float) -> int | None:
    """min{t > s : 1 - S_u(t) / S_u(s) >= q}, by a plain loop over weeks."""
    S = [1.0, *Su]
    for t in range(s + 1, tm.T_MAX + 1):
        if 1 - S[t] / S[s] >= q - 1e-12:
            return t
    return None


@pytest.fixture(scope="module")
def recess_forecast():
    """A model with a varying baseline and a time-varying feature, on a recess calendar."""
    prob = pd.Series(0.6, index=pd.RangeIndex(1, 54))
    df = one_idea(signal_date="2021-01-15", z_a=1.0, x_b="2021-09-01", data_end="2026-10-24")
    d = tm.prepare(df, recess_calendar(), week_probability=prob)
    return model_for(d, alpha=(-2.0, -2.6, -3.0, -3.3, -3.6, -4.0)), tm.row_inputs(d, 0)


def test_conditional_quantiles_are_after_s_for_every_s(recess_forecast):
    m, row = recess_forecast
    Su = m.survival_u(row["xweek_row"], row["mask"])
    t50_unconditional = m.predict(**{**row, "weeks_elapsed": 0})["t_50"]
    seen_late = 0
    for s in range(0, tm.T_MAX + 1, 3):
        p = m.predict(**{**row, "weeks_elapsed": s})
        reached = [p[k] for k in T_KEYS if p[k] is not None]
        assert all(t > s for t in reached), (s, reached)
        assert reached == sorted(reached)
        for k, q in zip(T_KEYS, tm.QUANTILES, strict=True):
            assert p[k] == brute_force_quantile(Su, s, q), (s, k)
            if p[k] is not None:
                assert p[f"{k}_sitting"] <= p[k]
        # (5) probabilities are ordered and below pi_now
        assert 0.0 <= p["p13"] <= p["p26"] <= p["p52"] <= p["pi_now"] <= 1.0
        seen_late += s > t50_unconditional and p["t_50"] is not None
    assert seen_late > 10  # many s beyond the unconditional median still have a t_50 > s


def test_conditional_quantiles_at_s0_equal_unconditional(recess_forecast):
    m, row = recess_forecast
    Su = m.survival_u(row["xweek_row"], row["mask"])
    p = m.predict(**{**row, "weeks_elapsed": 0})
    for k, q in zip(T_KEYS, tm.QUANTILES, strict=True):
        assert p[k] == int(np.nonzero(1 - Su >= q)[0][0] + 1)
    assert p["pi_now"] == pytest.approx(p["pi"])


def test_geometric_waiting_time_quantiles_shift_by_s():
    m = constant_hazard_model(g0=0.5)
    lam, pi = expit(CONST_LOGIT), expit(0.5)
    for k, q in zip(T_KEYS, tm.QUANTILES, strict=True):  # the hand values above
        assert GEOMETRIC_K[k] == int(np.ceil(np.log(1 - q) / np.log(1 - lam)))
    for s in (0, 1, 5, 30, 100, 200):
        p = predict_const(m, s)
        for k in T_KEYS:
            assert p[k] == s + GEOMETRIC_K[k]
            assert p[f"{k}_sitting"] == p[k]  # every week sits
        S_s = (1 - lam) ** s
        pi_now = pi * S_s / (1 - pi + pi * S_s)
        assert p["pi_now"] == pytest.approx(pi_now, rel=1e-12)
        for H in tm.HORIZONS:  # memoryless: S_u(s+H) / S_u(s) = (1 - lambda)^H
            assert p[f"p{H}"] == pytest.approx(pi_now * (1 - (1 - lam) ** H), rel=1e-10)
    # near the end of follow-up: quantiles not reached by T_MAX are None
    late = predict_const(m, tm.T_MAX - 3)
    assert late["t_10"] == tm.T_MAX - 2 and late["t_50"] is None and late["t_90"] is None
    assert late["t_50_sitting"] is None
    end = predict_const(m, tm.T_MAX)
    assert all(end[k] is None for k in T_KEYS) and end["p52"] == 0.0


def test_conditional_quantiles_skip_non_sitting_weeks_after_s():
    m = constant_hazard_model()
    lam = expit(CONST_LOGIT)
    s = 20
    mask = np.ones(tm.T_MAX)
    mask[[s, s + 1, s + 2, s + 5, s + 6, s + 7, s + 8]] = 0  # weeks s+1..s+3, s+6..s+9
    p = predict_const(m, s, mask)
    sitting_after = [t for t in range(s + 1, tm.T_MAX + 1) if mask[t - 1] == 1]
    # by hand: sitting weeks after s are s+4, s+5, s+10, s+11, ...
    assert (p["t_10"], p["t_50"], p["t_90"]) == (s + 4, s + 13, s + 26)
    for k in T_KEYS:
        assert p[k] == sitting_after[GEOMETRIC_K[k] - 1]
        assert mask[p[k] - 1] == 1  # never a week with no sitting
        assert p[f"{k}_sitting"] == s + GEOMETRIC_K[k]  # all s weeks before sat
    # masked weeks carry no probability: p26 counts only the 19 sitting weeks in s+1..s+26
    n_sit = int(mask[s : s + 26].sum())
    assert n_sit == 19
    assert p["p26"] == pytest.approx(p["pi_now"] * (1 - (1 - lam) ** n_sit), rel=1e-10)
    # and masked weeks before s do not lower pi_now
    shifted = np.concatenate([np.zeros(5), np.ones(tm.T_MAX - 5)])
    assert predict_const(m, s + 5, shifted)["pi_now"] == pytest.approx(
        predict_const(m, s)["pi_now"]
    )


def test_bootstrap_intervals_condition_on_the_same_s():
    models = [constant_hazard_model(g0) for g0 in np.linspace(-1.0, 1.0, 21)]
    mask = np.ones(tm.T_MAX)
    ci = tm.bootstrap_intervals(models, np.array([]), np.array([]), 40, mask, level=0.9)
    preds = [predict_const(b, 40, mask) for b in models]
    for key in ("pi_now", "p26", "p52"):
        expected = np.percentile([p[key] for p in preds], [5, 95])
        np.testing.assert_allclose(ci[key], expected)
    at_signal = tm.bootstrap_intervals(models, np.array([]), np.array([]), 0, mask)
    assert ci["pi_now"][1] < at_signal["pi_now"][0]  # 40 silent weeks lower pi_now
    assert inspect.signature(tm.bootstrap).parameters["B"].default >= 200


# ----------------------------------------------------------- gradient check


def test_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    data = synthetic_masked(n=150, seed=3)[0]
    d = tm.prepare(data, recess_calendar(), pd.Series(0.7, range(1, 54)), data_end="2026-10-24")
    m = model_for(d)
    theta = m.theta + rng.normal(0, 0.3, m.n_params)
    _, grad = m._negloglik_and_grad(theta, d)
    eps = 1e-5
    for i in range(m.n_params):
        e = np.zeros(m.n_params)
        e[i] = eps
        num = (m._negloglik(theta + e, d) - m._negloglik(theta - e, d)) / (2 * eps)
        assert grad[i] == pytest.approx(num, rel=1e-4, abs=1e-4)


# ------------------------------------------------------ parameter recovery (a)

TRUE_G0 = -0.4
TRUE_GAMMA = {"z_manifesto": 1.0, "z_cost": -0.6}
TRUE_ALPHA = (-2.4, -2.7, -3.0, -3.0, -3.2, -3.4)
TRUE_BETA = {"x_consultation_response": 1.0, "x_draft_bill": 0.7}
DATA_END = pd.Timestamp("2026-10-24")


def synthetic_masked(n: int = 2500, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """Ideas generated from the true parameters, with bills only in sitting weeks."""
    rng = np.random.default_rng(seed)
    cal = recess_calendar()
    sat = set(cal.date)
    start = [date(2008, 1, 1) + timedelta(days=int(k)) for k in rng.integers(0, 365 * 17, n)]
    z = np.column_stack([rng.random(n) < 0.45, rng.uniform(0, 2, n)]).astype(float)
    taken = rng.random(n) < expit(TRUE_G0 + z @ np.array(list(TRUE_GAMMA.values())))
    xw = np.where(rng.random((n, 2)) < 0.5, rng.integers(1, 80, (n, 2)), tm.T_MAX + 1)
    M = tm.sitting_mask(start, np.full(n, tm.T_MAX), cal)
    weeks = np.arange(1, tm.T_MAX + 1)
    h = np.array(TRUE_ALPHA)[tm.bin_index(tm.T_MAX)][None, :].repeat(n, axis=0)
    for j, b in enumerate(TRUE_BETA.values()):
        h += b * (weeks[None, :] >= xw[:, j : j + 1])
    hit = rng.random((n, tm.T_MAX)) < M * expit(h)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)

    # a quarter of ideas stop being followed early (e.g. a change of government)
    early = rng.random(n) < 0.25
    follow = rng.integers(30, 900, n)
    end = [
        s + timedelta(days=int(f)) if e else DATA_END.date()
        for s, e, f in zip(start, early, follow, strict=True)
    ]
    rows = []
    for i in range(n):
        s = start[i]
        bill = None
        if taken[i] and first[i]:
            week_start = s + timedelta(days=int(7 * (first[i] - 1)))
            days = (week_start + timedelta(days=k) for k in range(7))
            bill = next(day for day in days if day in sat)
        rows.append(
            {
                "signal_id": f"S{i:04d}",
                "title": f"Synthetic idea {i}",
                "signal_date": s,
                "bill_date": bill,
                "data_end": end[i],
                "z_manifesto": z[i, 0],
                "z_cost": z[i, 1],
                **{
                    c: s + timedelta(days=int(7 * (xw[i, j] - 1))) if xw[i, j] <= tm.T_MAX else None
                    for j, c in enumerate(TRUE_BETA)
                },
            }
        )
    return pd.DataFrame(rows), M


@pytest.fixture(scope="module")
def recovered():
    df, _ = synthetic_masked()
    prob = pd.Series(0.7, index=pd.RangeIndex(1, 54))
    d = tm.prepare(df, recess_calendar(), week_probability=prob, data_end=DATA_END)
    return d, tm.TakeUpModel().fit(d)


def test_synthetic_parameters_recovered_within_3_se(recovered):
    d, m = recovered
    assert m.converged
    assert d["n_forced_sitting"] == 0
    assert 0.2 < d["delta"].mean() < 0.6
    table = m.parameter_table().set_index("parameter")
    truth = {
        "intercept": TRUE_G0,
        **TRUE_GAMMA,
        **TRUE_BETA,
        **{f"alpha_weeks_{e}+": a for e, a in zip(tm.BIN_EDGES[:4], TRUE_ALPHA[:4], strict=True)},
    }
    for name, value in truth.items():
        est, se = table.loc[name, "estimate"], table.loc[name, "std_err"]
        assert 0 < se < 0.5, name
        assert abs(est - value) < 3 * se, (name, est, value, se)


def test_bootstrap_intervals(recovered):
    d, m = recovered
    k = int(np.nonzero(d["delta"] == 0)[0][0])
    boots = tm.bootstrap(tm.resample(d, np.arange(400)), B=3, seed=1, base=m)
    assert len(boots) == 3 and all(b.Sigma is None for b in boots)
    ci = tm.bootstrap_intervals(boots, **tm.row_inputs(d, k))
    assert set(ci) == {"pi_now", "p26", "p52"}
    for lo, hi in ci.values():
        assert 0.0 <= lo <= hi <= 1.0


# ------------------------------------------------- prototype equivalence check

# takeup_model.py (prototype) fitted on make_synthetic_signals.py data (seed 42, N = 900),
# run 2026-10-24 on a scratch copy of ~/Downloads/law-in-time 2.
PROTOTYPE_THETA = np.array(
    [-1.572246, 1.112025, 1.315008, -0.724191, 0.190325, -2.956003, -2.88335, -3.205017,
     -2.913523, -2.879279, -3.005174, 1.347498, 1.446144]
)  # fmt: skip
PROTOTYPE_NEGLOGLIK = 2012.2595237934538


def prototype_synthetic() -> pd.DataFrame:
    """make_synthetic_signals.py, unchanged in logic (same random draws in the same order)."""
    rng = np.random.default_rng(42)
    gamma0 = -2.0
    gamma = {
        "z_manifesto": 1.2,
        "z_minister_statement": 1.5,
        "z_high_cost": -0.7,
        "z_years_to_election": 0.3,
    }
    alpha = -3.0
    beta = {"x_consultation_response": 1.2, "x_draft_bill": 1.5}
    data_end = date(2026, 10, 24)
    rows = []
    for i in range(900):
        start = date(2010, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 16)))
        z = {
            "z_manifesto": rng.random() < 0.4,
            "z_minister_statement": rng.random() < 0.5,
            "z_high_cost": rng.random() < 0.3,
            "z_years_to_election": round(float(rng.uniform(0, 5)), 2),
        }
        eta = gamma0 + sum(gamma[k] * float(v) for k, v in z.items())
        taken_up = rng.random() < 1 / (1 + np.exp(-eta))
        ev = {k: (int(rng.integers(1, 60)) if rng.random() < 0.5 else None) for k in beta}
        bill_week = None
        if taken_up:
            for t in range(1, 261):
                h = alpha + sum(beta[k] for k, w in ev.items() if w is not None and t >= w)
                if rng.random() < 1 / (1 + np.exp(-h)):
                    bill_week = t
                    break

        def to_date(w, start=start):
            return start + timedelta(weeks=w - 1) if w else None

        bill = to_date(bill_week)
        rows.append(
            {
                "signal_id": f"S{i:04d}",
                "title": f"Synthetic idea {i}",
                "signal_date": start,
                "bill_date": bill if bill and bill <= data_end else None,
                "data_end": data_end,
                **{k: int(v) if isinstance(v, bool | np.bool_) else v for k, v in z.items()},
                **{k: to_date(w) if w and to_date(w) <= data_end else None for k, w in ev.items()},
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def prototype_data():
    every_day = pd.date_range("2009-01-01", "2032-01-01")
    d = tm.prepare(prototype_synthetic(), every_day)
    assert d["M"].min() == 1.0 and d["n_forced_sitting"] == 0
    return d


def test_all_ones_mask_reproduces_prototype(prototype_data, monkeypatch):
    monkeypatch.setattr(tm, "ALPHA_SMOOTH", 0.0)  # the prototype has no baseline smoothing
    d = prototype_data
    m = tm.TakeUpModel()
    m.zcols, m.xcols, m.pz, m.px = d["zcols"], d["xcols"], len(d["zcols"]), len(d["xcols"])
    # same objective as the prototype ...
    assert m._negloglik(PROTOTYPE_THETA, d) == pytest.approx(PROTOTYPE_NEGLOGLIK, abs=1e-5)
    # ... and the prototype's optimiser settings land on the prototype's estimates
    theta0 = np.zeros(m.n_params)
    theta0[1 + m.pz : 1 + m.pz + len(tm.BIN_EDGES)] = tm.ALPHA_START
    res = minimize(m._negloglik, theta0, args=(d,), method="L-BFGS-B")
    np.testing.assert_allclose(res.x, PROTOTYPE_THETA, atol=1e-3)


def test_port_fit_is_at_least_as_good_as_prototype(prototype_data, monkeypatch):
    monkeypatch.setattr(tm, "ALPHA_SMOOTH", 0.0)
    m = tm.TakeUpModel().fit(prototype_data, hessian=False)
    assert m.converged
    # analytic gradient converges further (the week-105+ bin has no bills, so its MLE is
    # -inf; the prototype stopped near its start value), but the effects agree
    assert m._negloglik(m.theta, prototype_data) <= PROTOTYPE_NEGLOGLIK
    keep = [*range(0, 5), 11, 12]  # gamma0, gamma, beta
    np.testing.assert_allclose(m.theta[keep], PROTOTYPE_THETA[keep], atol=0.02)


def test_smoothing_keeps_empty_bin_finite_with_prototype_effects(prototype_data):
    d = prototype_data
    assert d["bin_events"][-1] == 0 and d["bin_exposure"][-1] > 0  # week 105+: no bills
    m = tm.TakeUpModel().fit(d)
    assert m.converged
    table = m.parameter_table().set_index("parameter")
    empty = table.loc["alpha_weeks_105+"]
    assert -15 < empty["estimate"] < table.loc["alpha_weeks_53+", "estimate"]
    assert np.isfinite(empty["std_err"]) and empty["std_err"] > 0
    keep = [*range(0, 5), 11, 12]
    np.testing.assert_allclose(m.theta[keep], PROTOTYPE_THETA[keep], atol=0.05)


def test_empty_bin_on_short_history_stays_finite_and_predicts(caplog):
    # prototype data cut early: no bills after week 52, many silent ideas past it
    df = prototype_synthetic()
    cut = "2013-06-01"
    train = df[pd.to_datetime(df["signal_date"]) < cut]
    d = tm.prepare(train, pd.date_range("2009-01-01", "2032-01-01"), data_end=cut)
    assert d["bin_events"][4:].sum() == 0 and d["bin_exposure"][4] > 0
    with caplog.at_level("WARNING", logger=tm.__name__):
        m = tm.TakeUpModel().fit(d)
    assert "weeks_53+" in caplog.text
    se = m.parameter_table().set_index("parameter")["std_err"]
    assert m.theta.min() > -30
    assert not (se == 0).any()  # never shown as exactly determined
    p = m.predict(np.array([1, 1, 0, 2.0]), np.array([tm.T_MAX + 1] * 2), weeks_elapsed=60)
    assert p["p52"] > 0
