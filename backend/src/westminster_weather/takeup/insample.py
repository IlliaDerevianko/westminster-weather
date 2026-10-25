"""Stage 1 in-sample check: fit the hazard model on every idea, then compare its forecast made
at each idea's trigger (information known on that day only) with what really happened.
Measures fit on the training data, not out-of-sample skill. Read-only.

`updating()` repeats the comparison one year after each promise, for promises still waiting:
forecast made with the data as it stood that month (no hindsight about later merges), outcome =
bill in the following 52 weeks, split by how often the promise was repeated in its first year.
Run: uv run python -m westminster_weather.takeup.insample [--updating]"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from westminster_weather.takeup import evaluate as ev
from westminster_weather.takeup.signals import TakeupData

warnings.filterwarnings("ignore")
OUT = "reports/takeup/stage1_insample.csv"
OUT_UPDATING = "reports/takeup/stage1_updating.csv"


def main() -> None:
    td = TakeupData()
    end = ev.default_end(td)
    sig = td.get_takeup_signals(end).reset_index(drop=True)
    model = ev.fit_hazard(td, sig, end)
    wp, sd = td.week_probability(end), td.sitting_dates(end)
    tl = td.mention_timeline(end)

    rows = []
    for _, s in sig.iterrows():
        trig = pd.Timestamp(s["signal_date"])
        m = tl[(tl["signal_id"] == s["signal_id"]) & (pd.to_datetime(tl["event_date"]) <= trig)]
        p = model.predict(s, m, trig, sd, wp)
        bill = pd.Timestamp(s["bill_date"]) if pd.notna(s["bill_date"]) else None
        weeks = (bill - trig).days // 7 + 1 if bill is not None else None
        followed = (pd.Timestamp(s["data_end"]) - trig).days // 7
        rows.append(
            {
                "signal_id": s["signal_id"],
                "topic": s.get("title"),
                "promised": trig.date(),
                "pred_pi": p["pi"],
                "pred_p52": p["p52"],
                "pred_t50_weeks": p.get("t_50"),
                "pred_t10_weeks": p.get("t_10"),
                "pred_t90_weeks": p.get("t_90"),
                "verdict": p["verdict"],
                "real_bill": bill is not None,
                "real_bill_date": bill.date() if bill is not None else None,
                "real_weeks_to_bill": weeks,
                "weeks_followed": followed,
            }
        )
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    print(f"Stage 1 in-sample: {len(d)} promises, {d['real_bill'].sum()} got a Government Bill")
    known52 = d[d["real_bill"] | (d["weeks_followed"] >= 52)]
    y52 = (known52["real_weeks_to_bill"].fillna(1e9) <= 52).astype(int)
    print(
        f"\nBill within 52 weeks (n={len(known52)}, {y52.sum()} yes): "
        f"AUC {roc_auc_score(y52, known52['pred_p52']):.3f}, "
        f"Brier {brier_score_loss(y52, known52['pred_p52']):.3f} "
        f"(base-rate Brier {brier_score_loss(y52, np.full(len(y52), y52.mean())):.3f})"
    )
    known5 = d[d["real_bill"] | (d["weeks_followed"] >= 260)]
    y5 = known5["real_bill"].astype(int)
    print(
        f"Bill ever (within 5 years; n={len(known5)}, {y5.sum()} yes): "
        f"AUC {roc_auc_score(y5, known5['pred_pi']):.3f}, "
        f"Brier {brier_score_loss(y5, known5['pred_pi']):.3f} "
        f"(base-rate Brier {brier_score_loss(y5, np.full(len(y5), y5.mean())):.3f})"
    )

    print("\nCalibration, bill within 52 weeks (predicted vs real share):")
    bins = pd.cut(known52["pred_p52"], [0, 0.05, 0.1, 0.2, 0.4, 1.0], include_lowest=True)
    for b, g in known52.assign(y=y52.values).groupby(bins, observed=True):
        print(
            f"  predicted {b}: n={len(g):4d}  mean predicted {g['pred_p52'].mean():5.1%}  "
            f"real {g['y'].mean():5.1%}"
        )

    t = d[d["real_bill"] & d["pred_t10_weeks"].notna() & d["pred_t90_weeks"].notna()]
    inside = (t["pred_t10_weeks"] <= t["real_weeks_to_bill"]) & (
        t["real_weeks_to_bill"] <= t["pred_t90_weeks"]
    )
    err = (t["pred_t50_weeks"] - t["real_weeks_to_bill"]).abs()
    print(
        f"\nTiming for promises that got a bill (n={len(t)}): real date inside the 80% window "
        f"{inside.mean():.0%}; median error of the predicted median {err.median():.0f} weeks"
    )
    print("\nVerdicts vs reality (promises followed 2+ years or with a bill):")
    v = d[d["real_bill"] | (d["weeks_followed"] >= 104)]
    print(pd.crosstab(v["verdict"], v["real_bill"].map({True: "got bill", False: "no bill"})))
    print(f"\nwrote {OUT}")


def updating(after_weeks: int = 52, horizon: int = 52) -> pd.DataFrame:
    """Forecasts made ``after_weeks`` after each promise (still no bill) vs a bill in the next
    ``horizon`` weeks. The model is fitted on all data (in-sample); each forecast uses the
    signals, mention timeline and calendar as they stood on its month's first day."""
    td = TakeupData()
    end = ev.default_end(td)
    full = td.get_takeup_signals(end)
    model = ev.fit_hazard(td, full.reset_index(drop=True), end)
    ideas = full.assign(signal_date=pd.to_datetime(full["signal_date"]))
    ideas["check"] = (ideas["signal_date"] + pd.Timedelta(weeks=after_weeks)).dt.to_period(
        "M"
    ).dt.to_timestamp() + pd.offsets.MonthBegin(1)
    ideas = ideas[ideas["check"] + pd.Timedelta(weeks=horizon) <= end]
    rows = []
    for as_of, group in ideas.groupby("check"):
        sigs = td.get_takeup_signals(as_of)
        tl = td.mention_timeline(as_of)
        wp, sd = td.week_probability(as_of), td.sitting_dates(as_of)
        for idea_id in group["idea_id"].unique():
            cand = sigs[sigs["idea_id"] == idea_id]
            if cand.empty or cand["bill_date"].notna().any():
                continue  # not visible yet, or the bill already came
            s = cand.sort_values("signal_date").iloc[0]
            m = tl[tl["signal_id"] == s["signal_id"]]
            p = model.predict(s, m, as_of, sd, wp)
            trig = pd.Timestamp(s["signal_date"])
            n_year1 = int((pd.to_datetime(m["event_date"]) > trig).sum()) if len(m) else 0
            bill = full.loc[full["idea_id"] == idea_id, "bill_date"].dropna()
            bill = pd.Timestamp(bill.min()) if len(bill) else None
            rows.append(
                {
                    "idea_id": idea_id,
                    "topic": s.get("title"),
                    "promised": trig.date(),
                    "as_of": as_of.date(),
                    "repeat_mentions_since_promise": n_year1,
                    "pred_p52": p["p52"],
                    "pred_pi_now": p["pi_now"],
                    "verdict": p["verdict"],
                    "real_bill_next_52w": bill is not None
                    and bill <= as_of + pd.Timedelta(weeks=horizon),
                    "real_bill_date": bill.date() if bill is not None else None,
                }
            )
    d = pd.DataFrame(rows)
    d.to_csv(OUT_UPDATING, index=False)
    y = d["real_bill_next_52w"].astype(int)
    base = np.full(len(y), y.mean())
    print(
        f"\nOne year after the promise, still no bill: {len(d)} promises, "
        f"{y.sum()} got a bill in the next 52 weeks ({y.mean():.1%})"
    )
    print(
        f"  model: AUC {roc_auc_score(y, d['pred_p52']):.3f}, "
        f"Brier {brier_score_loss(y, d['pred_p52']):.4f} "
        f"(base-rate Brier {brier_score_loss(y, base):.4f})"
    )
    print("\n  By repetition in the first year (real share with a bill in the next 52 weeks):")
    bands = pd.cut(d["repeat_mentions_since_promise"], [-1, 0, 2, 1000], labels=["0", "1-2", "3+"])
    for band, g in d.groupby(bands, observed=True):
        print(
            f"    repeated {band:>3} times: n={len(g):4d}  model said {g['pred_p52'].mean():5.1%}"
            f"  real {g['real_bill_next_52w'].mean():5.1%}"
        )
    print("\n  Calibration (model said vs real):")
    bins = pd.cut(d["pred_p52"], [0, 0.05, 0.1, 0.2, 0.4, 1.0], include_lowest=True)
    for b, g in d.groupby(bins, observed=True):
        print(
            f"    {str(b):15s} n={len(g):4d}  mean predicted {g['pred_p52'].mean():5.1%}  "
            f"real {g['real_bill_next_52w'].mean():5.1%}"
        )
    print(f"\nwrote {OUT_UPDATING}")
    return d


if __name__ == "__main__":
    import sys

    updating() if "--updating" in sys.argv else main()
