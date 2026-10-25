"""End-to-end journey test: promise -> bill -> Act -> law applied.

For a sample of past cases that went all the way, we stop at a few checkpoints in each stage
(not every step) and ask: "when will this law be applied?" using only information known on
that day. Error = |predicted - real| date the law first applies, in weeks; it should shrink
from stage 1 to stage 3.

Law applied = date of the first commencement order if the Act needed orders, else the date its
sections started on Royal Assent (the data cannot show "fully in force").

Predicted law-applied date at each checkpoint:
- Stage 1 (promise, +1 year): stage-1 hazard model (trained only on earlier promises) gives
  the expected first-reading date (t_50), plus the median first reading -> Royal Assent and
  median Royal Assent -> law applied of Acts finished before that day.
- Stage 2 (first reading, a middle stage, last stage before Royal Assent): Aleks's passage
  model (rolling backtest stored in `backtest_forecasts`) gives the Royal Assent date (p50),
  plus the median Royal Assent -> law applied of Acts finished before that day.
- Stage 3 (Royal Assent, +13 weeks): stage 3 XGBoost (`journey.stage3_xgb`): median date the
  law applies.

Read-only. Run: uv run python -m westminster_weather.journey.journey_test [n_cases]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from westminster_weather.db import get_sync_pooled_engine
from westminster_weather.journey import stage3_xgb
from westminster_weather.takeup import evaluate as ev
from westminster_weather.takeup.bill_text import first_readings
from westminster_weather.takeup.signals import TakeupData

warnings.filterwarnings("ignore")
OUT_DIR = Path("reports/journey")
SEED = 7
CHECKPOINTS = [
    "S1: promise",
    "S1: +1 year",
    "S2: first reading",
    "S2: middle stage",
    "S2: last stage",
    "S3: Royal Assent",
    "S3: +3 months",
]


def load() -> dict[str, pd.DataFrame]:
    e = get_sync_pooled_engine()
    try:
        bt = pd.read_sql(text("select * from backtest_forecasts"), e)
        ac = pd.read_sql(text("select * from act_commencements"), e)
    finally:
        e.dispose()
    for c in ("forecast_date", "ra_date_p50"):
        bt[c] = pd.to_datetime(bt[c])
    for c in ("royal_assent_date", "first_commencement_order_date", "some_sections_start_date"):
        ac[c] = pd.to_datetime(ac[c])
    ac["applied"] = np.where(
        ac["commencement_orders_count"] > 0,
        ac["first_commencement_order_date"],
        ac["some_sections_start_date"].fillna(ac["royal_assent_date"]),
    )
    ac["applied"] = pd.to_datetime(ac["applied"])
    return {"bt": bt, "ac": ac}


def medians(ac: pd.DataFrame, fr: pd.Series, as_of: pd.Timestamp) -> tuple[float, float]:
    """Median days first reading -> Royal Assent and Royal Assent -> applied, for Acts whose
    law had applied before ``as_of`` (no future information)."""
    done = ac[ac["applied"] < as_of].copy()
    done["fr"] = pd.to_datetime(done["bill_id"].map(fr))
    fr_ra = (done["royal_assent_date"] - done["fr"]).dt.days.dropna()
    ra_ap = (done["applied"] - done["royal_assent_date"]).dt.days.dropna()
    return float(fr_ra.median()), float(ra_ap.median())


def run(n_cases: int = 30) -> pd.DataFrame:
    td = TakeupData()
    T = td.tables
    data = load()
    bt, ac = data["bt"], data["ac"]
    fr = first_readings(T["stage_sittings"])
    links = (
        T["idea_bill_links"]
        .dropna(subset=["bill_id"])
        .merge(T["ideas"][["idea_id", "trigger_date", "topic"]], on="idea_id")
    )
    links["bill_id"] = links["bill_id"].astype(int)
    cases = links[links["bill_id"].isin(bt["bill_id"])].merge(ac, on="bill_id")
    cases = cases[cases["royal_assent_date"] <= "2025-06-30"]
    cases = cases.sample(min(n_cases, len(cases)), random_state=SEED).sort_values("trigger_date")
    titles = T["bills"].set_index("bill_id")["short_title"]

    rows = []
    for _, c in cases.iterrows():
        bill, applied = int(c["bill_id"]), pd.Timestamp(c["applied"])
        trig, fr_date = pd.Timestamp(c["trigger_date"]), pd.Timestamp(fr[bill])
        ra = pd.Timestamp(c["royal_assent_date"])

        def add(
            label: str, as_of: pd.Timestamp, pred: pd.Timestamp, case=c, b=bill, applied=applied
        ) -> None:
            rows.append(
                {
                    "checkpoint": label,
                    "idea_id": case["idea_id"],
                    "bill_id": b,
                    "bill": titles.get(b),
                    "as_of": as_of.date(),
                    "predicted_applied": pred.date(),
                    "real_applied": applied.date(),
                    "weeks_left_predicted": round((pred - as_of).days / 7, 1),
                    "weeks_left_real": round((applied - as_of).days / 7, 1),
                    "abs_error_weeks": abs((pred - applied).days) / 7,
                }
            )

        # ---- stage 1: promise, and one year later (if still no bill)
        train = td.get_takeup_signals(trig)
        train = train[pd.to_datetime(train["signal_date"]) < trig]
        model = ev.fit_hazard(td, train, trig)
        for label, as_of in (("S1: promise", trig), ("S1: +1 year", trig + pd.Timedelta(weeks=52))):
            if as_of >= fr_date:
                continue
            sigs = td.get_takeup_signals(as_of)
            s = sigs[sigs["idea_id"] == c["idea_id"]]
            if s.empty or s["bill_date"].notna().any():
                continue
            s = s.sort_values("signal_date").iloc[0]
            tl = td.mention_timeline(as_of)
            p = model.predict(
                s,
                tl[tl["signal_id"] == s["signal_id"]],
                as_of,
                td.sitting_dates(as_of),
                td.week_probability(as_of),
            )
            t50 = p.get("t_50") if pd.notna(p.get("t_50")) else 260
            pred_fr = pd.Timestamp(s["signal_date"]) + pd.Timedelta(weeks=int(t50))
            m_fr_ra, m_ra_ap = medians(ac, fr, as_of)
            add(label, as_of, pred_fr + pd.Timedelta(days=m_fr_ra + m_ra_ap))

        # ---- stage 2: first, middle and last forecast before Royal Assent
        f = bt[(bt["bill_id"] == bill) & bt["ra_date_p50"].notna()].sort_values("forecast_date")
        f = f[f["forecast_date"] < ra]
        if len(f):
            picks = {"S2: first reading": f.iloc[0], "S2: last stage": f.iloc[-1]}
            if len(f) >= 3:
                picks["S2: middle stage"] = f.iloc[len(f) // 2]
            for label, r in picks.items():
                as_of = pd.Timestamp(r["forecast_date"])
                _, m_ra_ap = medians(ac, fr, as_of)
                add(label, as_of, pd.Timestamp(r["ra_date_p50"]) + pd.Timedelta(days=m_ra_ap))

        # ---- stage 3: at Royal Assent, and 13 weeks later (if the law has not applied yet)
        for label, as_of in (
            ("S3: Royal Assent", ra),
            ("S3: +3 months", ra + pd.Timedelta(weeks=13)),
        ):
            if as_of > applied or (label != "S3: Royal Assent" and as_of >= applied):
                continue
            f = stage3_xgb.forecast(bill, as_of.date())
            if f is None:
                continue
            add(label, as_of, as_of + pd.Timedelta(days=f["days"][0.5]))

        print(
            f"  {c['idea_id']} -> {titles.get(bill)} (applied {applied.date()}): "
            f"{sum(r['idea_id'] == c['idea_id'] for r in rows)} checkpoints"
        )

    d = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_DIR / "journey_test.csv", index=False)
    return d


def chart(d: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = d.groupby("checkpoint")["abs_error_weeks"].agg(["mean", "median", "count"])
    g = g.reindex([c for c in CHECKPOINTS if c in g.index])
    colors = [
        "#5b8def" if i.startswith("S1") else "#f2a541" if i.startswith("S2") else "#3fb37f"
        for i in g.index
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(g.index, g["mean"], color=colors)
    ax.scatter(g.index, g["median"], color="black", zorder=3, label="median error")
    for b, (_, r) in zip(bars, g.iterrows(), strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 3,
            f"{r['mean']:.0f} wk\n(n={int(r['count'])})",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("Error in the predicted 'law applied' date (weeks)")
    ax.set_title(
        "How far off is 'when will this law apply?' at each checkpoint\n"
        "Stage 1 (promise) -> Stage 2 (in Parliament) -> Stage 3 (after Royal Assent)"
    )
    ax.legend(loc="upper right")
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = OUT_DIR / "journey_error_by_stage.png"
    fig.savefig(path, dpi=150)
    return path


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    d = run(n)
    print("\nError in weeks by checkpoint (mean / median, n):")
    g = d.groupby("checkpoint")["abs_error_weeks"].agg(["mean", "median", "count"])
    print(g.reindex([c for c in CHECKPOINTS if c in g.index]).round(1).to_string())
    print(f"\nchart: {chart(d)}")
    s3 = d[d["checkpoint"] == "S3: Royal Assent"]
    print("\nStage 3 at Royal Assent: weeks until the law applies, predicted vs real")
    print(s3[["bill", "weeks_left_predicted", "weeks_left_real"]].to_string(index=False))


if __name__ == "__main__":
    main()
