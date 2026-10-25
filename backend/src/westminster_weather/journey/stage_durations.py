"""Predicted vs real time spent in each stage, for past cases that went promise -> law applied.

At the START of each stage the model for that stage predicts how long the stage will take,
using only information known that day:
- Stage 1 (promise -> first reading): stage-1 hazard model trained only on earlier promises,
  predicted first reading = promise + t_50 weeks.
- Stage 2 (first reading -> Royal Assent): Aleks's passage model, first stored rolling
  forecast for the bill (`backtest_forecasts`), Royal Assent p50.
- Stage 3 (Royal Assent -> law applied): stage 3 XGBoost (`journey.stage3_xgb`) at Royal
  Assent, median days until the law applies.
`--firm`: stage 1 starts at the idea's first firm commitment (firmness >= 2), not its first
mention; ideas never firmly committed are left out.
Read-only. Run: uv run python -m westminster_weather.journey.stage_durations [--firm]
"""

import sys
import warnings

import pandas as pd

from westminster_weather.journey import stage3_xgb
from westminster_weather.journey.journey_test import OUT_DIR, load
from westminster_weather.takeup import evaluate as ev
from westminster_weather.takeup.bill_text import first_readings
from westminster_weather.takeup.signals import TakeupData

warnings.filterwarnings("ignore")


def run(firm: bool = False) -> pd.DataFrame:
    td = TakeupData(firm_start=firm)
    T = td.tables
    data = load()
    bt, ac = data["bt"], data["ac"]
    fr = first_readings(T["stage_sittings"])
    links = (
        T["idea_bill_links"]
        .dropna(subset=["bill_id"])
        .merge(T["ideas"][["idea_id", "trigger_date"]], on="idea_id")
    )
    links["bill_id"] = links["bill_id"].astype(int)
    cases = links[links["bill_id"].isin(bt["bill_id"])].merge(ac, on="bill_id")
    cases = cases[cases["royal_assent_date"] <= "2025-06-30"]
    if firm:  # the promise is the first firm commitment
        m = td.visible_mentions(pd.Timestamp("2025-06-30"))
        starts = m.groupby("idea_id")["event_date"].min()
        cases = cases.assign(trigger_date=cases["idea_id"].map(starts)).dropna(
            subset=["trigger_date"]
        )
    cases = cases.sort_values("trigger_date")

    rows = []
    for _, c in cases.iterrows():
        bill = int(c["bill_id"])
        promise = pd.Timestamp(c["trigger_date"])
        first = pd.Timestamp(fr[bill])
        ra = pd.Timestamp(c["royal_assent_date"])
        applied = pd.Timestamp(c["applied"])
        row = {"act": c["act_title"], "promise": promise.date()}

        # stage 1, predicted at the promise
        pred1 = None
        if promise < first:
            train = td.get_takeup_signals(promise)
            train = train[pd.to_datetime(train["signal_date"]) < promise]
            model = ev.fit_hazard(td, train, promise)
            sigs = td.get_takeup_signals(promise)
            s = sigs[sigs["idea_id"] == c["idea_id"]]
            if not s.empty and not s["bill_date"].notna().any():
                s = s.sort_values("signal_date").iloc[0]
                tl = td.mention_timeline(promise)
                p = model.predict(
                    s,
                    tl[tl["signal_id"] == s["signal_id"]],
                    promise,
                    td.sitting_dates(promise),
                    td.week_probability(promise),
                )
                if pd.notna(p.get("t_50")):
                    pred1 = float(p["t_50"])
        row.update(stage1_pred_weeks=pred1, stage1_real_weeks=(first - promise).days / 7)

        # stage 2, predicted at first reading
        f = bt[(bt["bill_id"] == bill) & bt["ra_date_p50"].notna() & (bt["forecast_date"] < ra)]
        f = f.sort_values("forecast_date")
        pred2 = (pd.Timestamp(f.iloc[0]["ra_date_p50"]) - first).days / 7 if len(f) else None
        row.update(stage2_pred_weeks=pred2, stage2_real_weeks=(ra - first).days / 7)

        # stage 3, predicted at Royal Assent
        f3 = stage3_xgb.forecast(bill, ra.date())
        pred3 = f3["days"][0.5] / 7 if f3 is not None else None
        row.update(stage3_pred_weeks=pred3, stage3_real_weeks=(applied - ra).days / 7)
        rows.append(row)

    d = pd.DataFrame(rows)
    for k in (1, 2, 3):
        d[f"stage{k}_error_weeks"] = d[f"stage{k}_pred_weeks"] - d[f"stage{k}_real_weeks"]
    d.round(1).to_csv(OUT_DIR / out_name(firm), index=False)
    return d


def out_name(firm: bool) -> str:
    return "stage_durations_predicted_firm.csv" if firm else "stage_durations_predicted.csv"


def main() -> None:
    firm = "--firm" in sys.argv
    d = run(firm)
    print(f"{len(d)} cases (promise -> Government Bill -> Act -> law applied)\n")
    print(
        f"{'stage':40s} {'n':>3s} {'pred median':>12s} {'real median':>12s} "
        f"{'median |err|':>13s} {'within 4 wk':>12s}"
    )
    names = {
        1: f"Stage 1: {'firm commitment' if firm else 'promise'} -> 1st reading",
        2: "Stage 2: first reading -> RA",
        3: "Stage 3: RA -> law applied",
    }
    for k in (1, 2, 3):
        g = d.dropna(subset=[f"stage{k}_pred_weeks"])
        err = g[f"stage{k}_error_weeks"].abs()
        print(
            f"{names[k]:40s} {len(g):3d} {g[f'stage{k}_pred_weeks'].median():10.1f}wk "
            f"{g[f'stage{k}_real_weeks'].median():10.1f}wk {err.median():11.1f}wk "
            f"{(err <= 4).mean():11.0%}"
        )
    print(f"\nwrote {OUT_DIR / out_name(firm)}")


if __name__ == "__main__":
    main()
