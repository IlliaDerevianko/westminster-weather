"""Stage 2 check: Royal Assent predicted the day the bill came out vs the real Royal Assent.

For every Government Bill in Aleks's rolling backtest (`backtest_forecasts`) that got Royal
Assent, take its first forecast (made at first reading, information known that day only) and
compare the predicted Royal Assent (p50, 80% band p10-p90) with the real date.
Error = predicted - real, in weeks (positive = predicted too late).
Read-only. Run: uv run python -m westminster_weather.journey.stage2_ra
"""

import warnings

import pandas as pd

from westminster_weather.journey.journey_test import OUT_DIR, load

warnings.filterwarnings("ignore")
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#c3c2b7", "#0b0b0b", "#52514e"


def run() -> pd.DataFrame:
    bt = load()["bt"]
    bt = bt[bt["actual_ra_date"].notna() & bt["ra_date_p50"].notna()].copy()
    for c in ("ra_date_p10", "ra_date_p90", "actual_ra_date"):
        bt[c] = pd.to_datetime(bt[c])
    first = bt.sort_values("forecast_date").drop_duplicates("bill_id")
    first = first[first["forecast_date"] < first["actual_ra_date"]]

    def weeks(col: str) -> pd.Series:
        return (first[col] - first["forecast_date"]).dt.days / 7

    d = pd.DataFrame(
        {
            "bill_id": first["bill_id"],
            "forecast_date": first["forecast_date"].dt.date,
            "state_at_forecast": first["state_at_forecast"],
            "predicted_ra": first["ra_date_p50"].dt.date,
            "real_ra": first["actual_ra_date"].dt.date,
            "pred_weeks": weeks("ra_date_p50"),
            "pred_p10_weeks": weeks("ra_date_p10"),
            "pred_p90_weeks": weeks("ra_date_p90"),
            "real_weeks": weeks("actual_ra_date"),
        }
    )
    d["error_weeks"] = d["pred_weeks"] - d["real_weeks"]
    d["inside_80"] = (d["pred_p10_weeks"] <= d["real_weeks"]) & (
        d["real_weeks"] <= d["pred_p90_weeks"]
    )
    d = d.sort_values("real_weeks").reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.round(1).to_csv(OUT_DIR / "stage2_ra_pred_vs_real.csv", index=False)
    return d


def chart(d: pd.DataFrame, titles: pd.Series | None = None) -> pd.Series:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": INK})
    s = summary(d)
    title = "Stage 2: Royal Assent predicted at first reading (passage model, rolling backtest)"

    # 1: per bill, predicted (with 80% band) vs real time to Royal Assent
    fig, a = plt.subplots(figsize=(16, 6.5))
    x = range(len(d))
    a.vlines(x, d["pred_p10_weeks"], d["pred_p90_weeks"], color="#b7d3f6", lw=3, label="80% band")
    a.vlines(x, d["pred_weeks"], d["real_weeks"], color=GRAY, lw=1)
    a.scatter(x, d["pred_weeks"], s=12, color=BLUE, zorder=3, label="predicted (p50)")
    a.scatter(x, d["real_weeks"], s=12, color=ORANGE, zorder=3, label="real")
    a.set_xticks([])
    a.set_xlabel(f"{int(s['n'])} Government Bills, sorted by real time to Royal Assent")
    a.set_ylabel("weeks from first reading to Royal Assent")
    a.set_title(f"{title}\nEach bill: predicted on the day it came out vs real", loc="left")
    a.legend(loc="upper left", frameon=False)
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#ecebe7", lw=0.6)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "stage2_ra_by_bill.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    fig, (b, c) = plt.subplots(1, 2, figsize=(13, 5.5))
    # 2: predicted vs real
    top = max(d["pred_weeks"].max(), d["real_weeks"].max()) * 1.05
    b.plot([0, top], [0, top], color=GRAY, lw=1, ls="--", label="perfect")
    b.scatter(d["real_weeks"], d["pred_weeks"], s=16, color=BLUE, alpha=0.8, label="bill")
    b.set_xlim(0, top)
    b.set_ylim(0, top)
    b.set_aspect("equal")
    b.set_xlabel("real weeks to Royal Assent")
    b.set_ylabel("predicted weeks to Royal Assent")
    b.set_title("Predicted vs real", loc="left")
    b.legend(loc="upper left", frameon=False)
    # 3: error distribution
    err = d["error_weeks"]
    c.hist(err, bins=range(int(err.min()) - 4, int(err.max()) + 8, 4), color=BLUE, rwidth=0.85)
    c.axvline(0, color=INK, lw=1)
    c.set_xlabel("error, weeks (predicted - real; + = predicted too late)")
    c.set_ylabel("bills")
    c.set_title("Error", loc="left")
    c.text(
        0.02,
        0.97,
        f"n = {int(s['n'])} bills\nmedian |error| {s['median_abs']:.1f} wk\n"
        f"mean |error| {s['mean_abs']:.1f} wk\nwithin 4 wk {s['within_4']:.0%}\n"
        f"inside 80% band {s['inside_80']:.0%}",
        transform=c.transAxes,
        ha="left",
        va="top",
        color=MUTED,
    )
    for ax in (b, c):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    plt.tight_layout()
    path = OUT_DIR / "stage2_ra_pred_vs_real.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return s


def summary(d: pd.DataFrame) -> pd.Series:
    e = d["error_weeks"]
    return pd.Series(
        {
            "n": len(d),
            "median_abs": e.abs().median(),
            "mean_abs": e.abs().mean(),
            "median_signed": e.median(),
            "within_4": (e.abs() <= 4).mean(),
            "inside_80": d["inside_80"].mean(),
            "pred_median": d["pred_weeks"].median(),
            "real_median": d["real_weeks"].median(),
        }
    )


def main() -> None:
    d = run()
    s = chart(d)
    print(f"Stage 2, forecast at first reading vs real Royal Assent ({int(s['n'])} bills):")
    print(f"  predicted median {s['pred_median']:.1f} wk, real median {s['real_median']:.1f} wk")
    print(
        f"  median |error| {s['median_abs']:.1f} wk, mean |error| {s['mean_abs']:.1f} wk, "
        f"median signed {s['median_signed']:+.1f} wk"
    )
    print(f"  within 4 weeks {s['within_4']:.0%}, real date inside 80% band {s['inside_80']:.0%}")
    print(d["state_at_forecast"].value_counts().to_string())
    print(f"\nwrote stage2_ra_by_bill.png, stage2_ra_pred_vs_real.png/.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
