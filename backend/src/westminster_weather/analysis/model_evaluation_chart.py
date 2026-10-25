"""One stakeholder chart of how well the passage model's forecasts held up in the backtest.

Reads the existing backtest results (no re-running): ``reports/metrics.json`` (Royal Assent date
error by the stage the forecast was made at) and ``reports/deadline_calibration.json`` (chance of
Royal Assent before a deadline vs how often it happened, test era, as served).

Run from ``backend/``::

    uv run python -m westminster_weather.analysis.model_evaluation_chart
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path(__file__).resolve().parents[3] / "reports"
OUT = REPORTS / "model_evaluation_for_stakeholders.png"

STAGES = {
    "commons_1st_reading": "1st\nreading",
    "commons_2nd_reading": "2nd\nreading",
    "commons_committee": "Committee",
    "commons_report_stage": "Report",
    "commons_3rd_reading": "3rd\nreading",
    "lords_1st_and_2nd_reading": "1st & 2nd\nreading",
    "lords_committee": "Committee",
    "lords_report_and_3rd_reading": "Report &\n3rd reading",
    "ping_pong": "Final\nagreement",
}
N_COMMONS = 5  # first five stages are the Commons, the rest the Lords and ping-pong

MODEL_COLOUR = "#2a78d6"
BASELINE_COLOUR = "#8f8d87"
INK, INK_SOFT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK_SOFT, length=0)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)


def plot(reports: Path = REPORTS, out: Path = OUT) -> Path:
    backtest = json.loads((reports / "metrics.json").read_text())["backtest"]
    deadlines = json.loads((reports / "deadline_calibration.json").read_text())
    by_stage = backtest["mae_days_by_start_state"]
    ours = [by_stage[s]["p50_err"] for s in STAGES]
    rule = [by_stage[s]["b1_err"] for s in STAGES]
    mae = backtest["mae_days"]
    bins = deadlines.get("table_test_after") or deadlines["table_test"]
    brier = deadlines["brier_test"]
    model_brier = brier.get("model_recalibrated", brier["model"])
    better = 1 - model_brier / brier["baseline_historical_average"]
    seventy = next(b for b in bins if b["bin"] == "70-80%")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.9), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.75, 1]}
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.25, wspace=0.22)

    fig.text(0.06, 0.94, "Our forecasts beat the rule of thumb, and their chances hold up",
             fontsize=19, fontweight="bold", color=INK)
    fig.text(0.06, 0.895,
             f"Back-tested on {backtest['n_bills']} government bills from 2019–2026, each "
             "forecast made using only what was known on that day.",
             fontsize=12, color=INK_SOFT)

    # Panel 1: Royal Assent date error by stage
    _style(ax1)
    x = range(len(STAGES))
    ax1.fill_between(x, ours, rule, color=MODEL_COLOUR, alpha=0.08, lw=0)
    ax1.plot(x, rule, color=BASELINE_COLOUR, lw=2, marker="o", ms=6,
             markeredgecolor=SURFACE, markeredgewidth=1.5,
             label=f"Rule of thumb  ·  average ±{mae['B1']:.0f} days")
    ax1.plot(x, ours, color=MODEL_COLOUR, lw=3, marker="o", ms=8,
             markeredgecolor=SURFACE, markeredgewidth=2,
             label=f"Our model  ·  average ±{mae['model']:.0f} days")
    ax1.text(0.0, 1.1, "When will it become law?", transform=ax1.transAxes,
             fontsize=13.5, fontweight="bold", color=INK)
    ax1.text(0.0, 1.035, "Average error in the predicted Royal Assent date, by the stage the "
             "forecast was made at", transform=ax1.transAxes, fontsize=10.5, color=INK_SOFT)
    ax1.set_ylabel("Average error (days)", color=INK_SOFT)
    ax1.set_ylim(0, 185)
    ax1.set_xticks(list(x), list(STAGES.values()), fontsize=9.5)
    ax1.set_xlim(-0.4, len(STAGES) - 0.6)
    for lo, hi, name in ((0, N_COMMONS - 1, "Commons"), (N_COMMONS, len(STAGES) - 1, "Lords")):
        ax1.annotate("", xy=(lo - 0.3, -0.155), xytext=(hi + 0.3, -0.155),
                     xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                     arrowprops={"arrowstyle": "-", "color": GRID, "lw": 1.5})
        ax1.text((lo + hi) / 2, -0.175, name, transform=ax1.get_xaxis_transform(),
                 ha="center", va="top", color=INK_SOFT, fontsize=10.5)
    handles, labels = ax1.get_legend_handles_labels()
    leg = ax1.legend(handles[::-1], labels[::-1], loc="upper right", frameon=False,
                     fontsize=11.5, handlelength=2.2)
    leg.get_texts()[0].set_fontweight("bold")
    leg.get_texts()[0].set_color(INK)
    leg.get_texts()[1].set_color(INK_SOFT)
    cut = 1 - mae["model"] / mae["B1"]
    ax1.annotate(f"{cut:.0%} smaller error on average;\ncloser at every stage",
                 xy=(6, (ours[6] + rule[6]) / 2), xytext=(5.0, 128), fontsize=11,
                 fontweight="bold", color=INK,
                 arrowprops={"arrowstyle": "-", "color": INK_SOFT, "lw": 1})

    # Panel 2: deadline chances, said vs happened
    _style(ax2)
    ax2.grid(axis="x", color=GRID, lw=1)
    ax2.plot([0, 1], [0, 1], color=BASELINE_COLOUR, lw=1.5, ls=(0, (4, 4)))
    ax2.text(0.97, 0.9, "perfect", rotation=38, ha="right", va="bottom", color=MUTED,
             fontsize=10)
    said = [b["predicted"] for b in bins]
    happened = [b["actual"] for b in bins]
    ax2.plot(said, happened, color=MODEL_COLOUR, lw=2, alpha=0.5)
    ax2.scatter(said, happened, s=70, color=MODEL_COLOUR, edgecolor=SURFACE, lw=2, zorder=3)
    ax2.text(0.0, 1.1, "How likely is it by a deadline?", transform=ax2.transAxes,
             fontsize=13.5, fontweight="bold", color=INK)
    ax2.text(0.0, 1.035, "Chance we gave vs how often it happened", transform=ax2.transAxes,
             fontsize=10.5, color=INK_SOFT)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax2.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax2.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax2.set_xlabel("We said", color=INK_SOFT)
    ax2.set_ylabel("It happened", color=INK_SOFT)
    ax2.annotate(f"Said about 70%:\nhappened {seventy['actual']:.0%} of the time",
                 xy=(seventy["predicted"], seventy["actual"]), xytext=(0.05, 0.8),
                 fontsize=11, fontweight="bold", color=INK,
                 arrowprops={"arrowstyle": "-", "color": INK_SOFT, "lw": 1})
    ax2.text(0.97, 0.05,
             f"{better:.0%} more accurate than always\nguessing the historical average",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=10.5, color=INK_SOFT)

    n_deadline = deadlines["n_forecasts"]["test"]
    fig.text(0.06, 0.035,
             f"Rule of thumb = typical time left for past bills at the same stage. Deadlines = "
             f"next Christmas recess, next summer recess and session end ({n_deadline:,} "
             "forecasts).\nModel trained on 2006–2019 bills only; deadline chances "
             "recalibrated on 2014–2019 forecasts. Accuracy vs average: Brier score.",
             fontsize=9.5, color=MUTED)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot())
