"""One stakeholder chart of how well the passage model's deadline chances held up on the test set.

Reads the existing calibration results (no re-running): ``reports/deadline_calibration.json``,
the chance of Royal Assent before a deadline vs how often it happened, on held-out test-era
bills, as served.

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

MODEL_COLOUR = "#2a78d6"
REFERENCE_COLOUR = "#8f8d87"
INK, INK_SOFT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def plot(reports: Path = REPORTS, out: Path = OUT) -> Path:
    deadlines = json.loads((reports / "deadline_calibration.json").read_text())
    bins = deadlines.get("table_test_after") or deadlines["table_test"]
    brier = deadlines["brier_test"]
    model_brier = brier.get("model_recalibrated", brier["model"])
    better = 1 - model_brier / brier["baseline_historical_average"]
    seventy = next(b for b in bins if b["bin"] == "70-80%")
    n_forecasts = deadlines["n_forecasts"]["test"]
    n_bills = deadlines["n_bills"]["test"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.11, right=0.95, top=0.8, bottom=0.17)

    fig.text(0.04, 0.945, "When we give a chance, it holds up", fontsize=19,
             fontweight="bold", color=INK)
    fig.text(0.04, 0.905, "Chance of Royal Assent before a deadline: what we said vs how often "
             "it happened", fontsize=12, color=INK_SOFT)
    fig.text(0.04, 0.872,
             f"Held-out test set: {n_bills} government bills, 2019–2026 ({n_forecasts:,} "
             "forecasts), never used in training.",
             fontsize=12, color=INK_SOFT)

    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=11)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)

    ax.plot([0, 1], [0, 1], color=REFERENCE_COLOUR, lw=1.5, ls=(0, (4, 4)))
    ax.text(0.9, 0.92, "perfect", rotation=45, ha="center", va="bottom", color=MUTED,
            fontsize=10.5, rotation_mode="anchor")
    said = [b["predicted"] for b in bins]
    happened = [b["actual"] for b in bins]
    ax.plot(said, happened, color=MODEL_COLOUR, lw=2, alpha=0.5)
    ax.scatter(said, happened, s=90, color=MODEL_COLOUR, edgecolor=SURFACE, lw=2, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xlabel("We said", color=INK_SOFT, fontsize=12)
    ax.set_ylabel("It happened", color=INK_SOFT, fontsize=12)

    ax.text(0.03, 0.97, "TEST SET", transform=ax.transAxes, va="top", fontsize=9.5,
            fontweight="bold", color=MODEL_COLOUR,
            bbox={"boxstyle": "round,pad=0.35", "fc": SURFACE, "ec": MODEL_COLOUR, "lw": 1})
    ax.annotate(f"Said about 70%:\nhappened {seventy['actual']:.0%} of the time",
                xy=(seventy["predicted"], seventy["actual"]), xytext=(0.06, 0.78),
                fontsize=12.5, fontweight="bold", color=INK,
                arrowprops={"arrowstyle": "-", "color": INK_SOFT, "lw": 1})
    ax.text(0.97, 0.05,
            f"{better:.0%} more accurate than always\nguessing the historical average",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12, color=INK_SOFT)

    fig.text(0.04, 0.035,
             "Deadlines = next Christmas recess, next summer recess and end of the session. "
             "Model trained on 2006–2019\nbills only; chances recalibrated on 2014–2019 "
             "forecasts. Accuracy vs average: Brier score on the test set.",
             fontsize=9.5, color=MUTED)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot())
