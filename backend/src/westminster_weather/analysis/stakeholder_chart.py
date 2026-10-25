"""One clean summary chart of forecast error by stage, for stakeholders.

Reads the error table written by ``remaining_time_models`` (no retraining) and draws the
chosen model against the simple baseline, in plain language.

Run from ``backend/`` after ``remaining_time_models``::

    uv run python -m westminster_weather.analysis.stakeholder_chart
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from westminster_weather.analysis.remaining_time_dataset import STAGE_ANCHORS  # noqa: E402
from westminster_weather.analysis.remaining_time_models import REPORT_DIR  # noqa: E402

BASELINE = "B1: typical days left at this point"
PLAIN_LABELS = {
    "1st House: 1st reading": "Introduced\n(1st reading)",
    "1st House: 2nd reading": "2nd\nreading",
    "1st House: committee": "Committee",
    "1st House: report": "Report",
    "1st House: 3rd reading": "3rd\nreading",
    "2nd House: 1st reading": "1st\nreading",
    "2nd House: 2nd reading": "2nd\nreading",
    "2nd House: committee": "Committee",
    "2nd House: report": "Report",
    "2nd House: 3rd reading": "3rd\nreading",
    "Ping-pong": "Final\nagreement",
}
MODEL_COLOUR = "#2a78d6"
BASELINE_COLOUR = "#8f8d87"
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def plot(report_dir: Path = REPORT_DIR) -> Path:
    table = pd.read_csv(report_dir / "mae_by_stage.csv", index_col=0).reindex(STAGE_ANCHORS)
    model = json.loads((report_dir / "metrics.json").read_text())["best_model_by_cv"]
    bills = json.loads((report_dir / "metrics.json").read_text())["data"]["bills_test"]
    ours, base = table[model], table[BASELINE]
    x = range(len(table))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(x, ours, base, color=MODEL_COLOUR, alpha=0.08, lw=0)
    ax.plot(x, base, color=BASELINE_COLOUR, lw=2, marker="o", ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, solid_capstyle="round")
    ax.plot(x, ours, color=MODEL_COLOUR, lw=3, marker="o", ms=8,
            markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round")

    # Direct labels instead of a legend box.
    last = len(table) - 1
    ax.annotate("Simple rule of thumb", (last, base.iloc[-1]), xytext=(14, 26),
                textcoords="offset points", color=INK_SOFT, fontsize=11, va="bottom")
    ax.annotate(f"Our model\n±{ours.iloc[-1]:.0f} days at\nthe final stage",
                (last, ours.iloc[-1]), xytext=(14, -2), textcoords="offset points",
                color=INK, fontsize=11, fontweight="bold", va="center")

    # The three numbers worth remembering.
    ax.annotate(f"±{ours.iloc[0]:.0f} days\nat introduction", (0, ours.iloc[0]),
                xytext=(0.2, ours.iloc[0] * 0.55), textcoords="data", color=INK, fontsize=11,
                fontweight="bold", va="top",
                arrowprops={"arrowstyle": "-", "color": INK_SOFT, "lw": 0.8})
    gain = base - ours
    best_i = int(gain.values.argmax())
    ax.annotate(
        f"{gain.iloc[best_i]:.0f} days more accurate\nthan the rule of thumb",
        ((best_i), (ours.iloc[best_i] + base.iloc[best_i]) / 2), xytext=(40, 28),
        textcoords="offset points", color=INK_SOFT, fontsize=10, va="bottom",
        arrowprops={"arrowstyle": "-", "color": INK_SOFT, "lw": 0.8},
    )

    # Axes: recessive, plain language.
    ax.set_xticks(list(x), [PLAIN_LABELS[a] for a in STAGE_ANCHORS], fontsize=10,
                  color=INK_SOFT)
    ax.set_xlim(-0.4, last + 2.6)
    ax.set_ylim(0, max(base.max(), ours.max()) * 1.12)
    ax.set_ylabel("Average forecast error (days)", color=INK_SOFT)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK_SOFT)

    # Group the stages by House under the axis.
    for start, end, label in ((0, 4, "First House"), (5, 9, "Second House")):
        ax.annotate("", xy=(start - 0.3, -0.155), xytext=(end + 0.3, -0.155),
                    xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 1.5})
        ax.text((start + end) / 2, -0.17, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=INK_SOFT, fontsize=10)

    fig.text(0.06, 0.95, "Royal Assent forecasts get sharper as a bill moves through Parliament",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(0.06, 0.905,
             "Average error in predicted days to Royal Assent, by the stage at which the "
             f"forecast is made. Tested on {bills} bills that passed in 2019–2026.",
             fontsize=11, color=INK_SOFT)
    fig.text(0.06, 0.02,
             "Rule of thumb = typical time left for past bills at the same stage. "
             f"Our model = {model}, trained on 2006–2019 bills.",
             fontsize=9, color=INK_SOFT)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.25)

    path = report_dir / "0_summary_for_stakeholders.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(plot())
