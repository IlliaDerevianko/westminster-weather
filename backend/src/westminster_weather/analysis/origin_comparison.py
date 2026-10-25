"""How does a bill's progress differ depending on whether it starts in the Commons or the Lords?

For every UK Government Bill this measures time spent in each House, the final stretch from
the second House's 3rd reading to Royal Assent, and the total time to Royal Assent, then
compares Commons-first with Lords-first bills. It also compares how often each group passes.

Run from ``backend/``::

    uv run python -m westminster_weather.analysis.origin_comparison

Charts, a CSV of per-bill durations and a summary are written to
``backend/reports/origin_comparison/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "origin_comparison"

# Bills API stage IDs marking the start and end of each House's stages.
COMMONS_1ST_READING, COMMONS_3RD_READING = 6, 10
LORDS_1ST_READING, LORDS_3RD_READING = 1, 5
ROYAL_ASSENT = 11

# Same cut-off as stage_durations: bills through their first House this fast are emergency,
# finance or wash-up bills (all finance/supply bills must start in the Commons).
FAST_TRACK_MAX_FIRST_HOUSE_DAYS = 14

ORIGIN_COLOURS = {"Commons": "#2e7d5b", "Lords": "#a8323e"}
OUTCOME_ORDER = ["royal_assent", "fell_session_end", "fell_dissolution", "withdrawn"]
OUTCOME_LABELS = {
    "royal_assent": "Royal Assent",
    "fell_session_end": "Fell at session end",
    "fell_dissolution": "Fell at dissolution",
    "withdrawn": "Withdrawn",
}
OUTCOME_COLOURS = {
    "royal_assent": "#3b6ea5",
    "fell_session_end": "#e0a030",
    "fell_dissolution": "#c0504d",
    "withdrawn": "#8c8c8c",
}


def house_durations(bills: pd.DataFrame, sittings: pd.DataFrame) -> pd.DataFrame:
    """Per bill: calendar days in the Commons, in the Lords, end-game and total to Royal Assent.

    A House's time runs from its 1st reading to its 3rd reading. The end-game runs from the
    second House's 3rd reading to Royal Assent (ping-pong and waiting for Assent). NaN means
    the bill never finished that part.
    """
    first = (
        sittings.assign(event_date=pd.to_datetime(sittings["event_date"]))
        .groupby(["bill_id", "api_stage_id"])["event_date"]
        .min()
        .unstack()
    )

    def col(stage_id: int) -> pd.Series:
        return first[stage_id] if stage_id in first.columns else pd.Series(pd.NaT, first.index)

    def days(start: pd.Series, end: pd.Series) -> pd.Series:
        d = (end - start).dt.days.astype(float)
        return d.where(d >= 0)

    origin = bills.set_index("bill_id")["originating_house"].reindex(first.index)
    commons_first = origin == "Commons"
    second_house_end = col(LORDS_3RD_READING).where(commons_first, col(COMMONS_3RD_READING))
    first_house_start = col(COMMONS_1ST_READING).where(commons_first, col(LORDS_1ST_READING))

    out = pd.DataFrame(
        {
            "origin": origin,
            "commons_days": days(col(COMMONS_1ST_READING), col(COMMONS_3RD_READING)),
            "lords_days": days(col(LORDS_1ST_READING), col(LORDS_3RD_READING)),
            "end_game_days": days(second_house_end, col(ROYAL_ASSENT)),
            "total_to_ra_days": days(first_house_start, col(ROYAL_ASSENT)),
        }
    )
    out["first_house_days"] = out["commons_days"].where(commons_first, out["lords_days"])
    out["second_house_days"] = out["lords_days"].where(commons_first, out["commons_days"])
    out["fast_tracked"] = out["first_house_days"] <= FAST_TRACK_MAX_FIRST_HOUSE_DAYS
    return out.join(bills.set_index("bill_id")["outcome"])


def compare(durations: pd.DataFrame, measure: str) -> dict:
    """Median days for each origin and a Mann-Whitney U test of the difference."""
    c = durations.loc[durations["origin"] == "Commons", measure].dropna()
    lo = durations.loc[durations["origin"] == "Lords", measure].dropna()
    p = float(stats.mannwhitneyu(c, lo).pvalue) if len(c) and len(lo) else float("nan")
    return {
        "commons_first": {"median": float(c.median()), "n": int(len(c))},
        "lords_first": {"median": float(lo.median()), "n": int(len(lo))},
        "mann_whitney_p": p,
    }


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


MEASURES = (
    ("first_house_days", "First House\n(1st → 3rd reading)"),
    ("second_house_days", "Second House\n(1st → 3rd reading)"),
    ("end_game_days", "Second House 3rd reading\n→ Royal Assent"),
    ("total_to_ra_days", "Total: introduction\n→ Royal Assent"),
)


def plot_journey(durations: pd.DataFrame, path: Path) -> None:
    """Box plots of each part of the journey, Commons-first vs Lords-first, all vs regular."""
    groups = [
        ("All bills", durations),
        (
            f"Regular bills only (first House > {FAST_TRACK_MAX_FIRST_HOUSE_DAYS} days)",
            durations[~durations["fast_tracked"]],
        ),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharey=True, layout="constrained")
    for ax, (title, df) in zip(axes, groups, strict=True):
        positions, data, colours = [], [], []
        for i, (m, _) in enumerate(MEASURES):
            for k, origin in enumerate(("Commons", "Lords")):
                positions.append(i * 3 + k)
                data.append(df.loc[df["origin"] == origin, m].dropna() + 1)
                colours.append(ORIGIN_COLOURS[origin])
        bp = ax.boxplot(
            data, positions=positions, widths=0.75, patch_artist=True, showfliers=False,
            medianprops={"color": "black", "lw": 1.5},
        )
        for patch, colour in zip(bp["boxes"], colours, strict=True):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
        # Median label just above each box's upper whisker, so it never covers the box.
        for idx, (pos, d) in enumerate(zip(positions, data, strict=True)):
            if len(d):
                top = bp["caps"][2 * idx + 1].get_ydata()[0]
                ax.text(pos, top * 1.15, f"median {d.median() - 1:.0f}d\nn={len(d)}",
                        ha="center", va="bottom", fontsize=8)
        for i, (m, _) in enumerate(MEASURES):
            p = compare(df, m)["mann_whitney_p"]
            ax.text(i * 3 + 0.5, 6e3, f"difference: {_stars(p)}", ha="center", fontsize=9,
                    color="0.3")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 1.5e4)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(MEASURES))], [lbl for _, lbl in MEASURES])
        ax.set_ylabel("Calendar days + 1 (log scale)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(True, axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ORIGIN_COLOURS[o], alpha=0.75)
        for o in ("Commons", "Lords")
    ]
    fig.legend(handles, ["Starts in Commons", "Starts in Lords"], loc="outside lower center",
               fontsize=9, ncols=2)
    fig.suptitle(
        "How long each part of the journey takes, by the House a bill starts in\n"
        "Box = middle 50% of bills, black line = median.  Difference test: "
        "* p<0.05  ** p<0.01  *** p<0.001  n.s. = not significant",
        fontsize=11,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_house_by_order(durations: pd.DataFrame, path: Path) -> None:
    """Time in each House when it is the first House vs the second House (regular bills)."""
    df = durations[~durations["fast_tracked"]]
    fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
    ticks, tick_labels = [], []
    for i, (house, col, other) in enumerate(
        (("Commons", "commons_days", "Lords"), ("Lords", "lords_days", "Commons"))
    ):
        for k, (order, origin) in enumerate((("first", house), ("second", other))):
            d = df.loc[df["origin"] == origin, col].dropna()
            med = d.median()
            q1, q3 = d.quantile([0.25, 0.75])
            pos = i * 3 + k
            ax.bar(pos, med, color=ORIGIN_COLOURS[house], alpha=0.95 if k == 0 else 0.45,
                   width=0.8)
            ax.errorbar(pos, med, yerr=[[med - q1], [q3 - med]], color="black", capsize=4)
            ax.text(pos, q3 + 4, f"{med:.0f} days\nn={len(d)}", ha="center", va="bottom",
                    fontsize=9)
            ticks.append(pos)
            tick_labels.append(f"{house}\ngoes {order}")
        p = stats.mannwhitneyu(
            df.loc[df["origin"] == house, col].dropna(),
            df.loc[df["origin"] == other, col].dropna(),
        ).pvalue
        ax.text(i * 3 + 0.5, 200, f"difference: {_stars(p)}", ha="center", fontsize=9,
                color="0.3")
    ax.set_xticks(ticks, tick_labels, fontsize=9)
    ax.set_ylabel("Median calendar days in that House (1st → 3rd reading)")
    ax.set_ylim(0, 215)
    ax.set_title(
        "Is a House faster when it goes first or second? (regular bills)\n"
        "Bar = median, whiskers = middle 50% of bills.  * p<0.05  ** p<0.01  *** p<0.001",
        fontsize=11,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def outcome_shares(bills: pd.DataFrame) -> pd.DataFrame:
    closed = bills[bills["outcome"] != "in_progress"]
    shares = pd.crosstab(closed["originating_house"], closed["outcome"], normalize="index")
    counts = closed["originating_house"].value_counts()
    return shares.reindex(columns=OUTCOME_ORDER, fill_value=0).assign(n=counts)


def plot_outcomes(shares: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    origins = ["Commons", "Lords"]
    left = np.zeros(len(origins))
    for outcome in OUTCOME_ORDER:
        vals = shares.loc[origins, outcome].values * 100
        ax.barh(origins, vals, left=left, color=OUTCOME_COLOURS[outcome],
                label=OUTCOME_LABELS[outcome])
        for y, (v, lft) in enumerate(zip(vals, left, strict=True)):
            if v >= 3:
                ax.text(lft + v / 2, y, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=9)
        left += vals
    ax.set_yticks(range(len(origins)),
                  [f"Starts in {o}\n(n={int(shares.loc[o, 'n'])})" for o in origins])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of closed bills (%)")
    ax.invert_yaxis()
    ax.legend(ncols=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title("How bills end, by the House they start in (in-progress bills excluded)",
                 fontsize=11, pad=28)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(report_dir: Path = REPORT_DIR) -> dict:
    from westminster_weather.db import fetch_bills_df, fetch_stage_sittings_df

    report_dir.mkdir(parents=True, exist_ok=True)
    bills = fetch_bills_df()
    sittings = fetch_stage_sittings_df()
    sittings = sittings[sittings["bill_type_id"] == 1]  # DATA_SCOPE: Government Bills only

    durations = house_durations(bills, sittings)
    regular = durations[~durations["fast_tracked"]]
    shares = outcome_shares(bills)

    plot_journey(durations, report_dir / "1_journey_by_origin.png")
    plot_house_by_order(durations, report_dir / "2_house_first_vs_second.png")
    plot_outcomes(shares, report_dir / "3_outcomes_by_origin.png")
    durations.to_csv(report_dir / "durations_by_origin_per_bill.csv")

    measures = [m for m, _ in MEASURES] + ["commons_days", "lords_days"]
    metrics = {
        "bills": {o: int((bills["originating_house"] == o).sum()) for o in ("Commons", "Lords")},
        "fast_tracked": {
            o: int(durations.loc[durations["origin"] == o, "fast_tracked"].sum())
            for o in ("Commons", "Lords")
        },
        "all_bills": {m: compare(durations, m) for m in measures},
        "regular_bills": {m: compare(regular, m) for m in measures},
        "outcome_shares_closed_bills": {
            o: {k: round(float(v), 3) for k, v in shares.loc[o].items()}
            for o in ("Commons", "Lords")
        },
        "royal_assent_rate_fisher_p": float(
            stats.fisher_exact(
                [
                    [
                        int(((bills.originating_house == o) & (bills.outcome == "royal_assent"))
                            .sum()),
                        int(((bills.originating_house == o)
                             & ~bills.outcome.isin(["royal_assent", "in_progress"])).sum()),
                    ]
                    for o in ("Commons", "Lords")
                ]
            ).pvalue
        ),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
