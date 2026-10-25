"""Do bills that move slowly through early stages also move slowly through later ones?

Measures how long Commons-first UK Government Bills spend in five segments of the
Parliamentary journey and tests whether time in earlier segments correlates with time in
later ones (Spearman rank correlation). Also checks whether a slow start predicts failing
to reach Royal Assent.

Run from ``backend/``::

    uv run python -m westminster_weather.analysis.stage_durations

Charts, a CSV of per-bill durations and a summary are written to
``backend/reports/stage_correlation/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from westminster_weather.data.stage_map import STAGE_ID_TO_STATE  # noqa: E402
from westminster_weather.domain.stages import BillStage  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "stage_correlation"

# Bills that clear the whole Commons this fast are emergency, finance or wash-up bills that
# are fast at every stage; they are shown separately because they drive much of the
# all-bills correlation on their own.
FAST_TRACK_MAX_COMMONS_DAYS = 14


@dataclass(frozen=True)
class Segment:
    key: str
    label: str
    start: BillStage
    end: BillStage


# A sample of segments spread across the whole journey, rather than every stage gap
# (many gaps, e.g. Commons report → 3rd reading, are usually the same day).
SEGMENTS: tuple[Segment, ...] = (
    Segment(
        "c_1r_2r", "Commons 1st → 2nd reading",
        BillStage.COMMONS_FIRST_READING, BillStage.COMMONS_SECOND_READING,
    ),
    Segment(
        "c_2r_3r", "Commons 2nd → 3rd reading\n(committee + report)",
        BillStage.COMMONS_SECOND_READING, BillStage.COMMONS_THIRD_READING,
    ),
    Segment(
        "l_1r_comm", "Lords 1st reading → committee",
        BillStage.LORDS_FIRST_AND_SECOND_READING, BillStage.LORDS_COMMITTEE,
    ),
    Segment(
        "l_comm_rep", "Lords committee → report",
        BillStage.LORDS_COMMITTEE, BillStage.LORDS_REPORT_AND_THIRD_READING,
    ),
    Segment(
        "l_rep_ra", "Lords report → Royal Assent\n(incl. ping-pong)",
        BillStage.LORDS_REPORT_AND_THIRD_READING, BillStage.ROYAL_ASSENT,
    ),
)
TOTALS: tuple[Segment, ...] = (
    Segment(
        "commons_total", "Whole Commons\n(1st → 3rd reading)",
        BillStage.COMMONS_FIRST_READING, BillStage.COMMONS_THIRD_READING,
    ),
    Segment(
        "lords_total", "Lords to Royal Assent\n(Lords 1st reading → RA)",
        BillStage.LORDS_FIRST_AND_SECOND_READING, BillStage.ROYAL_ASSENT,
    ),
)


def stage_entry_dates(sittings: pd.DataFrame) -> pd.DataFrame:
    """One row per bill, one column per model state: the date of its first sitting."""
    df = sittings.assign(state=sittings["api_stage_id"].map(STAGE_ID_TO_STATE)).dropna(
        subset=["state"]
    )
    df["state"] = df["state"].map(lambda s: s.value)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df.groupby(["bill_id", "state"])["event_date"].min().unstack()


def segment_durations(
    entry: pd.DataFrame, segments: tuple[Segment, ...], sitting_dates: pd.DatetimeIndex | None
) -> pd.DataFrame:
    """Days between segment start and end, per bill (NaN when a stage was not reached).

    With ``sitting_dates`` the duration counts sitting days in (start, end] instead of
    calendar days, so recesses do not count.
    """
    out = pd.DataFrame(index=entry.index)
    sorted_days = np.sort(sitting_dates.values) if sitting_dates is not None else None
    for seg in segments:
        start = entry.get(seg.start.value)
        end = entry.get(seg.end.value)
        if start is None or end is None:
            out[seg.key] = np.nan
            continue
        if sorted_days is None:
            days = (end - start).dt.days.astype(float)
        else:
            valid = start.notna() & end.notna()
            days = pd.Series(np.nan, index=entry.index)
            lo = np.searchsorted(sorted_days, start[valid].values, side="right")
            hi = np.searchsorted(sorted_days, end[valid].values, side="right")
            days[valid] = (hi - lo).astype(float)
        # Out-of-order dates (e.g. a re-committal recorded later) are not usable durations.
        out[seg.key] = days.where(days >= 0)
    return out


def spearman_matrix(durations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman rho, p-value and n over bills that have both durations."""
    cols = list(durations.columns)
    rho = pd.DataFrame(np.nan, index=cols, columns=cols)
    pval = rho.copy()
    n = pd.DataFrame(0, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            pair = durations[[a, b]].dropna()
            n.loc[a, b] = len(pair)
            if a == b:
                rho.loc[a, b] = 1.0
                pval.loc[a, b] = 0.0
            elif len(pair) >= 10:
                r, p = stats.spearmanr(pair[a], pair[b])
                rho.loc[a, b], pval.loc[a, b] = r, p
    return rho, pval, n


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# Short axis labels for the heatmap, in SEGMENTS order.
SHORT_LABELS: tuple[str, ...] = (
    "C 1st → 2nd rdg",
    "C 2nd → 3rd rdg",
    "L 1st rdg → cttee",
    "L cttee → report",
    "L report → RA",
)


def _draw_heatmap(ax, rho: pd.DataFrame, pval: pd.DataFrame, n: pd.DataFrame, title: str):
    k = len(SEGMENTS)
    # Rows = later segments (2nd..last), columns = earlier segments (1st..second-to-last);
    # cells above the diagonal (earlier "later" than itself) are blank.
    rows, cols = range(1, k), range(k - 1)
    grid = np.array([[rho.iat[i, j] for j in cols] for i in rows])
    data = np.ma.array(grid, mask=np.array([[j >= i for j in cols] for i in rows]))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    for ri, i in enumerate(rows):
        for cj, j in enumerate(cols):
            if j < i:
                r = rho.iat[i, j]
                ax.text(
                    cj, ri, f"{r:+.2f}{_stars(pval.iat[i, j])}\nn={n.iat[i, j]}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(r) > 0.5 else "black",
                )
    ax.set_xticks(range(k - 1), SHORT_LABELS[:-1], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(k - 1), SHORT_LABELS[1:], fontsize=9)
    ax.set_xlabel("Earlier segment")
    ax.set_ylabel("Later segment")
    ax.set_title(title, fontsize=11)
    return im


def plot_heatmap(panels: list[tuple[str, tuple]], path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(panels), figsize=(7 * len(panels), 6.4), layout="constrained"
    )
    for ax, (title, (rho, pval, n)) in zip(axes, panels, strict=True):
        im = _draw_heatmap(ax, rho, pval, n, title)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Spearman ρ  (+ = slow early → slow later)")
    fig.suptitle(
        "Does time in earlier stages predict time in later stages?\n"
        "Spearman correlation of calendar days, Commons-first Government Bills",
        fontsize=12,
    )
    fig.supxlabel(
        "C = Commons, L = Lords, rdg = reading, cttee = committee, RA = Royal Assent.   "
        "* p<0.05  ** p<0.01  *** p<0.001.   n = bills that reached both segments.",
        fontsize=8, color="0.35",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatters(durations: pd.DataFrame, fast: pd.Series, path: Path) -> None:
    pairs = [(SEGMENTS[i], SEGMENTS[i + 1]) for i in range(len(SEGMENTS) - 1)]
    pairs.append(TOTALS)
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.4))
    for ax, (a, b) in zip(axes, pairs, strict=True):
        pair = durations[[a.key, b.key]].dropna()
        is_fast = fast.reindex(pair.index, fill_value=False)
        reg = pair[~is_fast]
        r, p = stats.spearmanr(pair[a.key], pair[b.key])
        r_reg, p_reg = stats.spearmanr(reg[a.key], reg[b.key])
        # +1 so same-day (0 days) fits on a log axis.
        ax.scatter(reg[a.key] + 1, reg[b.key] + 1, s=12, alpha=0.45, color="#3b6ea5",
                   edgecolors="none", label="regular bills")
        ax.scatter(pair.loc[is_fast, a.key] + 1, pair.loc[is_fast, b.key] + 1, s=12,
                   alpha=0.6, color="#b0b0b0", edgecolors="none",
                   label=f"fast-tracked (Commons ≤ {FAST_TRACK_MAX_COMMONS_DAYS} days)")
        # Median of the later segment for each quartile of the earlier one (regular bills).
        q = pd.qcut(reg[a.key].rank(method="first"), 4, labels=False)
        med = reg.groupby(q).median() + 1
        ax.plot(med[a.key], med[b.key], "o-", color="#d9822b", lw=2,
                label="regular: quartile medians")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{a.label} (days + 1)", fontsize=9)
        ax.set_ylabel(f"{b.label} (days + 1)", fontsize=9)
        ax.set_title(
            f"all: ρ = {r:+.2f}{_stars(p)} (n={len(pair)})\n"
            f"regular: ρ = {r_reg:+.2f}{_stars(p_reg)} (n={len(reg)})",
            fontsize=10,
        )
        ax.grid(True, which="major", alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Each step vs the next, and whole Commons vs whole Lords (log scale, calendar days)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def passage_by_early_speed(
    bills: pd.DataFrame, durations: pd.DataFrame
) -> pd.DataFrame:
    """Share of bills reaching Royal Assent, by quartile of Commons 1st → 2nd reading time."""
    df = bills.set_index("bill_id")[["outcome"]].join(durations["c_1r_2r"], how="inner")
    df = df[(df["outcome"] != "in_progress") & df["c_1r_2r"].notna()]
    df["quartile"] = pd.qcut(df["c_1r_2r"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    grouped = df.groupby("quartile", observed=True)
    return pd.DataFrame(
        {
            "days_min": grouped["c_1r_2r"].min(),
            "days_max": grouped["c_1r_2r"].max(),
            "bills": grouped.size(),
            "royal_assent_share": grouped["outcome"].apply(lambda s: (s == "royal_assent").mean()),
        }
    )


def plot_passage(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [
        f"{q}\n{int(r.days_min)}–{int(r.days_max)} days\n(n={int(r.bills)})"
        for q, r in table.iterrows()
    ]
    bars = ax.bar(labels, table["royal_assent_share"] * 100, color="#3b6ea5")
    for bar, v in zip(bars, table["royal_assent_share"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{v:.0%}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Bills that reached Royal Assent (%)")
    ax.set_xlabel("Time from Commons 1st to 2nd reading (quartiles, fastest → slowest)")
    ax.set_title("Does a slow start predict failing to pass?", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(report_dir: Path = REPORT_DIR) -> dict:
    from westminster_weather.db import (
        fetch_bills_df,
        fetch_sitting_days_df,
        fetch_stage_sittings_df,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    bills = fetch_bills_df()
    sittings = fetch_stage_sittings_df()
    sittings = sittings[sittings["bill_type_id"] == 1]  # DATA_SCOPE: Government Bills only
    commons_first = bills.loc[bills["originating_house"] == "Commons", "bill_id"]
    sittings = sittings[sittings["bill_id"].isin(commons_first)]

    entry = stage_entry_dates(sittings)
    all_segments = SEGMENTS + TOTALS
    cal = segment_durations(entry, all_segments, sitting_dates=None)
    sit_days = pd.DatetimeIndex(pd.to_datetime(fetch_sitting_days_df()["date"]).unique())
    sit = segment_durations(entry, all_segments, sitting_dates=sit_days)

    seg_keys = [s.key for s in SEGMENTS]
    fast = cal["commons_total"] <= FAST_TRACK_MAX_COMMONS_DAYS
    regular = cal[~fast]
    all_m = spearman_matrix(cal[seg_keys])
    reg_m = spearman_matrix(regular[seg_keys])
    sit_m = spearman_matrix(sit[seg_keys])
    sit_reg_m = spearman_matrix(sit.loc[~fast, seg_keys])
    passage = passage_by_early_speed(bills, cal)

    plot_heatmap(
        [
            (f"All bills (n={int(cal[seg_keys].notna().any(axis=1).sum())})", all_m),
            (
                f"Regular bills only: Commons > {FAST_TRACK_MAX_COMMONS_DAYS} days "
                f"(n={int(regular[seg_keys].notna().any(axis=1).sum())})",
                reg_m,
            ),
        ],
        report_dir / "1_correlation_heatmap.png",
    )
    plot_scatters(cal, fast, report_dir / "2_stage_pairs_scatter.png")
    plot_passage(passage, report_dir / "3_passage_by_early_speed.png")

    cal.add_suffix("_calendar_days").join(sit.add_suffix("_sitting_days")).assign(
        fast_tracked=fast
    ).to_csv(
        report_dir / "stage_durations_per_bill.csv"
    )

    metrics = {
        "bills_commons_first": int(len(commons_first)),
        "bills_with_any_duration": int(cal[seg_keys].notna().any(axis=1).sum()),
        "fast_tracked_bills": int(fast.sum()),
        "median_days": {k: float(cal[k].median()) for k in cal.columns},
        "spearman_calendar_days_all": _pairs(*all_m),
        "spearman_calendar_days_regular": _pairs(*reg_m),
        "spearman_sitting_days_all": _pairs(*sit_m),
        "spearman_sitting_days_regular": _pairs(*sit_reg_m),
        "commons_total_vs_lords_total": {
            "all": _totals_rho(cal),
            "regular": _totals_rho(regular),
        },
        "royal_assent_share_by_c_1r_2r_quartile": {
            str(q): {
                "days": f"{int(r.days_min)}-{int(r.days_max)}",
                "bills": int(r.bills),
                "share": round(float(r.royal_assent_share), 3),
            }
            for q, r in passage.iterrows()
        },
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _totals_rho(durations: pd.DataFrame) -> dict:
    t = durations[[seg.key for seg in TOTALS]].dropna()
    r, p = stats.spearmanr(t.iloc[:, 0], t.iloc[:, 1])
    return {"rho": round(float(r), 3), "p": float(p), "n": int(len(t))}


def _pairs(rho: pd.DataFrame, pval: pd.DataFrame, n: pd.DataFrame) -> list[dict]:
    keys = list(rho.columns)
    return [
        {
            "earlier": keys[j], "later": keys[i],
            "rho": round(float(rho.iat[i, j]), 3), "p": float(pval.iat[i, j]),
            "n": int(n.iat[i, j]),
        }
        for i in range(len(keys))
        for j in range(i)
    ]


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
