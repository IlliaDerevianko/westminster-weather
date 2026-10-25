"""Days to Royal Assent: can day-one features fix the fast (< 10 weeks) and long (> 50 weeks) bills?

The v2 XGBoost (``remaining_time_v2``) predicts almost every bill at 8-45 weeks from first
reading: routine money and emergency bills (real: days) come out too late, big flagship bills
(real: 1-2 years) too early. Those bills look alike on day one except for what the bill does,
which the long title ("A Bill to ...; to ...; and for connected purposes") says. This adds
long-title features, all known at introduction, and retrains the same tuned XGBoost on the same
bills, forecast points and time split (train sessions 20-30, test 34-39).

Variants:
- ``v2``: v2 features and tuned XGBoost (the reference, retrained here).
- ``+ long title``: plus the long-title features below.
- ``+ long title, bill weights``: each bill counts once (big bills have more forecast points,
  so without weights the long, slow bills dominate the late stages but not day one).
- ``+ long title, recent bills weigh more``: bill weights that also decay with session age.
An 80% range (p10-p90 quantile XGBoost, same features and bill weights) is scored too.

Run from ``backend/``: uv run python -m westminster_weather.analysis.remaining_time_tails
(``--chart``: stakeholder chart from the saved predictions, no retraining;
``--75``: the 75/25 split, see ``SPLITS``)
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from westminster_weather.analysis.remaining_time_dataset import build_dataset, load_source_data
from westminster_weather.analysis.remaining_time_models import TEST_SESSIONS, TRAIN_SESSIONS
from westminster_weather.analysis.remaining_time_v2 import (
    REPORT_DIR as V2_DIR,
)
from westminster_weather.analysis.remaining_time_v2 import (
    Design,
    make_model,
    matrix,
    prepare,
    target_of,
    to_days,
    weights_of,
)
from westminster_weather.takeup.bill_text import bill_long_titles

warnings.filterwarnings("ignore")
REPORT_DIR = V2_DIR.parent / "remaining_time_tails"
# Time splits by session (train, test). "62/38" is the v2 split; "75/25" moves sessions 34-35
# (2019-21) into training: 72% / 28% of bills, the closest session boundary to 75/25 (training
# through session 36 would give 79/21). Hyperparameters stay those tuned on sessions 20-30.
SPLITS = {
    "62/38": (TRAIN_SESSIONS, TEST_SESSIONS),
    "75/25": (frozenset(range(20, 36)), frozenset(range(36, 40))),
}


def report_dir_for(split: str) -> Path:
    return REPORT_DIR if split == "62/38" else REPORT_DIR.parent / "remaining_time_tails_75_25"


FIRST_READING = "1st House: 1st reading"
BANDS = [-1, 70, 210, 350, 10_000]  # days: < 10, 10-30, 30-50, > 50 weeks
BAND_LABELS = ["< 10 wk", "10-30 wk", "30-50 wk", "> 50 wk"]

LONG_TITLE_FLAGS = {
    # routine money bills: Supply and Appropriation, Contingencies Fund, NICs rates, NI budget
    "lt_routine_money": r"supply and appropriation|contingencies fund|consolidated fund|"
    r"national insurance contributions|appropriate .* sums? out of",
    "lt_temporary": r"temporar|for a limited period|expir",
    "lt_emergency": r"emergency|coronavirus|pandemic|in consequence of|urgent|special measures",
    "lt_establish": r"establish|abolish|to confer functions|new (?:body|regulator)",
    "lt_regulate": r"regulat",
    "lt_criminal": r"offence|criminal|sentenc|polic",
    "lt_ni_executive": r"northern ireland (?:executive|assembly|ministers?)|executive formation",
}


def long_title_features(bill_ids: list[int]) -> pd.DataFrame:
    """Size and purpose of each bill from its long title (known at introduction)."""
    titles = bill_long_titles(bill_ids, fetch=False)
    rows = []
    for bill_id in bill_ids:
        t = str(titles.get(bill_id) or "").lower()
        row = {
            "bill_id": bill_id,
            "lt_missing": float(not t),
            "lt_words": float(len(t.split())),
            "lt_purposes": float(len(re.findall(r";\s*(?:and\s+)?to\s|\band to\s", t)) + 1)
            if t
            else 0.0,
            "lt_acts_named": float(len(re.findall(r"\bact \d{4}\b", t))),
            "lt_make_provision": float(len(re.findall(r"make (?:further )?provision", t))),
        }
        for name, pattern in LONG_TITLE_FLAGS.items():
            row[name] = float(bool(re.search(pattern, t)))
        rows.append(row)
    return pd.DataFrame(rows)


def act_length_features() -> pd.DataFrame:
    """Length of each bill's final Act, from ``bill_texts`` (database, read-only). Hindsight:
    an Act exists only after Royal Assent, and bills grow while they are debated, so this is a
    what-if for "if we knew the bill's length", not a feature a live forecast can use."""
    from sqlalchemy import text

    from westminster_weather.db import get_sync_pooled_engine

    e = get_sync_pooled_engine()
    try:
        t = pd.read_sql(
            text(
                "select bill_id, n_characters, text from bill_texts "
                "where text_type = 'act_as_enacted'"
            ),
            e,
        )
    finally:
        e.dispose()
    return pd.DataFrame(
        {
            "bill_id": t["bill_id"].astype(int),
            "size_act_chars_log": np.log1p(t["n_characters"].astype(float)),
            "size_act_schedules": t["text"].str.count(r"\bSCHEDULE \d+\b").astype(float),
        }
    )


def _intro_years(df: pd.DataFrame) -> str:
    intro = pd.to_datetime(df.loc[df["anchor"] == FIRST_READING, "as_of"])
    return f"{intro.min():%Y}–{intro.max():%Y}"


def run(split: str = "62/38") -> dict:
    train_sessions, test_sessions = SPLITS[split]
    report_dir = report_dir_for(split)
    src = load_source_data()
    df = build_dataset(src, set(train_sessions), extended=True)
    lt = long_title_features(sorted(df["bill_id"].astype(int).unique()))
    df = df.merge(lt, on="bill_id", how="left")
    lt_cols = [c for c in lt.columns if c != "bill_id"]
    size = act_length_features()
    df = df.merge(size, on="bill_id", how="left")
    size_cols = [c for c in size.columns if c != "bill_id"]

    v2 = json.loads((V2_DIR / "metrics.json").read_text())
    # v2 features; the top-department one-hots follow this split's training bills
    depts = [c for c in df.columns if c.startswith("dept_")]
    base_cols = [c for c in v2["design"]["features"] if not c.startswith("dept_")] + depts
    params = v2["tuned"]["XGBoost"]["params"]

    train = df[df["session_id"].isin(train_sessions)]
    test = df[df["session_id"].isin(test_sessions)]
    train_p, test_p = prepare(train, test)

    variants = {
        "v2 XGBoost": Design(base_cols),
        "+ long title": Design(base_cols + lt_cols),
        "+ long title, bill weights": Design(base_cols + lt_cols, weights="bill"),
        "+ long title, recent bills weigh more": Design(base_cols + lt_cols, weights="bill_recent"),
        # hindsight: the final Act's length is not known on day one (what-if only)
        "+ long title, Act length (hindsight)": Design(base_cols + lt_cols + size_cols),
    }
    preds = {}
    for name, d in variants.items():
        runs = []
        for seed in range(5):  # seed-average: subsample makes single fits noisy
            m = make_model("XGBoost", params, d, d.features, seed=seed)
            m.fit(
                matrix(train_p, d.features),
                target_of(train_p, d),
                sample_weight=weights_of(train_p, d),
            )
            runs.append(to_days(m.predict(matrix(test_p, d.features)), test_p, d))
        preds[name] = np.mean(runs, axis=0)

    out = test_p[["bill_id", "short_title", "anchor", "as_of", "target_days"]].copy()
    for name, p in preds.items():
        out[name] = p
    # rule of thumb: typical days left at this stage for training bills (v2's B1)
    out[RULE] = np.expm1(test_p["b1_log"].to_numpy(float))
    # 80% range: p10 and p90 quantile models (log days), same features and bill weights
    band_design = variants["+ long title, bill weights"]
    for q in (0.1, 0.9):
        m = XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            objective="reg:quantileerror",
            quantile_alpha=q,
            tree_method="hist",
            random_state=0,
        )
        m.fit(
            matrix(train_p, band_design.features),
            target_of(train_p, band_design),
            sample_weight=weights_of(train_p, band_design),
        )
        pred = m.predict(matrix(test_p, band_design.features))
        out[f"p{int(q * 100)}"] = to_days(pred, test_p, band_design)
    report_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(report_dir / "test_predictions.csv", index=False)

    first = out[out["anchor"] == FIRST_READING].copy()
    first["band"] = pd.cut(first["target_days"], BANDS, labels=BAND_LABELS)
    inside = (first["p10"] <= first["target_days"]) & (first["target_days"] <= first["p90"])
    n_tr, n_te = train["bill_id"].nunique(), test["bill_id"].nunique()
    result: dict = {
        "split": {
            "name": split,
            "train_sessions": sorted(train_sessions),
            "test_sessions": sorted(test_sessions),
            "train_bills": n_tr,
            "test_bills": n_te,
            "train_share": round(n_tr / (n_tr + n_te), 3),
            "train_years": _intro_years(train),
            "test_years": _intro_years(test),
        },
        "all_points_mae_days": {},
        "first_reading": {},
        "first_reading_inside_80_band": {
            "all": round(float(inside.mean()), 2),
            **{
                str(b): round(float(g.mean()), 2)
                for b, g in inside.groupby(first["band"], observed=True)
            },
        },
    }
    for name in variants:
        err = out[name] - out["target_days"]
        result["all_points_mae_days"][name] = round(float(err.abs().mean()), 1)
        e = (first[name] - first["target_days"]) / 7
        by_band = {}
        for band, g in first.assign(e=e).groupby("band", observed=True):
            by_band[str(band)] = {
                "n": len(g),
                "pred_median_wk": round(float(g[name].median() / 7), 1),
                "real_median_wk": round(float(g["target_days"].median() / 7), 1),
                "median_abs_err_wk": round(float(g["e"].abs().median()), 1),
                "median_signed_err_wk": round(float(g["e"].median()), 1),
            }
        result["first_reading"][name] = {
            "median_abs_err_wk": round(float(e.abs().median()), 1),
            "mean_abs_err_wk": round(float(e.abs().mean()), 1),
            "corr_pred_real": round(float(first[name].corr(first["target_days"])), 2),
            "by_band": by_band,
        }
    (report_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main(split: str = "62/38") -> None:
    r = run(split)
    print("Split:", r["split"])
    print("Mean error over all forecast points (days):")
    for name, v in r["all_points_mae_days"].items():
        print(f"  {name:30s} {v:6.1f}")
    print("\nAt first reading (weeks):")
    for name, v in r["first_reading"].items():
        print(
            f"\n  {name}: median |err| {v['median_abs_err_wk']}, mean |err| "
            f"{v['mean_abs_err_wk']}, corr {v['corr_pred_real']}"
        )
        for band, b in v["by_band"].items():
            print(
                f"    {band:9s} n={b['n']:3d}  pred {b['pred_median_wk']:5.1f}  "
                f"real {b['real_median_wk']:5.1f}  |err| {b['median_abs_err_wk']:5.1f}  "
                f"signed {b['median_signed_err_wk']:+6.1f}"
            )
    print("\nReal date inside the 80% range at first reading:", r["first_reading_inside_80_band"])
    print(f"\nwrote {report_dir_for(split)}")


RULE = "Rule of thumb"
NEW = "+ long title, bill weights"
OLD = "v2 XGBoost"


def stakeholder_chart(report_dir: Path = REPORT_DIR) -> Path:
    """Two panels in the style of ``stakeholder_chart``: day-one miss by kind of bill (previous
    vs new model), and the miss on all bills by the stage the forecast is made at."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from westminster_weather.analysis.remaining_time_dataset import STAGE_ANCHORS
    from westminster_weather.analysis.stakeholder_chart import (
        GRID,
        INK,
        INK_SOFT,
        MODEL_COLOUR,
        PLAIN_LABELS,
        SURFACE,
        V1_COLOUR,
    )

    t = pd.read_csv(report_dir / "test_predictions.csv")
    sp = json.loads((report_dir / "metrics.json").read_text())["split"]
    total = t[t["anchor"] == FIRST_READING].set_index("bill_id")["target_days"]
    t["total_days"] = t["bill_id"].map(total)
    t["band"] = pd.cut(t["total_days"], BANDS, labels=BAND_LABELS)
    first = t[t["anchor"] == FIRST_READING]

    def miss(df: pd.DataFrame, col: str) -> float:
        return float(((df[col] - df["target_days"]).abs() / 7).median())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(14, 6.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.35]}
    )

    # A: day-one miss by kind of bill
    kinds = {
        "< 10 wk": "Quick bills\n(under 10 weeks)",
        "10-30 wk": "Typical\n(10-30 weeks)",
        "30-50 wk": "Longer\n(30-50 weeks)",
        "> 50 wk": "Long bills\n(over a year)",
    }
    x = np.arange(len(kinds))
    groups = [first[first["band"] == k] for k in kinds]
    old = [miss(g, OLD) for g in groups]
    new = [miss(g, NEW) for g in groups]
    w = 0.36
    a.bar(x - w / 2 - 0.01, old, w, color=V1_COLOUR, label="Previous model")
    a.bar(x + w / 2 + 0.01, new, w, color=MODEL_COLOUR, label="New model")
    for i, (o, n) in enumerate(zip(old, new, strict=True)):
        a.text(i - w / 2 - 0.01, o + 0.6, f"{o:.0f}", ha="center", color=INK_SOFT, fontsize=10)
        a.text(
            i + w / 2 + 0.01,
            n + 0.6,
            f"{n:.0f}",
            ha="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )
    a.text(
        0,
        max(old[0], new[0]) + max(old) * 0.1,
        f"{old[0]:.0f} → {new[0]:.0f} weeks",
        ha="center",
        color=INK,
        fontsize=10.5,
        fontweight="bold",
    )
    a.set_xticks(x, list(kinds.values()), fontsize=10, color=INK_SOFT)
    a.set_ylabel("Typical miss on day one (weeks)", color=INK_SOFT)
    a.set_title("On day one, by kind of bill", loc="left", color=INK, fontsize=12.5)
    a.legend(frameon=False, loc="upper left", labelcolor=INK_SOFT)
    a.set_ylim(0, max(old + new) * 1.15)

    # B: every bill, previous vs new model, by the stage the forecast is made at
    anchors = [s for s in STAGE_ANCHORS if s in set(t["anchor"])]
    om = [miss(t[t["anchor"] == s], OLD) for s in anchors]
    nm = [miss(t[t["anchor"] == s], NEW) for s in anchors]
    xs = range(len(anchors))
    b.fill_between(xs, nm, om, color=MODEL_COLOUR, alpha=0.07, lw=0)
    b.plot(xs, om, color=V1_COLOUR, lw=2, marker="o", ms=6, markeredgecolor=SURFACE)
    b.plot(
        xs,
        nm,
        color=MODEL_COLOUR,
        lw=3,
        marker="o",
        ms=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    b.annotate(
        f"±{nm[0]:.0f} weeks on day one\n(previous ±{om[0]:.0f})",
        (0, nm[0]),
        xytext=(14, 6),
        textcoords="offset points",
        color=INK,
        fontsize=10.5,
        fontweight="bold",
    )
    b.annotate(
        f"±{nm[-1]:.0f} week{'s' if round(nm[-1]) != 1 else ''}\nat the end",
        (len(anchors) - 1, nm[-1]),
        xytext=(-6, 22),
        textcoords="offset points",
        color=INK,
        fontsize=10,
        ha="right",
    )
    b.set_xticks(
        list(xs),
        [{FIRST_READING: "1st\nreading"}.get(s, PLAIN_LABELS[s]) for s in anchors],
        fontsize=9,
        color=INK_SOFT,
    )
    b.set_ylabel("Typical miss, all bills (weeks)", color=INK_SOFT)
    b.set_title(
        "Every forecast sharpens as the bill moves through Parliament",
        loc="left",
        color=INK,
        fontsize=12.5,
    )
    b.set_ylim(0, max(om + nm) * 1.3)
    for start, end, label in ((0, 4, "First House"), (5, 9, "Second House")):
        b.annotate(
            "",
            xy=(start - 0.3, -0.17),
            xytext=(end + 0.3, -0.17),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "-", "color": GRID, "lw": 1.5},
        )
        b.text(
            (start + end) / 2,
            -0.185,
            label,
            transform=b.get_xaxis_transform(),
            ha="center",
            va="top",
            color=INK_SOFT,
            fontsize=10,
        )

    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.yaxis.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, colors=INK_SOFT)

    n = len(first)
    fig.text(
        0.04,
        0.95,
        "Sharper Royal Assent forecasts, most of all for quick bills",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.905,
        "Typical miss in predicted time to Royal Assent (median, weeks). "
        f"Tested on {n} Government Bills introduced in {sp['test_years']}"
        f" ({1 - sp['train_share']:.0%} of bills; {sp['train_share']:.0%} used for training).",
        fontsize=11,
        color=INK_SOFT,
    )
    fig.text(
        0.04,
        0.02,
        "Previous model = Illia's tuned XGBoost. New model = the same, plus what the bill's "
        "long title says it does (e.g. a routine money\nor emergency bill, how many "
        f"purposes), with each bill counted once. Both trained on {sp['train_years']} bills only.",
        fontsize=9,
        color=INK_SOFT,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.25, wspace=0.22)
    path = report_dir / "0_summary_for_stakeholders.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def stage_chart(report_dir: Path = REPORT_DIR) -> Path:
    """The general stakeholder chart in the style of ``stakeholder_chart.plot_v2``: average
    error in days by the stage the forecast is made at, new vs previous model vs rule of
    thumb, all bills."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from westminster_weather.analysis.remaining_time_dataset import STAGE_ANCHORS
    from westminster_weather.analysis.stakeholder_chart import (
        BASELINE_COLOUR,
        GRID,
        INK,
        INK_SOFT,
        MODEL_COLOUR,
        PLAIN_LABELS,
        SURFACE,
        V1_COLOUR,
    )

    t = pd.read_csv(report_dir / "test_predictions.csv")
    sp = json.loads((report_dir / "metrics.json").read_text())["split"]
    t = t[t["anchor"].isin(STAGE_ANCHORS)]
    anchors = [a for a in STAGE_ANCHORS if a in set(t["anchor"])]

    def mae(col: str) -> pd.Series:
        e = (t[col] - t["target_days"]).abs()
        return e.groupby(t["anchor"]).mean().reindex(anchors)

    new, old, base = mae(NEW), mae(OLD), mae(RULE)
    overall = {c: float((t[c] - t["target_days"]).abs().mean()) for c in (NEW, OLD, RULE)}
    x = range(len(anchors))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(x, new, base, color=MODEL_COLOUR, alpha=0.07, lw=0)
    for series, colour, lw, ms in (
        (base, BASELINE_COLOUR, 2, 6),
        (old, V1_COLOUR, 2, 6),
        (new, MODEL_COLOUR, 3, 8),
    ):
        ax.plot(
            x,
            series,
            color=colour,
            lw=lw,
            marker="o",
            ms=ms,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            solid_capstyle="round",
        )
    handles = [
        plt.Line2D([], [], color=c, lw=lw, marker="o", ms=ms, markeredgecolor=SURFACE)
        for c, lw, ms in ((MODEL_COLOUR, 3, 8), (V1_COLOUR, 2, 6), (BASELINE_COLOUR, 2, 6))
    ]
    labels = [
        f"New model  ·  average ±{overall[NEW]:.0f} days",
        f"Previous model  ·  average ±{overall[OLD]:.0f} days",
        f"Rule of thumb  ·  average ±{overall[RULE]:.0f} days",
    ]
    leg = ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        fontsize=11,
        handlelength=2.6,
        labelcolor=INK_SOFT,
    )
    leg.get_texts()[0].set_color(INK)
    leg.get_texts()[0].set_fontweight("bold")
    ax.annotate(
        f"±{new.iloc[0]:.0f} days at introduction\n(previous model ±{old.iloc[0]:.0f})",
        (0, new.iloc[0]),
        xytext=(0.25, new.iloc[0] * 0.5),
        textcoords="data",
        color=INK,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        arrowprops={"arrowstyle": "-", "color": INK_SOFT, "lw": 0.8},
    )
    ax.set_xticks(list(x), [PLAIN_LABELS[a] for a in anchors], fontsize=10, color=INK_SOFT)
    ax.set_xlim(-0.4, len(anchors) - 0.5)
    ax.set_ylim(0, max(base.max(), old.max(), new.max()) * 1.12)
    ax.set_ylabel("Average forecast error (days)", color=INK_SOFT)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK_SOFT)
    for start, end, label in ((0, 4, "First House"), (5, 9, "Second House")):
        ax.annotate(
            "",
            xy=(start - 0.3, -0.155),
            xytext=(end + 0.3, -0.155),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "-", "color": GRID, "lw": 1.5},
        )
        ax.text(
            (start + end) / 2,
            -0.17,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=INK_SOFT,
            fontsize=10,
        )

    fig.text(
        0.06,
        0.95,
        "Royal Assent forecasts get sharper as a bill moves through Parliament",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.905,
        "Average error in predicted days to Royal Assent, by the stage at which the "
        f"forecast is made. Tested on {sp['test_bills']} bills introduced in "
        f"{sp['test_years']}.",
        fontsize=11,
        color=INK_SOFT,
    )
    fig.text(
        0.06,
        0.02,
        "Rule of thumb = typical time left for past bills at the same stage. "
        "Previous model = Illia's tuned XGBoost.\nNew model = the same, plus what the "
        "bill's long title says it does, with each bill counted once. "
        f"All trained on {sp['train_years']} bills only ({sp['train_share']:.0%} of bills).",
        fontsize=9,
        color=INK_SOFT,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.25)
    path = report_dir / "1_by_stage_for_stakeholders.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def pred_vs_real_chart(report_dir: Path = REPORT_DIR) -> Path:
    """Test bills only: predicted (new model, made at introduction) and real weeks to Royal
    Assent, one line each, bills sorted by real time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from westminster_weather.analysis.stakeholder_chart import (
        GRID,
        INK,
        INK_SOFT,
        MODEL_COLOUR,
        SURFACE,
        V1_COLOUR,
    )

    t = pd.read_csv(report_dir / "test_predictions.csv")
    sp = json.loads((report_dir / "metrics.json").read_text())["split"]
    f = t[t["anchor"] == FIRST_READING].sort_values("target_days").reset_index(drop=True)
    real, pred = f["target_days"] / 7, f[NEW] / 7
    miss = (pred - real).abs().median()
    x = range(len(f))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(x, pred, real, color=MODEL_COLOUR, alpha=0.07, lw=0)
    ax.plot(x, real, color=V1_COLOUR, lw=2.5, solid_capstyle="round")
    ax.plot(x, pred, color=MODEL_COLOUR, lw=2, solid_capstyle="round")
    last = len(f) - 1
    ax.annotate(
        "Real",
        (last, real.iloc[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        color=V1_COLOUR,
        fontsize=12,
        fontweight="bold",
        va="center",
    )
    ax.annotate(
        "Predicted",
        (last, pred.iloc[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        color=MODEL_COLOUR,
        fontsize=12,
        fontweight="bold",
        va="center",
    )
    ax.set_xlim(-1, last + 12)
    ax.set_ylim(0, max(real.max(), pred.max()) * 1.08)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(f)} test bills, sorted by how long they really took", color=INK_SOFT)
    ax.set_ylabel("Weeks from introduction to Royal Assent", color=INK_SOFT)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK_SOFT)

    fig.text(
        0.06,
        0.95,
        "Predicted vs real time to Royal Assent, forecast on day one",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.905,
        f"Test bills only ({sp['test_years']}), never seen in training. "
        f"Typical miss {miss:.0f} weeks.",
        fontsize=11,
        color=INK_SOFT,
    )
    fig.text(
        0.06,
        0.02,
        "Predicted = new model (Illia's tuned XGBoost plus what the bill's long title "
        "says it does), forecast on the day the bill is introduced.\n"
        f"Trained on {sp['train_years']} bills only ({sp['train_share']:.0%} of bills).",
        fontsize=9,
        color=INK_SOFT,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.14)
    path = report_dir / "2_predicted_vs_real_test.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def miss_vs_left_chart(report_dir: Path = REPORT_DIR) -> Path:
    """Test bills, by the stage the forecast is made at: typical days still to go and the new
    model's typical miss (medians, days), so the shrinking miss reads against the time left."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from westminster_weather.analysis.remaining_time_dataset import STAGE_ANCHORS
    from westminster_weather.analysis.stakeholder_chart import (
        GRID,
        INK,
        INK_SOFT,
        MODEL_COLOUR,
        PLAIN_LABELS,
        SURFACE,
    )

    t = pd.read_csv(report_dir / "test_predictions.csv")
    sp = json.loads((report_dir / "metrics.json").read_text())["split"]
    t = t[t["anchor"].isin(STAGE_ANCHORS)].assign(e=lambda d: (d[NEW] - d["target_days"]).abs())
    anchors = [a for a in STAGE_ANCHORS if a in set(t["anchor"])]
    g = t.groupby("anchor").agg(left=("target_days", "median"), miss=("e", "median"))
    g = g.reindex(anchors)
    x = np.arange(len(anchors))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(x, g["left"], 0.62, color="#e6e5e1", label="Days still to go (typical bill)")
    ax.plot(
        x,
        g["miss"],
        color=MODEL_COLOUR,
        lw=3,
        marker="o",
        ms=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label="How far off the forecast is (typical miss)",
    )
    for i, (left, miss) in enumerate(zip(g["left"], g["miss"], strict=True)):
        ax.text(
            i,
            miss + 5,
            f"±{miss:.0f}",
            ha="center",
            va="bottom",
            color=MODEL_COLOUR,
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            i, max(left, miss + 18) + 3, f"{left:.0f}", ha="center", color=INK_SOFT, fontsize=9.5
        )
    ax.set_xticks(list(x), [PLAIN_LABELS[a] for a in anchors], fontsize=10, color=INK_SOFT)
    ax.set_xlim(-0.6, len(anchors) - 0.4)
    ax.set_ylim(0, g["left"].max() * 1.15)
    ax.set_ylabel("Days", color=INK_SOFT)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=11, labelcolor=INK_SOFT)
    leg.get_texts()[0].set_color(INK)
    leg.get_texts()[0].set_fontweight("bold")
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK_SOFT)
    for start, end, label in ((0, 4, "First House"), (5, 9, "Second House")):
        ax.annotate(
            "",
            xy=(start - 0.3, -0.155),
            xytext=(end + 0.3, -0.155),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "-", "color": GRID, "lw": 1.5},
        )
        ax.text(
            (start + end) / 2,
            -0.17,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=INK_SOFT,
            fontsize=10,
        )

    first, final = g["miss"].iloc[0], g["miss"].iloc[-2]
    fig.text(
        0.06,
        0.95,
        f"The forecast closes in on Royal Assent: ±{first:.0f} days at introduction, "
        f"±{final:.0f} at the end",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.905,
        "Typical days still to go, and how far off the forecast made at that stage is. "
        f"Test bills only: {sp['test_bills']} introduced in {sp['test_years']}.",
        fontsize=11,
        color=INK_SOFT,
    )
    fig.text(
        0.06,
        0.02,
        "Forecast = new model (Illia's tuned XGBoost plus what the bill's long title says "
        "it does), using only what was known at each stage.\n"
        f"Trained on {sp['train_years']} bills only ({sp['train_share']:.0%} of bills).",
        fontsize=9,
        color=INK_SOFT,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.25)
    path = report_dir / "3_miss_vs_time_left.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    split = "75/25" if "--75" in sys.argv else "62/38"
    if "--chart" not in sys.argv:
        main(split)
    print(stakeholder_chart(report_dir_for(split)))
    print(stage_chart(report_dir_for(split)))
    print(pred_vs_real_chart(report_dir_for(split)))
    print(miss_vs_left_chart(report_dir_for(split)))
