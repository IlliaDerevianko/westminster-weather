"""Stage 3 with XGBoost: days from Royal Assent until the law applies.

Built on the stage 2 remaining-time experiments (since removed; see git history) with a new
target, new checkpoints and new features.

- **Target:** days from the checkpoint until the law applies: the first commencement order if
  the Act needed one, else the day its sections started.
- **Checkpoints:** Royal Assent, then 30, 90, 180 and 365 days after it, for Acts whose law
  has not applied yet.
- **Features** (all known at the checkpoint; the Act's text is final at Royal Assent):
  how the Act says it comes into force, its regulation-making powers, length and schedules,
  department, time in Parliament, parliament and calendar timing; at later checkpoints,
  regulations laid and drafted so far.
- **Models:** one XGBoost on log days (Illia's tuned settings), and a two-step version: a
  classifier for "applies on Royal Assent day" at the Royal Assent checkpoint, then the
  regressor for the rest. Each Act counts once in training.
- **Compared with:** the rule of thumb (typical days left at the checkpoint for training Acts)
  and the current stage 3 model, ``journey.commencement.forecast_act``.
- **Split:** by the bill's session, 75/25: train 20-35,
  test 36-39. Acts with no commencement order whose Royal Assent was in the last year of the
  data are left out of the test (their first order may still be coming).

Reads the database only. Run from ``backend/``:
uv run python -m westminster_weather.analysis.commencement_xgb
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
REPORTS = Path(__file__).resolve().parents[3] / "reports"
REPORT_DIR = REPORTS / "commencement_xgb"
TRAIN_SESSIONS = frozenset(range(20, 36))
TEST_SESSIONS = frozenset(range(36, 40))
CHECKPOINTS = {"Royal Assent": 0, "+30 days": 30, "+90 days": 90, "+180 days": 180, "+1 year": 365}
TOP_DEPARTMENTS = 12
RECENT_DAYS = 365  # no order yet and Royal Assent this recent: outcome not known
BANDS = [-1, 0, 90, 365, 100_000]
BAND_LABELS = ["Same day", "Within 3 months", "3-12 months", "Over a year"]
SEEDS = 5

CLAUSES = {
    "c_on_passing": r"comes? into force on the day on which this act is passed",
    "c_months_after": r"at the end of the period of [\w ]{1,20}? ?(?:months|days) beginning "
    r"with the day on which this act is passed",
    "c_appoint": r"on such day as the [^.]{0,80}may by (?:regulations|order) appoint",
    "c_specific_date": r"comes? into force on \d{1,2} \w+ \d{4}",
}
MONEY_TITLE = (
    r"^(?:finance|supply and appropriation|appropriation|consolidated fund|"
    r"national insurance|contingencies fund)"
)


# --------------------------------------------------------------------------------- data


def load_tables() -> dict[str, pd.DataFrame]:
    from sqlalchemy import text

    from westminster_weather.db import get_sync_pooled_engine

    queries = {
        "acts": "select * from act_commencements",
        "texts": "select bill_id, text from act_texts",
        "bills": "select bill_id, short_title, introduced_session_id, department, "
        "originating_house from bills",
        "events": "select bill_id, stage, event_date from post_royal_assent_events",
        "first_readings": "select bill_id, min(event_date) as first_reading from stage_sittings "
        "where api_stage_id in (1, 6) and bill_type_id = 1 group by bill_id",
        "parliaments": "select * from parliaments",
    }
    e = get_sync_pooled_engine()
    try:
        return {k: pd.read_sql(text(q), e) for k, q in queries.items()}
    finally:
        e.dispose()


def act_features(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per Act: target dates and everything known at Royal Assent."""
    a = t["acts"].copy()
    for c in ("royal_assent_date", "first_commencement_order_date", "some_sections_start_date"):
        a[c] = pd.to_datetime(a[c])
    a["applied"] = np.where(
        a["commencement_orders_count"] > 0,
        a["first_commencement_order_date"],
        a["some_sections_start_date"].fillna(a["royal_assent_date"]),
    )
    a["applied"] = pd.to_datetime(a["applied"])
    a = a.merge(t["bills"], on="bill_id", how="left").merge(t["texts"], on="bill_id", how="left")
    a = a.merge(t["first_readings"], on="bill_id", how="left")

    low = a["text"].fillna("").str.lower()
    for name, pattern in CLAUSES.items():
        a[name] = low.str.contains(pattern, regex=True).astype(float)
    a["n_appoint"] = low.str.count(r"may by (?:regulations|order) appoint").astype(float)
    a["n_regulation_powers"] = low.str.count(r"may by regulations").astype(float)
    a["act_chars_log"] = np.log1p(low.str.len().astype(float))
    a["n_schedules"] = low.str.count(r"\bschedule \d+\b").astype(float)
    a["money_title"] = (
        a["short_title"].fillna("").str.lower().str.contains(MONEY_TITLE, regex=True).astype(float)
    )
    a["origin_lords"] = (a["originating_house"].fillna("").str.lower() == "lords").astype(float)
    ra = a["royal_assent_date"]
    a["days_in_parliament"] = (ra - pd.to_datetime(a["first_reading"])).dt.days.astype(float)
    a["ra_month"] = ra.dt.month.astype(float)
    elections = pd.to_datetime(t["parliaments"]["election_date"]).dropna().sort_values()
    last = [elections[elections <= d].max() for d in ra]
    a["parliament_age_years"] = [
        (d - e).days / 365.25 if pd.notna(e) else np.nan for d, e in zip(ra, last, strict=True)
    ]
    return a


def build_rows(a: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per (Act, checkpoint) while the law has not applied yet."""
    ev = events.assign(event_date=pd.to_datetime(events["event_date"]))
    laid = ev[ev["stage"] == "regulations_laid"]
    drafted = ev[ev["stage"] == "regulations_drafted"]
    rows = []
    for _, act in a.iterrows():
        ra, applied = act["royal_assent_date"], act["applied"]
        if pd.isna(ra) or pd.isna(applied):
            continue
        for label, d in CHECKPOINTS.items():
            as_of = ra + pd.Timedelta(days=d)
            if d > 0 and applied <= as_of:
                break  # already applied: nothing left to predict
            b = act["bill_id"]
            rows.append(
                {
                    "bill_id": b,
                    "checkpoint": label,
                    "as_of": as_of,
                    "days_since_ra": float(d),
                    "regs_laid_so_far": float(
                        ((laid["bill_id"] == b) & (laid["event_date"] <= as_of)).sum()
                    ),
                    "regs_drafted_so_far": float(
                        ((drafted["bill_id"] == b) & (drafted["event_date"] <= as_of)).sum()
                    ),
                    "target_days": float((applied - as_of).days),
                }
            )
    return pd.DataFrame(rows).merge(a, on="bill_id")


STATIC = [
    *CLAUSES,
    "n_appoint",
    "n_regulation_powers",
    "act_chars_log",
    "n_schedules",
    "money_title",
    "origin_lords",
    "days_in_parliament",
    "ra_month",
    "parliament_age_years",
]
DYNAMIC = ["days_since_ra", "regs_laid_so_far", "regs_drafted_so_far"]


# -------------------------------------------------------------------------------- model


# Illia's tuned XGBoost settings (Optuna, 150 trials, CV MAE 42.4 days) from the remaining-time
# v2 experiment, copied here when that experiment's reports were removed
XGB_PARAMS = {
    "n_estimators": 152,
    "learning_rate": 0.04483432342553568,
    "max_depth": 6,
    "min_child_weight": 38.596023091305504,
    "subsample": 0.5614272056991553,
    "colsample_bytree": 0.9805096814081286,
    "reg_lambda": 0.0012337574175209259,
    "reg_alpha": 0.0034953079096790934,
    "gamma": 0.011393413712761357,
}


def _params() -> dict:
    return dict(XGB_PARAMS)


def _weights(df: pd.DataFrame) -> np.ndarray:
    w = 1.0 / df.groupby("bill_id")["bill_id"].transform("size").to_numpy(float)
    return w * len(w) / w.sum()


def _x(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).fillna(-1.0).to_numpy()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    p = _params()
    single, positive, p_same_day = [], [], []
    pos = train[train["target_days"] > 0]
    at_ra = train[train["checkpoint"] == "Royal Assent"]
    for seed in range(SEEDS):
        reg = XGBRegressor(
            **p, objective="reg:absoluteerror", tree_method="hist", random_state=seed
        )
        reg.fit(_x(train, cols), np.log1p(train["target_days"]), sample_weight=_weights(train))
        single.append(np.expm1(reg.predict(_x(test, cols))).clip(0))
        reg_pos = XGBRegressor(
            **p, objective="reg:absoluteerror", tree_method="hist", random_state=seed
        )
        reg_pos.fit(_x(pos, cols), np.log1p(pos["target_days"]), sample_weight=_weights(pos))
        positive.append(np.expm1(reg_pos.predict(_x(test, cols))).clip(0))
        # classifier leaves are measured in p(1-p) <= 0.25 per Act, so the regression setting
        # (min_child_weight ~39) would stop a 374-Act classifier from splitting at all
        clf = XGBClassifier(
            **{**p, "min_child_weight": 1.0},
            objective="binary:logistic",
            tree_method="hist",
            random_state=seed,
        )
        clf.fit(_x(at_ra, cols), (at_ra["target_days"] <= 0).astype(int))
        p_same_day.append(clf.predict_proba(_x(test, cols))[:, 1])
    p0 = np.mean(p_same_day, axis=0)
    p0 = np.where(test["checkpoint"].to_numpy() == "Royal Assent", p0, 0.0)
    two_step = np.where(p0 >= 0.5, 0.0, np.mean(positive, axis=0))
    return {
        "XGBoost (one step)": np.mean(single, axis=0),
        "XGBoost (two step)": two_step,
        "p_same_day": p0,
    }


def current_model(test: pd.DataFrame) -> np.ndarray:
    """Days left from the current stage 3 model (``journey.commencement``) at each row."""
    from westminster_weather.journey import commencement as cm

    data = cm.load_data()
    out = []
    for _, r in test.iterrows():
        try:
            f = cm.forecast_act(int(r["bill_id"]), r["as_of"].date(), data=data)
        except (KeyError, ValueError):
            out.append(np.nan)
            continue
        d50 = (f["first_order"].get("calibrated_dates") or {}).get("t_50")
        if r["checkpoint"] == "Royal Assent" and f["needs_orders"]["p"] < 0.5:
            out.append(0.0)
        elif d50:
            out.append(float(max((pd.Timestamp(d50) - r["as_of"]).days, 0)))
        else:
            out.append(np.nan)
    return np.array(out)


# ------------------------------------------------------------------------------ run


def run() -> dict:
    t = load_tables()
    a = act_features(t)
    rows = build_rows(a, t["events"])
    data_end = pd.to_datetime(t["events"]["event_date"]).max()
    train = rows[rows["introduced_session_id"].isin(TRAIN_SESSIONS)].copy()
    test = rows[rows["introduced_session_id"].isin(TEST_SESSIONS)].copy()
    unknown = (test["commencement_orders_count"] == 0) & (
        test["royal_assent_date"] > data_end - pd.Timedelta(days=RECENT_DAYS)
    )
    n_unknown = int(test.loc[unknown, "bill_id"].nunique())
    test = test[~unknown].reset_index(drop=True)

    depts = train["department"].value_counts().head(TOP_DEPARTMENTS).index
    for i, d in enumerate(depts):
        for df in (train, test):
            df[f"dept_{i}"] = (df["department"] == d).astype(float)
    cols = STATIC + DYNAMIC + [f"dept_{i}" for i in range(len(depts))]

    preds = fit_predict(train, test, cols)
    out = test[
        ["bill_id", "act_title", "checkpoint", "as_of", "royal_assent_date", "target_days"]
    ].copy()
    for k, v in preds.items():
        out[k] = v
    rule = train.groupby("checkpoint")["target_days"].median()
    out["Rule of thumb"] = out["checkpoint"].map(rule)
    out["Current stage 3 model"] = current_model(test)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    models = ["XGBoost (two step)", "XGBoost (one step)", "Current stage 3 model", "Rule of thumb"]
    n_tr, n_te = train["bill_id"].nunique(), out["bill_id"].nunique()
    metrics: dict = {
        "split": {
            "train_sessions": sorted(TRAIN_SESSIONS),
            "test_sessions": sorted(TEST_SESSIONS),
            "train_acts": n_tr,
            "test_acts": n_te,
            "train_share": round(n_tr / (n_tr + n_te), 3),
            "test_left_out_too_recent": n_unknown,
            "test_years": f"{out['royal_assent_date'].min():%Y}–"
            f"{out['royal_assent_date'].max():%Y}",
            "train_years": f"{train['royal_assent_date'].min():%Y}–"
            f"{train['royal_assent_date'].max():%Y}",
        },
        "by_checkpoint": {},
        "at_royal_assent_by_kind": {},
    }
    for m in models:
        e = (out[m] - out["target_days"]).abs()
        metrics["by_checkpoint"][m] = {
            cp: {
                "n": int(g.notna().sum()),
                "median_miss_days": round(float(g.median()), 1),
                "mean_miss_days": round(float(g.mean()), 1),
            }
            for cp, g in e.groupby(out["checkpoint"], sort=False)
        }
        ra = out[out["checkpoint"] == "Royal Assent"]
        er = (ra[m] - ra["target_days"]).abs()
        band = pd.cut(ra["target_days"], BANDS, labels=BAND_LABELS)
        metrics["at_royal_assent_by_kind"][m] = {
            "all": {
                "n": int(er.notna().sum()),
                "median_miss_days": round(float(er.median()), 1),
                "mean_miss_days": round(float(er.mean()), 1),
                "within_30_days": round(float((er <= 30).mean()), 2),
            },
            **{
                str(b): {"n": int(g.notna().sum()), "median_miss_days": round(float(g.median()), 1)}
                for b, g in er.groupby(band, observed=True)
            },
        }
    ra = out[out["checkpoint"] == "Royal Assent"]
    same = ra["target_days"] <= 0
    metrics["same_day_classifier_accuracy"] = round(
        float(((ra["p_same_day"] >= 0.5) == same).mean()), 2
    )
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


STAGE3_MODEL = "XGBoost (two step)"
STAGE3_CHART_CHECKPOINTS = ["Royal Assent"]  # every Act; later checkpoints only cover slow ones


def stage3_chart(report_dir: Path = REPORT_DIR) -> Path:
    """Stakeholder chart for stage 3 on the test Acts, forecast on Royal Assent day: typical
    miss by kind of Act, and the share forecast within 20 days, for the new XGBoost, the current
    stage 3 model and the rule of thumb."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from westminster_weather.analysis.chart_style import (
        BASELINE_COLOUR,
        GRID,
        INK,
        INK_SOFT,
        MODEL_COLOUR,
        SURFACE,
        V1_COLOUR,
    )

    t = pd.read_csv(report_dir / "test_predictions.csv")
    ra = t[t["checkpoint"] == "Royal Assent"].copy()
    ra["kind"] = pd.cut(ra["target_days"], BANDS, labels=BAND_LABELS)
    models = [
        ("Rule of thumb", "Rule of thumb", BASELINE_COLOUR),
        ("Current stage 3 model", "Current model", V1_COLOUR),
        (STAGE3_MODEL, "New model (XGBoost)", MODEL_COLOUR),
    ]
    kinds = {
        "Same day": "Applies on\nRoyal Assent day",
        "Within 3 months": "Within\n3 months",
        "3-12 months": "3-12\nmonths",
        "Over a year": "Over\na year",
    }
    counts = ra["kind"].value_counts()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(14, 6.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # A: typical miss by kind of Act (log scale: 0 to 500+ days)
    x = np.arange(len(kinds))
    w = 0.26
    for j, (col, label, colour) in enumerate(models):
        vals = [
            float(
                (ra.loc[ra["kind"] == k, col] - ra.loc[ra["kind"] == k, "target_days"])
                .abs()
                .median()
            )
            for k in kinds
        ]
        xs = x + (j - 1) * (w + 0.02)
        a.bar(xs, [max(v, 0.6) for v in vals], w, color=colour, label=label)
        for xi, v in zip(xs, vals, strict=True):
            a.text(
                xi,
                max(v, 0.6) * 1.12,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=INK if col == STAGE3_MODEL else INK_SOFT,
                fontweight="bold" if col == STAGE3_MODEL else "normal",
            )
    a.set_yscale("log")
    a.set_ylim(0.5, 3000)
    a.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    a.set_xticks(
        x,
        [f"{v}\n({counts.get(k, 0)} Acts)" for k, v in kinds.items()],
        fontsize=10,
        color=INK_SOFT,
    )
    a.set_ylabel("Typical miss (days, log scale)", color=INK_SOFT)
    a.set_title(
        "On Royal Assent day, by how long the law really took", loc="left", color=INK, fontsize=12.5
    )
    a.legend(frameon=False, loc="upper left", labelcolor=INK_SOFT, ncol=3, fontsize=10)

    # B: share of Acts forecast within 20 days
    shares = [float(((ra[col] - ra["target_days"]).abs() <= 20).mean()) for col, _, _ in models]
    ys = np.arange(len(models))
    b.barh(ys, shares, 0.55, color=[c for _, _, c in models])
    for y, (sh, (col, _label, _)) in enumerate(zip(shares, models, strict=True)):
        b.text(
            sh + 0.015,
            y,
            f"{sh:.0%}",
            va="center",
            fontsize=11,
            color=INK if col == STAGE3_MODEL else INK_SOFT,
            fontweight="bold" if col == STAGE3_MODEL else "normal",
        )
    b.set_yticks(ys, [label for _, label, _ in models], fontsize=10.5, color=INK_SOFT)
    b.set_xlim(0, 1)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    b.set_title(
        f"Forecast within 20 days (all {len(ra)} Acts)", loc="left", color=INK, fontsize=12.5
    )

    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, colors=INK_SOFT)
    a.yaxis.grid(True, color=GRID, lw=1)
    b.xaxis.grid(True, color=GRID, lw=1)
    a.set_axisbelow(True)
    b.set_axisbelow(True)

    new_e = (ra[STAGE3_MODEL] - ra["target_days"]).abs()
    fig.text(
        0.04,
        0.95,
        f"When does a new Act's law apply? Typical miss ±{new_e.median():.0f} days on "
        "Royal Assent day",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.905,
        "Forecast made on the day of Royal Assent, using only what was known that day. "
        f"Test data only: {len(ra)} Acts from bills introduced in 2021–2026.",
        fontsize=11,
        color=INK_SOFT,
    )
    fig.text(
        0.04,
        0.02,
        "Law applies = the first commencement order, or the day the Act's sections start "
        "if no order is needed. New model = XGBoost reading the Act's commencement wording "
        "(applies on the day? then how long).\nCurrent model = the existing stage 3 "
        "model. Rule of thumb = typical wait of earlier Acts. All trained on 2006–2021 "
        "bills only.",
        fontsize=9,
        color=INK_SOFT,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.8, bottom=0.2, wspace=0.3)
    path = report_dir / "1_stage3_for_stakeholders.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    m = run()
    print("Split:", m["split"])
    print("\nAt Royal Assent (days):")
    for model, v in m["at_royal_assent_by_kind"].items():
        print(
            f"  {model:24s}",
            {k: (d["median_miss_days"], d["n"]) for k, d in v.items()},
            f"mean {v['all']['mean_miss_days']}, within 30d {v['all']['within_30_days']}",
        )
    print("\nBy checkpoint, median miss (days):")
    for model, v in m["by_checkpoint"].items():
        print(f"  {model:24s}", {cp: d["median_miss_days"] for cp, d in v.items()})
    print("\nBy checkpoint, mean miss (days):")
    for model, v in m["by_checkpoint"].items():
        print(f"  {model:24s}", {cp: d["mean_miss_days"] for cp, d in v.items()})
    print("\nSame-day classifier accuracy at Royal Assent:", m["same_day_classifier_accuracy"])
    print(f"\nwrote {REPORT_DIR}")


if __name__ == "__main__":
    import sys

    if "--chart" not in sys.argv:
        main()
    print(stage3_chart())
