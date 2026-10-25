"""Tree models for the days left until Royal Assent, evaluated by forecast point.

Trains decision-tree-style models on the point-in-time dataset from
``remaining_time_dataset`` (finished, Royal Assent bills only), tunes their hyperparameters
with time-ordered cross-validation inside the training sessions, and evaluates them on later
sessions, showing how the error shrinks as the forecast starts later in the pipeline.

Run from ``backend/`` (takes a few minutes)::

    uv run python -m westminster_weather.analysis.remaining_time_models

Charts, tables and a summary are written to ``backend/reports/remaining_time_models/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from westminster_weather.analysis.remaining_time_dataset import (  # noqa: E402
    EVENT_FLAGS,
    GAP_FEATURES,
    STAGE_ANCHORS,
    TIME_ANCHOR_DAYS,
    build_dataset,
    feature_columns,
    load_source_data,
)

REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "remaining_time_models"
SEED = 42

# MODEL_SPEC §7: train on sessions 20–30 (2006-07 to 2017-19), test on 34–39 (2019 onwards).
# Session 40 is still running: only its fastest bills have finished, so it is left out.
TRAIN_SESSIONS = frozenset(range(20, 31))
TEST_SESSIONS = frozenset(range(34, 40))

# Expanding-window folds inside the training sessions: always validate on later sessions.
CV_FOLDS: tuple[tuple[frozenset[int], frozenset[int]], ...] = (
    (frozenset(range(20, 26)), frozenset({26, 27})),
    (frozenset(range(20, 28)), frozenset({28, 29})),
    (frozenset(range(20, 30)), frozenset({30})),
)
MISSING = -1.0  # durations are never negative, so -1 safely means "not reached yet"


# ---------------------------------------------------------------------------
# Features added after the split (they use training-set statistics)
# ---------------------------------------------------------------------------


def add_pace(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """``pace_vs_typical``: elapsed days ÷ the training median at the same stage and origin.

    Above 1 means the bill is slower than a typical training bill at the same point.
    """
    stage_rows = train[train["anchor_type"] == "stage"]
    med = stage_rows.groupby(["stage_index", "origin_lords"])["elapsed_days"].median()
    med_stage = stage_rows.groupby("stage_index")["elapsed_days"].median()
    keys = list(zip(other["stage_index"], other["origin_lords"], strict=True))
    typical = pd.Series([med.get(k, med_stage.get(k[0], np.nan)) for k in keys], other.index)
    out = other.copy()
    out["pace_vs_typical"] = (other["elapsed_days"] + 1) / (typical + 1)
    return out


def matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).fillna(MISSING).to_numpy()


# ---------------------------------------------------------------------------
# Baselines (MODEL_SPEC §7)
# ---------------------------------------------------------------------------


def baseline_b1(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """B1: median days left of training rows at the same forecast point and origin House."""
    by_anchor_origin = train.groupby(["anchor", "origin_lords"])["target_days"].median()
    by_anchor = train.groupby("anchor")["target_days"].median()
    overall = train["target_days"].median()
    return np.array(
        [
            by_anchor_origin.get((a, o), by_anchor.get(a, overall))
            for a, o in zip(test["anchor"], test["origin_lords"], strict=True)
        ],
        dtype=float,
    )


def baseline_b0(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """B0: median introduction-to-Royal-Assent days by origin House, minus days elapsed."""
    start = train[train["anchor"] == "1st House: 1st reading"]
    total = start.groupby("origin_lords")["target_days"].median()
    pred = test["origin_lords"].map(total).to_numpy(float) - test["elapsed_days"].to_numpy(float)
    return np.maximum(pred, 1.0)


# ---------------------------------------------------------------------------
# Models and search spaces
# ---------------------------------------------------------------------------


def _log_target(model):
    """Fit on log(1 + days) so long bills don't dominate; predict back in days."""
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


MODELS: dict[str, tuple[object, dict, int]] = {
    "Decision tree": (
        DecisionTreeRegressor(random_state=SEED),
        {
            "regressor__max_depth": [2, 3, 4, 5, 6, 8, 10, 12, None],
            "regressor__min_samples_leaf": randint(1, 80),
            "regressor__max_features": [None, 0.5, 0.8],
            "regressor__criterion": ["squared_error", "absolute_error"],
        },
        60,
    ),
    "Random forest": (
        RandomForestRegressor(random_state=SEED, n_jobs=1),
        {
            "regressor__n_estimators": [200, 400, 800],
            "regressor__max_depth": [None, 6, 10, 16],
            "regressor__min_samples_leaf": [1, 3, 5, 10, 20, 40],
            "regressor__max_features": [0.2, 0.3, 0.5, 0.8, 1.0],
        },
        30,
    ),
    "Extra trees": (
        ExtraTreesRegressor(random_state=SEED, n_jobs=1),
        {
            "regressor__n_estimators": [200, 400, 800],
            "regressor__max_depth": [None, 6, 10, 16],
            "regressor__min_samples_leaf": [1, 3, 5, 10, 20, 40],
            "regressor__max_features": [0.2, 0.3, 0.5, 0.8, 1.0],
        },
        30,
    ),
    "Hist gradient boosting": (
        HistGradientBoostingRegressor(random_state=SEED),
        {
            "regressor__learning_rate": loguniform(0.01, 0.3),
            "regressor__max_iter": randint(100, 1000),
            "regressor__max_leaf_nodes": randint(7, 64),
            "regressor__min_samples_leaf": randint(5, 80),
            "regressor__l2_regularization": loguniform(1e-3, 10),
            "regressor__loss": ["squared_error", "absolute_error"],
        },
        50,
    ),
    "LightGBM": (
        LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1, subsample_freq=1),
        {
            "regressor__n_estimators": randint(100, 1500),
            "regressor__learning_rate": loguniform(0.01, 0.3),
            "regressor__num_leaves": randint(7, 64),
            "regressor__min_child_samples": randint(5, 80),
            "regressor__subsample": uniform(0.6, 0.4),
            "regressor__colsample_bytree": uniform(0.5, 0.5),
            "regressor__reg_lambda": loguniform(1e-3, 10),
            "regressor__objective": ["regression", "regression_l1"],
        },
        50,
    ),
    "XGBoost": (
        XGBRegressor(random_state=SEED, n_jobs=1, tree_method="hist"),
        {
            "regressor__n_estimators": randint(100, 1500),
            "regressor__learning_rate": loguniform(0.01, 0.3),
            "regressor__max_depth": randint(2, 9),
            "regressor__min_child_weight": loguniform(1, 30),
            "regressor__subsample": uniform(0.6, 0.4),
            "regressor__colsample_bytree": uniform(0.5, 0.5),
            "regressor__reg_lambda": loguniform(1e-3, 10),
            "regressor__objective": ["reg:squarederror", "reg:absoluteerror"],
        },
        50,
    ),
    "Linear (ridge)": (
        make_pipeline(StandardScaler(), Ridge()),
        {"regressor__ridge__alpha": loguniform(1e-3, 1e3)},
        30,
    ),
}
TREE_MODELS = [m for m in MODELS if m != "Linear (ridge)"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "Progress so far: gaps between stages": [g for g, _, _ in GAP_FEATURES],
    "Time elapsed and pace": [
        "elapsed_days", "elapsed_sitting_days", "days_since_last_event", "pace_vs_typical",
    ],
    "Position in the pipeline": ["stage_index", "in_second_house"],
    "Procedural events so far": [
        *EVENT_FLAGS, "h1_same_day_2r_3r", "h1_report_skipped", "n_sittings", "n_sitting_dates",
    ],
    "Session and Parliament timing": [
        "intro_days_into_session", "intro_parliament_age_years", "asof_days_into_session",
        "asof_parliament_age_years", "asof_month",
    ],
    "Bill type: origin, money bill, department": ["origin_lords", "money_bill_title"],
}


def cv_splits(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Index pairs for the expanding-window folds (validation sessions always later)."""
    sess = train["session_id"].to_numpy()
    return [
        (np.flatnonzero(np.isin(sess, list(tr))), np.flatnonzero(np.isin(sess, list(va))))
        for tr, va in CV_FOLDS
    ]


def tune(name: str, x: np.ndarray, y: np.ndarray, splits) -> tuple[object, dict, float]:
    base, space, n_iter = MODELS[name]
    search = RandomizedSearchCV(
        _log_target(base), space, n_iter=n_iter, cv=splits,
        scoring="neg_mean_absolute_error", n_jobs=-1, random_state=SEED, refit=True,
    )
    search.fit(x, y)
    params = {k.replace("regressor__", ""): _jsonable(v) for k, v in search.best_params_.items()}
    return search.best_estimator_, params, float(-search.best_score_)


def _jsonable(v):
    return v.item() if hasattr(v, "item") else v


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def error_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = np.abs(pred - y)
    return {
        "mae_days": float(err.mean()),
        "median_ae_days": float(np.median(err)),
        "rmse_days": float(np.sqrt(((pred - y) ** 2).mean())),
        "within_30_days": float((err <= 30).mean()),
        "relative_mae": float(err.mean() / max(y.mean(), 1e-9)),
        "n": int(len(y)),
    }


def bootstrap_mae(
    df: pd.DataFrame, preds: dict[str, np.ndarray], reps: int = 1000
) -> dict[str, tuple[float, float]]:
    """95% CI of MAE, resampling whole test bills (rows of one bill are not independent)."""
    rng = np.random.default_rng(SEED)
    bill_ids = df["bill_id"].to_numpy()
    uniq = np.unique(bill_ids)
    rows_of = {b: np.flatnonzero(bill_ids == b) for b in uniq}
    y = df["target_days"].to_numpy(float)
    samples = {k: [] for k in preds}
    for _ in range(reps):
        idx = np.concatenate([rows_of[b] for b in rng.choice(uniq, len(uniq))])
        for k, p in preds.items():
            samples[k].append(np.abs(p[idx] - y[idx]).mean())
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in samples.items()}


def paired_diff_ci(df: pd.DataFrame, a: np.ndarray, b: np.ndarray, reps: int = 1000):
    """95% CI of MAE(a) − MAE(b), resampling bills; negative = a is better."""
    rng = np.random.default_rng(SEED + 1)
    bill_ids = df["bill_id"].to_numpy()
    uniq = np.unique(bill_ids)
    rows_of = {u: np.flatnonzero(bill_ids == u) for u in uniq}
    y = df["target_days"].to_numpy(float)
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([rows_of[u] for u in rng.choice(uniq, len(uniq))])
        diffs.append(np.abs(a[idx] - y[idx]).mean() - np.abs(b[idx] - y[idx]).mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def per_anchor_mae(df: pd.DataFrame, preds: dict[str, np.ndarray], anchors) -> pd.DataFrame:
    y = df["target_days"].to_numpy(float)
    out = []
    for a in anchors:
        m = (df["anchor"] == a).to_numpy()
        if not m.any():
            continue
        row = {"anchor": a, "n": int(m.sum()), "mean_days_left": float(y[m].mean()),
               "median_days_left": float(np.median(y[m]))}
        row.update({k: float(np.abs(p[m] - y[m]).mean()) for k, p in preds.items()})
        out.append(row)
    return pd.DataFrame(out).set_index("anchor")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

STYLE = {
    "B0: typical total − elapsed": ("#9e9e9e", "--"),
    "B1: typical days left at this point": ("#5f5f5f", "--"),
    "Linear (ridge)": ("#8d6e63", ":"),
    "Decision tree": ("#e0a030", "-"),
    "Random forest": ("#2e7d5b", "-"),
    "Extra trees": ("#7cb342", "-"),
    "Hist gradient boosting": ("#5c6bc0", "-"),
    "LightGBM": ("#3b6ea5", "-"),
    "XGBoost": ("#c0504d", "-"),
}


def _short_anchor(a: str) -> str:
    return a.replace("1st House: ", "H1 ").replace("2nd House: ", "H2 ")


def plot_error_curve(table: pd.DataFrame, best: str, title: str, xlabel: str, path: Path):
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 2]},
    )
    x = np.arange(len(table))
    for name, (colour, ls) in STYLE.items():
        if name not in table.columns:
            continue
        lw = 3 if name == best else 1.4
        ax.plot(x, table[name], ls, color=colour, lw=lw, marker="o", ms=4 if name != best else 6,
                label=f"{name}  ← best" if name == best else name)
        ax2.plot(x, table[name] / table["mean_days_left"] * 100, ls, color=colour, lw=lw,
                 marker="o", ms=3)
    ax.bar(x, table["mean_days_left"], color="0.9", width=0.6, zorder=0,
           label="average days actually left")
    for xi, n in zip(x, table["n"], strict=True):
        ax.text(xi, 2, f"n={n}", ha="center", fontsize=7, color="0.4")
    ax.set_ylabel("Mean absolute error (days)")
    ax.set_title(title, fontsize=12, loc="left")
    ax.legend(fontsize=8, ncols=2, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax2.set_ylabel("Error as % of days left")
    ax2.set_xticks(x, [_short_anchor(a) for a in table.index], rotation=30, ha="right")
    ax2.set_xlabel(xlabel)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(overall: dict, ci: dict, path: Path) -> None:
    names = sorted(overall, key=lambda k: overall[k]["mae_days"], reverse=True)
    maes = [overall[k]["mae_days"] for k in names]
    lo = [overall[k]["mae_days"] - ci[k][0] for k in names]
    hi = [ci[k][1] - overall[k]["mae_days"] for k in names]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    colours = [STYLE[k][0] for k in names]
    ax.barh(names, maes, xerr=[lo, hi], color=colours, capsize=3)
    for i, v in enumerate(maes):
        ax.text(v + hi[i] + 1, i, f"{v:.0f} days", va="center", fontsize=9)
    ax.set_xlabel("Mean absolute error on test bills, days (all stage forecast points)")
    ax.set_title("Which model predicts the days left to Royal Assent best?\n"
                 "Whiskers = 95% bootstrap interval over test bills", fontsize=11)
    ax.set_xlim(0, max(maes) * 1.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, best: str, path: Path) -> None:
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    ax.barh(top.index, top["mean"], xerr=top["std"], color="#3b6ea5", capsize=2)
    ax.set_xlabel("Increase in test MAE (days) when the feature is shuffled")
    ax.set_title(f"Most useful features ({best}, permutation importance)", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation(abl: pd.DataFrame, best: str, path: Path) -> None:
    a = abl.sort_values("mae_increase_days")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    colours = ["#c0504d" if v > 0 else "#7cb342" for v in a["mae_increase_days"]]
    ax.barh(a.index, a["mae_increase_days"], color=colours)
    for i, v in enumerate(a["mae_increase_days"]):
        ax.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(min(0.0, a["mae_increase_days"].min()) - 1, a["mae_increase_days"].max() * 1.2)
    ax.set_xlabel("Change in cross-validated MAE (days) when the group is removed")
    ax.set_title(f"Which feature groups matter? ({best}; red = model gets worse without it)",
                 fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_vs_actual(df: pd.DataFrame, pred: np.ndarray, best: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6.5), layout="constrained")
    second = df["in_second_house"].to_numpy() == 1
    y = df["target_days"].to_numpy(float)
    for mask, colour, label in ((~second, "#2e7d5b", "forecast made in 1st House"),
                                (second, "#a8323e", "forecast made in 2nd House / ping-pong")):
        ax.scatter(y[mask] + 1, pred[mask] + 1, s=8, alpha=0.4, color=colour, label=label)
    lim = (1, max(y.max(), pred.max()) * 1.3)
    ax.plot(lim, lim, color="black", lw=1, label="perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual days left to Royal Assent (+1)")
    ax.set_ylabel("Predicted days left (+1)")
    ax.set_title(f"{best}: predicted vs actual (test bills, stage forecast points)", fontsize=11)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(report_dir: Path = REPORT_DIR) -> dict:
    t0 = time.time()
    report_dir.mkdir(parents=True, exist_ok=True)
    data = build_dataset(load_source_data(), train_sessions=set(TRAIN_SESSIONS))
    train = data[data["session_id"].isin(TRAIN_SESSIONS)].reset_index(drop=True)
    test = data[data["session_id"].isin(TEST_SESSIONS)].reset_index(drop=True)
    test = add_pace(train, test)
    # Training rows get pace from all-training medians. Within cross-validation this lets a
    # validation fold's own rows shape its medians (a small optimistic bias in CV scores only);
    # test rows always use training-only medians.
    train_full = add_pace(train, train)
    cols = feature_columns(train_full)
    x_tr, y_tr = matrix(train_full, cols), train_full["target_days"].to_numpy(float)
    x_te, y_te = matrix(test, cols), test["target_days"].to_numpy(float)
    splits = cv_splits(train_full)

    fitted, tuning = {}, {}
    for name in MODELS:
        t = time.time()
        est, params, cv_mae = tune(name, x_tr, y_tr, splits)
        fitted[name] = est
        tuning[name] = {"best_params": params, "cv_mae_days": cv_mae,
                        "seconds": round(time.time() - t, 1)}
        print(f"{name:24s} CV MAE {cv_mae:6.1f} days  ({tuning[name]['seconds']}s)")

    preds = {name: np.maximum(est.predict(x_te), 0.0) for name, est in fitted.items()}
    preds["B0: typical total − elapsed"] = baseline_b0(train_full, test)
    preds["B1: typical days left at this point"] = baseline_b1(train_full, test)

    stage = (test["anchor_type"] == "stage").to_numpy()
    timed = ~stage
    overall_stage = {k: error_metrics(y_te[stage], p[stage]) for k, p in preds.items()}
    overall_time = {k: error_metrics(y_te[timed], p[timed]) for k, p in preds.items()}
    test_stage = test[stage].reset_index(drop=True)
    ci = bootstrap_mae(test_stage, {k: p[stage] for k, p in preds.items()})
    best = min(TREE_MODELS, key=lambda k: tuning[k]["cv_mae_days"])  # chosen on CV, not test
    b1 = "B1: typical days left at this point"
    diff_vs_b1 = paired_diff_ci(test_stage, preds[best][stage], preds[b1][stage])

    by_stage = per_anchor_mae(test_stage, {k: p[stage] for k, p in preds.items()}, STAGE_ANCHORS)
    by_time = per_anchor_mae(
        test[timed].reset_index(drop=True), {k: p[timed] for k, p in preds.items()},
        [f"{n} days in" for n in TIME_ANCHOR_DAYS],
    )

    # Fast-tracked vs regular test bills (bill-level, only for reporting).
    h1_3r = test_stage[test_stage["anchor"] == "1st House: 3rd reading"]
    fast_bills = set(h1_3r.loc[h1_3r["elapsed_days"] <= 14, "bill_id"])
    is_fast = test_stage["bill_id"].isin(fast_bills).to_numpy()
    by_type = {
        group: {k: error_metrics(test_stage["target_days"].to_numpy(float)[m], p[stage][m])
                for k, p in preds.items() if k in (best, b1)}
        for group, m in (("fast_tracked", is_fast), ("regular", ~is_fast))
    }

    # Permutation importance on test stage rows for the best model.
    perm = permutation_importance(
        fitted[best], x_te[stage], y_te[stage], scoring="neg_mean_absolute_error",
        n_repeats=10, random_state=SEED, n_jobs=-1,
    )
    imp = pd.DataFrame({"mean": perm.importances_mean, "std": perm.importances_std},
                       index=cols).sort_values("mean", ascending=False)

    # Group ablation on the cross-validation folds (never looks at the test set).
    groups = {g: list(c) for g, c in FEATURE_GROUPS.items()}
    groups["Bill type: origin, money bill, department"] += [
        c for c in cols if c.startswith("dept_")
    ]
    base_cv = _cv_mae(fitted[best].regressor, x_tr, y_tr, splits)
    ablation = {}
    for g, gcols in groups.items():
        keep = [i for i, c in enumerate(cols) if c not in gcols]
        ablation[g] = {"mae_increase_days": _cv_mae(
            fitted[best].regressor, x_tr[:, keep], y_tr, splits) - base_cv,
            "features": len(cols) - len(keep)}
    abl = pd.DataFrame(ablation).T

    # Charts
    plot_error_curve(
        by_stage, best,
        "How the prediction error shrinks as the forecast starts later in the pipeline\n"
        "(test bills, sessions 2019–2026; grey bars = average days actually left)",
        "Forecast made at the start of this stage (H1 = first House, H2 = second House)",
        report_dir / "1_error_by_stage.png",
    )
    plot_error_curve(
        by_time, best,
        "Prediction error by time since the bill was introduced (test bills)",
        "Forecast made this many days after introduction",
        report_dir / "2_error_by_days_since_introduction.png",
    )
    plot_model_comparison(overall_stage, ci, report_dir / "3_model_comparison.png")
    plot_importance(imp, best, report_dir / "4_feature_importance.png")
    plot_ablation(abl, best, report_dir / "5_feature_group_ablation.png")
    plot_pred_vs_actual(test_stage, preds[best][stage], best,
                        report_dir / "6_predicted_vs_actual.png")

    # Tables
    by_stage.round(1).to_csv(report_dir / "mae_by_stage.csv")
    by_time.round(1).to_csv(report_dir / "mae_by_days_since_introduction.csv")
    out = test[["bill_id", "short_title", "session_id", "anchor_type", "anchor", "as_of",
                "target_days"]].copy()
    for k, p in preds.items():
        out[k] = p.round(1)
    out.to_csv(report_dir / "test_predictions.csv", index=False)
    imp.round(2).to_csv(report_dir / "feature_importance.csv")

    metrics = {
        "data": {
            "rows_train": int(len(train_full)), "rows_test": int(len(test)),
            "bills_train": int(train_full["bill_id"].nunique()),
            "bills_test": int(test["bill_id"].nunique()),
            "train_sessions": sorted(TRAIN_SESSIONS), "test_sessions": sorted(TEST_SESSIONS),
            "features": len(cols), "fast_tracked_test_bills": len(fast_bills),
        },
        "best_model_by_cv": best,
        "tuning": tuning,
        "test_stage_points": overall_stage,
        "test_time_points": overall_time,
        "test_stage_mae_95ci": ci,
        "best_minus_b1_mae_95ci": diff_vs_b1,
        "by_bill_type": by_type,
        "feature_group_ablation_cv": ablation,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _cv_mae(base, x: np.ndarray, y: np.ndarray, splits) -> float:
    errs = []
    for tr, va in splits:
        m = _log_target(clone(base)).fit(x[tr], y[tr])
        errs.append(np.abs(np.maximum(m.predict(x[va]), 0) - y[va]).mean())
    return float(np.mean(errs))


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: m[k] for k in ("best_model_by_cv", "best_minus_b1_mae_95ci")}, indent=2))
    for k, v in sorted(m["test_stage_points"].items(), key=lambda kv: kv[1]["mae_days"]):
        print(f"{k:38s} test MAE {v['mae_days']:6.1f}  median AE {v['median_ae_days']:6.1f}")
