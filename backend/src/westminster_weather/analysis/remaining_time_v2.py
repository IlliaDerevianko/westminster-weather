"""Version 2 of the days-to-Royal-Assent tree models: more features, deeper tuning, ensembles.

Builds on ``remaining_time_models`` (v1) with the same bills, forecast points and time split, and
adds, in order:

1. **Extended features** (``build_dataset(extended=True)``): time spent in each House so far,
   bill-size proxies (sittings per stage), policy areas, congestion, calendar and political
   context, title signals.
2. **Target-derived features computed without leakage:** the rule-of-thumb value for the row
   (B1), the typical speed of the bill's policy area, and pace. Inside every validation fold they
   are re-computed from that fold's training bills; training rows get out-of-fold values.
3. **Design experiments on cross-validation** (LightGBM): feature groups, target
   (log days vs correction to the rule of thumb), sample weights, loss, monotonic constraints.
4. **Bayesian hyperparameter tuning** (Optuna TPE) of six tree model families on 5
   expanding-window folds.
5. **Ensembles:** seed-averaging and a CV-weighted blend.

Every choice is made on cross-validation inside the training sessions (2006–2019). The test
sessions (2019–2026) are scored once at the end.

Run from ``backend/`` (about 30–45 minutes)::

    uv run python -m westminster_weather.analysis.remaining_time_v2
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import GroupKFold  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from westminster_weather.analysis.remaining_time_dataset import (  # noqa: E402
    STAGE_ANCHORS,
    TIME_ANCHOR_DAYS,
    build_dataset,
    feature_columns,
    load_source_data,
)
from westminster_weather.analysis.remaining_time_models import (  # noqa: E402
    MISSING,
    TEST_SESSIONS,
    TRAIN_SESSIONS,
    add_pace,
    baseline_b1,
    bootstrap_mae,
    error_metrics,
    paired_diff_ci,
    per_anchor_mae,
)
from westminster_weather.analysis.remaining_time_models import (
    REPORT_DIR as V1_REPORT_DIR,
)

REPORT_DIR = V1_REPORT_DIR.parent / "remaining_time_models_v2"
SEED = 42
N_JOBS = -1

# Five expanding-window folds: always train on earlier sessions, validate on later ones.
CV_FOLDS: tuple[tuple[frozenset[int], frozenset[int]], ...] = tuple(
    (frozenset(range(20, min(val))), frozenset(val))
    for val in ({24}, {25}, {26, 27}, {28, 29}, {30})
)

# ---------------------------------------------------------------------------
# Feature groups (v1 features plus the new extended groups)
# ---------------------------------------------------------------------------

NEW_GROUPS: dict[str, list[str]] = {
    "Time in each House so far": [
        "commons_days_so_far", "lords_days_so_far", "first_house_days_so_far",
        "second_house_days_so_far", "current_house_days_so_far", "first_house_finished",
        "days_in_current_stage",
    ],
    "Bill size: sittings per stage": [
        "n_h1_2r_sittings", "n_h1_committee_sittings", "n_h1_report_sittings",
        "n_h2_committee_sittings", "n_h2_report_sittings", "n_ping_pong_rounds",
    ],
    "Calendar: recess and 5-year limit": [
        "days_to_summer_recess", "in_summer_recess", "days_to_parliament_5yr_limit",
    ],
    "Congestion: other bills in progress": ["active_government_bills", "bills_before_in_session"],
    "Politics: majority, coalition": ["government_majority", "coalition_or_minority"],
    "Title signals": ["title_words"],  # + title_* flags, added in _groups()
    "Policy area (merged departments)": [],  # area_* one-hots, added in _groups()
    "Rule of thumb + area speed (encoded)": ["b1_log", "area_speed_log", "area_resid_log"],
}


def _groups(cols: list[str]) -> dict[str, list[str]]:
    groups = {g: list(c) for g, c in NEW_GROUPS.items()}
    groups["Title signals"] += [c for c in cols if c.startswith("title_") and c != "title_words"]
    groups["Policy area (merged departments)"] += [c for c in cols if c.startswith("area_")]
    return groups


# ---------------------------------------------------------------------------
# Leak-free target-derived features
# ---------------------------------------------------------------------------

SMOOTHING = 10.0  # m-estimate: an area with n bills gets n / (n + 10) of its own mean


def _encodings_from(ref: pd.DataFrame) -> dict:
    """Statistics learned from ``ref`` (training rows only)."""
    ref = ref.assign(log_t=np.log1p(ref["target_days"]))
    b1 = ref.groupby(["anchor", "origin_lords"])["target_days"].median()
    b1_anchor = ref.groupby("anchor")["target_days"].median()
    resid = ref["log_t"] - np.log1p(
        [b1[(a, o)] for a, o in zip(ref["anchor"], ref["origin_lords"], strict=True)]
    )
    starts = ref[ref["anchor"] == "1st House: 1st reading"]
    g_speed = starts["log_t"].mean()
    speed = starts.groupby("policy_area")["log_t"].agg(["sum", "count"])
    r = pd.DataFrame({"area": ref["policy_area"], "resid": resid})
    resid_stats = r.groupby("area")["resid"].agg(["sum", "count"])
    return {
        "b1": b1, "b1_anchor": b1_anchor, "b1_all": ref["target_days"].median(),
        "speed": ((speed["sum"] + SMOOTHING * g_speed) / (speed["count"] + SMOOTHING)).to_dict(),
        "speed_global": g_speed,
        "resid": (resid_stats["sum"] / (resid_stats["count"] + SMOOTHING)).to_dict(),
    }


def _apply_encodings(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    out = df.copy()
    out["b1_log"] = np.log1p(
        [
            enc["b1"].get((a, o), enc["b1_anchor"].get(a, enc["b1_all"]))
            for a, o in zip(df["anchor"], df["origin_lords"], strict=True)
        ]
    )
    out["area_speed_log"] = df["policy_area"].map(enc["speed"]).fillna(enc["speed_global"])
    out["area_resid_log"] = df["policy_area"].map(enc["resid"]).fillna(0.0)
    return out


def prepare(train: pd.DataFrame, other: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add target-derived features to ``train`` (out-of-fold) and ``other`` (from all train).

    Out-of-fold for training rows means each bill's own target never feeds its own features.
    """
    parts = []
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(train, groups=train["bill_id"]):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        parts.append(add_pace(tr, _apply_encodings(va, _encodings_from(tr))))
    train_enc = pd.concat(parts).loc[train.index]
    other_enc = (
        add_pace(train, _apply_encodings(other, _encodings_from(train)))
        if other is not None else None
    )
    return train_enc, other_enc


# ---------------------------------------------------------------------------
# Design: target, weights, model families
# ---------------------------------------------------------------------------


@dataclass
class Design:
    features: list[str]
    target: str = "log"  # "log": log(1 + days); "resid": log days minus log rule of thumb
    weights: str = "none"  # "none", "bill" (each bill counts once), "bill_recent"
    monotone: bool = False
    extra: dict = field(default_factory=dict)


def target_of(df: pd.DataFrame, d: Design) -> np.ndarray:
    y = np.log1p(df["target_days"].to_numpy(float))
    return y - df["b1_log"].to_numpy(float) if d.target == "resid" else y


def to_days(pred: np.ndarray, df: pd.DataFrame, d: Design) -> np.ndarray:
    log_pred = pred + df["b1_log"].to_numpy(float) if d.target == "resid" else pred
    return np.maximum(np.expm1(log_pred), 0.0)


def weights_of(df: pd.DataFrame, d: Design) -> np.ndarray | None:
    if d.weights == "none":
        return None
    w = 1.0 / df.groupby("bill_id")["bill_id"].transform("size").to_numpy(float)
    if d.weights == "bill_recent":
        newest = df["session_id"].max()
        w = w * np.exp((df["session_id"].to_numpy(float) - newest) / 6.0)
    return w * len(w) / w.sum()


def matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).fillna(MISSING).to_numpy()


def make_model(family: str, p: dict, d: Design, cols: list[str], seed: int = SEED):
    mono = None
    if d.monotone:
        # Later in the pipeline can only mean the same or fewer days left, all else equal.
        dec = {"stage_index", "in_second_house", "first_house_finished"}
        mono = [-1 if c in dec else 0 for c in cols]
    if family == "LightGBM":
        return LGBMRegressor(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
            subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"], reg_alpha=p.get("reg_alpha", 0.0),
            objective=p["objective"], max_bin=p.get("max_bin", 255),
            monotone_constraints=mono, random_state=seed, n_jobs=N_JOBS, verbose=-1,
        )
    if family == "XGBoost":
        return XGBRegressor(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
            subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"], reg_alpha=p.get("reg_alpha", 0.0),
            gamma=p.get("gamma", 0.0),
            objective=p["objective"], tree_method="hist",
            monotone_constraints=tuple(mono) if mono else None,
            random_state=seed, n_jobs=N_JOBS,
        )
    if family == "CatBoost":
        return CatBoostRegressor(
            iterations=p["iterations"], learning_rate=p["learning_rate"], depth=p["depth"],
            l2_leaf_reg=p["l2_leaf_reg"], random_strength=p["random_strength"],
            bagging_temperature=p["bagging_temperature"], loss_function=p["loss_function"],
            random_seed=seed, thread_count=N_JOBS, verbose=0, allow_writing_files=False,
        )
    if family == "Hist gradient boosting":
        return HistGradientBoostingRegressor(
            learning_rate=p["learning_rate"], max_iter=p["max_iter"],
            max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"], loss=p["loss"],
            monotonic_cst=mono, random_state=seed,
        )
    if family in ("Random forest", "Extra trees"):
        cls = RandomForestRegressor if family == "Random forest" else ExtraTreesRegressor
        return cls(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
            random_state=seed, n_jobs=N_JOBS,
        )
    raise ValueError(family)


def search_space(family: str, trial: optuna.Trial) -> dict:
    if family == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 150, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
            "objective": trial.suggest_categorical(
                "objective", ["regression", "regression_l1", "huber"]),
        }
    if family == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "objective": trial.suggest_categorical(
                "objective", ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"]),
        }
    if family == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "loss_function": trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
        }
    if family == "Hist gradient boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1500, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 64, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 150, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 30, log=True),
            "loss": trial.suggest_categorical("loss", ["squared_error", "absolute_error"]),
        }
    if family in ("Random forest", "Extra trees"):
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [300, 600]),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 20]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.15, 1.0),
        }
    raise ValueError(family)


DEFAULT_LGBM = {
    "n_estimators": 600, "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 30,
    "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1.0, "reg_alpha": 0.01,
    "max_bin": 255, "objective": "regression",
}
N_TRIALS = {
    "LightGBM": 150, "XGBoost": 150, "CatBoost": 50, "Hist gradient boosting": 30,
    "Random forest": 25, "Extra trees": 25,
}

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


class Folds:
    """Pre-computed, leak-free fold data (encodings re-learned inside each fold)."""

    def __init__(self, train: pd.DataFrame) -> None:
        self.folds = []
        for tr_s, va_s in CV_FOLDS:
            tr = train[train["session_id"].isin(tr_s)]
            va = train[train["session_id"].isin(va_s)]
            self.folds.append(prepare(tr, va))

    def score(
        self, family: str, params: dict, d: Design, return_oof: bool = False, seed: int = SEED
    ):
        maes, oof = [], []
        for tr, va in self.folds:
            model = make_model(family, params, d, d.features, seed)
            model.fit(matrix(tr, d.features), target_of(tr, d), sample_weight=weights_of(tr, d))
            pred = to_days(model.predict(matrix(va, d.features)), va, d)
            maes.append(np.abs(pred - va["target_days"].to_numpy(float)).mean())
            oof.append(pred)
        return (float(np.mean(maes)), oof) if return_oof else float(np.mean(maes))


def tune(folds: Folds, family: str, d: Design, n_trials: int) -> tuple[dict, float, optuna.Study]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
    )
    study.optimize(lambda t: folds.score(family, search_space(family, t), d), n_trials=n_trials)
    return study.best_params, float(study.best_value), study


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(report_dir: Path = REPORT_DIR) -> dict:
    t0 = time.time()
    report_dir.mkdir(parents=True, exist_ok=True)
    log = []

    def note(msg: str) -> None:
        line = f"[{time.time() - t0:7.0f}s] {msg}"
        print(line, flush=True)
        log.append(line)

    data = build_dataset(load_source_data(), train_sessions=set(TRAIN_SESSIONS), extended=True)
    train = data[data["session_id"].isin(TRAIN_SESSIONS)].reset_index(drop=True)
    test = data[data["session_id"].isin(TEST_SESSIONS)].reset_index(drop=True)
    folds = Folds(train)
    all_cols = feature_columns(prepare(train, None)[0])
    groups = _groups(all_cols)
    new_cols = {c for g in groups.values() for c in g}
    v1_cols = [c for c in all_cols if c not in new_cols]
    note(f"rows: train {len(train)}, test {len(test)}; features: v1 {len(v1_cols)}, "
         f"all {len(all_cols)}")

    # ---- Stage A: design experiments with LightGBM (CV only) ----------------------------
    exp: dict[str, float] = {}
    d_v1 = Design(v1_cols)
    exp["v1 features (baseline)"] = folds.score("LightGBM", DEFAULT_LGBM, d_v1)
    # v1's tuned XGBoost on v1 features, re-scored on the new folds: the fair reference.
    v1_params = json.loads((V1_REPORT_DIR / "metrics.json").read_text())["tuning"]["XGBoost"][
        "best_params"]
    exp["v1 XGBoost setup (reference)"] = folds.score("XGBoost", v1_params, d_v1)
    for g, gcols in groups.items():
        exp[f"+ {g}"] = folds.score("LightGBM", DEFAULT_LGBM, Design(v1_cols + gcols))
    d_all = Design(all_cols)
    exp["all features"] = folds.score("LightGBM", DEFAULT_LGBM, d_all)
    note("feature groups: " + ", ".join(f"{k} {v:.1f}" for k, v in exp.items()))

    # Backward elimination from all features: drop a group if CV gets better without it.
    chosen = list(all_cols)
    best = exp["all features"]
    dropped = []
    for g, gcols in groups.items():
        trial_cols = [c for c in chosen if c not in gcols]
        score = folds.score("LightGBM", DEFAULT_LGBM, Design(trial_cols))
        exp[f"all − {g}"] = score
        if score < best - 0.1:
            chosen, best = trial_cols, score
            dropped.append(g)
    note(f"backward elimination dropped {dropped or 'nothing'}; CV {best:.1f}")

    design_trials = {}
    for target in ("log", "resid"):
        for weights in ("none", "bill", "bill_recent"):
            d = Design(chosen, target=target, weights=weights)
            design_trials[(target, weights)] = folds.score("LightGBM", DEFAULT_LGBM, d)
    (tgt, wts) = min(design_trials, key=design_trials.get)
    for (t, w), v in design_trials.items():
        exp[f"target={t}, weights={w}"] = v
    design = Design(chosen, target=tgt, weights=wts)
    exp["monotonic constraints"] = folds.score(
        "LightGBM", DEFAULT_LGBM, Design(chosen, tgt, wts, monotone=True))
    if exp["monotonic constraints"] < design_trials[(tgt, wts)] - 0.1:
        design.monotone = True
    note(f"design: target={tgt}, weights={wts}, monotone={design.monotone}")

    # ---- Stage B: Bayesian tuning per family -------------------------------------------
    tuned: dict[str, dict] = {}
    for family, n in N_TRIALS.items():
        t = time.time()
        params, cv, _ = tune(folds, family, design, n)
        tuned[family] = {"params": params, "cv_mae": cv, "seconds": round(time.time() - t)}
        note(f"tuned {family:24s} CV MAE {cv:6.2f}  ({tuned[family]['seconds']}s)")

    # ---- Stage C: ensembles on CV ------------------------------------------------------
    oof = {f: folds.score(f, tuned[f]["params"], design, return_oof=True)[1] for f in tuned}
    y_folds = [va["target_days"].to_numpy(float) for _, va in folds.folds]
    fams = list(tuned)

    def blend_cv(w: np.ndarray) -> float:
        w = np.abs(w) / np.abs(w).sum()
        return float(np.mean([
            np.abs(sum(wi * oof[f][k] for wi, f in zip(w, fams, strict=True)) - y_folds[k]).mean()
            for k in range(len(y_folds))
        ]))

    res = minimize(blend_cv, np.ones(len(fams)) / len(fams), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4})
    blend_w = np.abs(res.x) / np.abs(res.x).sum()
    blend_weights = {f: round(float(w), 3) for f, w in zip(fams, blend_w, strict=True)}
    top3 = sorted(fams, key=lambda f: tuned[f]["cv_mae"])[:3]
    candidates = {f: tuned[f]["cv_mae"] for f in fams}
    candidates["Blend (CV-weighted)"] = blend_cv(blend_w)
    candidates["Average of top 3"] = blend_cv(np.array([1.0 if f in top3 else 0 for f in fams]))
    final_name = min(candidates, key=candidates.get)
    note(f"blend weights {blend_weights}; candidates "
         + ", ".join(f"{k} {v:.2f}" for k, v in candidates.items()) + f" → final: {final_name}")

    # ---- Stage D: fit on all training sessions, score the test set once -----------------
    train_p, test_p = prepare(train, test)
    x_tr, x_te = matrix(train_p, design.features), matrix(test_p, design.features)
    y_tr, w_tr = target_of(train_p, design), weights_of(train_p, design)
    seeds = [SEED + i for i in range(5)]
    test_pred = {}
    for f in fams:
        preds = [
            to_days(make_model(f, tuned[f]["params"], design, design.features, s)
                    .fit(x_tr, y_tr, sample_weight=w_tr).predict(x_te), test_p, design)
            for s in (seeds if f not in ("Random forest", "Extra trees") else seeds[:1])
        ]
        test_pred[f] = np.mean(preds, axis=0)  # seed-averaged
    test_pred["Blend (CV-weighted)"] = sum(blend_w[i] * test_pred[f] for i, f in enumerate(fams))
    test_pred["Average of top 3"] = np.mean([test_pred[f] for f in top3], axis=0)

    # v1 best (XGBoost) and rule of thumb on exactly the same test rows.
    v1 = pd.read_csv(V1_REPORT_DIR / "test_predictions.csv")
    key = ["bill_id", "anchor_type", "anchor"]
    v1_aligned = test_p[key].merge(v1[key + ["XGBoost"]], on=key, how="left")["XGBoost"]
    assert v1_aligned.notna().all(), "v1 predictions must cover the same test rows"
    test_pred["v1 XGBoost (previous best)"] = v1_aligned.to_numpy(float)
    test_pred["Rule of thumb (B1)"] = baseline_b1(train_p, test_p)

    y_te = test_p["target_days"].to_numpy(float)
    stage = (test_p["anchor_type"] == "stage").to_numpy()
    ts = test_p[stage].reset_index(drop=True)
    sp = {k: p[stage] for k, p in test_pred.items()}
    overall = {k: error_metrics(y_te[stage], p) for k, p in sp.items()}
    overall_time = {k: error_metrics(y_te[~stage], p[~stage]) for k, p in test_pred.items()}
    ci = bootstrap_mae(ts, sp)
    vs_v1 = paired_diff_ci(ts, sp[final_name], sp["v1 XGBoost (previous best)"])
    vs_b1 = paired_diff_ci(ts, sp[final_name], sp["Rule of thumb (B1)"])
    by_stage = per_anchor_mae(ts, sp, STAGE_ANCHORS)
    by_time = per_anchor_mae(
        test_p[~stage].reset_index(drop=True), {k: p[~stage] for k, p in test_pred.items()},
        [f"{n} days in" for n in TIME_ANCHOR_DAYS],
    )
    note(f"TEST (stage points) final {final_name}: {overall[final_name]['mae_days']:.1f}; "
         f"v1 XGBoost {overall['v1 XGBoost (previous best)']['mae_days']:.1f}; "
         f"B1 {overall['Rule of thumb (B1)']['mae_days']:.1f}; diff vs v1 95% CI {vs_v1}")

    # Permutation importance of the final model on test stage rows, by feature group
    # (single-seed refits of the families the final model uses).
    used = fams if final_name in ("Blend (CV-weighted)", "Average of top 3") else [final_name]
    fitted = {
        f: make_model(f, tuned[f]["params"], design, design.features, SEED).fit(
            x_tr, y_tr, sample_weight=w_tr)
        for f in used
    }
    stage_df = test_p[stage]

    def final_predict(x: np.ndarray) -> np.ndarray:
        out = {f: to_days(m.predict(x), stage_df, design) for f, m in fitted.items()}
        if final_name == "Blend (CV-weighted)":
            return sum(blend_w[i] * out[f] for i, f in enumerate(fams))
        if final_name == "Average of top 3":
            return np.mean([out[f] for f in top3], axis=0)
        return out[final_name]

    rng = np.random.default_rng(SEED)
    x_st = x_te[stage]
    base_mae = np.abs(final_predict(x_st) - y_te[stage]).mean()
    group_of = {c: g for g, cs in _groups(all_cols).items() for c in cs}
    importance = {}
    for gname in sorted({group_of.get(c, "v1 features") for c in design.features}):
        idx = [i for i, c in enumerate(design.features) if group_of.get(c, "v1 features") == gname]
        incs = []
        for _ in range(5):
            xp = x_st.copy()
            perm = rng.permutation(len(xp))
            xp[:, idx] = xp[perm][:, idx]
            incs.append(np.abs(final_predict(xp) - y_te[stage]).mean() - base_mae)
        importance[gname] = (float(np.mean(incs)), float(np.std(incs)))

    # ---- Outputs -----------------------------------------------------------------------
    by_stage.round(1).to_csv(report_dir / "mae_by_stage.csv")
    by_time.round(1).to_csv(report_dir / "mae_by_days_since_introduction.csv")
    out = test_p[["bill_id", "short_title", "session_id", "anchor_type", "anchor", "as_of",
                  "target_days"]].copy()
    for k, p in test_pred.items():
        out[k] = np.round(p, 1)
    out.to_csv(report_dir / "test_predictions.csv", index=False)
    pd.Series(exp).round(2).to_csv(report_dir / "design_experiments_cv.csv", header=["cv_mae"])

    _plot_experiments(exp, report_dir / "3_what_helped_cv.png")
    _plot_stage_curve(by_stage, final_name, report_dir / "1_error_by_stage.png")
    _plot_models(overall, ci, final_name, report_dir / "2_model_comparison.png")
    _plot_importance(importance, final_name, report_dir / "4_feature_group_importance.png")

    metrics = {
        "data": {"rows_train": len(train), "rows_test": len(test),
                 "bills_train": int(train["bill_id"].nunique()),
                 "bills_test": int(test["bill_id"].nunique()),
                 "features_all": len(all_cols), "features_used": len(design.features)},
        "design": {"target": design.target, "weights": design.weights,
                   "monotone": design.monotone, "dropped_groups": dropped,
                   "features": design.features},
        "design_experiments_cv": exp,
        "tuned": tuned,
        "ensemble_candidates_cv": candidates,
        "blend_weights": blend_weights,
        "final_model": final_name,
        "test_stage_points": overall,
        "test_time_points": overall_time,
        "test_stage_mae_95ci": ci,
        "final_minus_v1_xgboost_mae_95ci": vs_v1,
        "final_minus_b1_mae_95ci": vs_b1,
        "group_permutation_importance_test": importance,
        "log": log,
        "runtime_seconds": round(time.time() - t0),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _plot_experiments(exp: dict, path: Path) -> None:
    base = exp["v1 features (baseline)"]
    keys = [k for k in exp if k.startswith("+ ")] + ["all features"]
    vals = [exp[k] - base for k in keys]
    order = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.barh([keys[i].removeprefix("+ ") for i in order], [vals[i] for i in order],
            color=["#1baf7a" if vals[i] < 0 else "#e34948" for i in order], height=0.6)
    for y, i in enumerate(order):
        ax.text(vals[i] + (0.1 if vals[i] >= 0 else -0.1), y, f"{vals[i]:+.1f}", va="center",
                ha="left" if vals[i] >= 0 else "right", fontsize=9, color=INK_SOFT)
    ax.axvline(0, color=INK, lw=0.8)
    lo, hi = min(vals + [0]), max(vals + [0])
    ax.set_xlim(lo - 1.2, hi + 1.2)
    ax.set_xlabel(f"Change in cross-validated error (days) vs v1 features ({base:.1f} days)")
    ax.set_title("Which new feature groups help? (green = lower error; LightGBM, CV only)",
                 fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_stage_curve(table: pd.DataFrame, final: str, path: Path) -> None:
    cols = [c for c in table.columns if c not in ("n", "mean_days_left", "median_days_left")]
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    x = np.arange(len(table))
    for c in cols:
        if c == final:
            ax.plot(x, table[c], color="#2a78d6", lw=3, marker="o", label=f"{c} (v2 final)")
        elif c == "v1 XGBoost (previous best)":
            ax.plot(x, table[c], color="#eb6834", lw=2, marker="o", label=c)
        elif c == "Rule of thumb (B1)":
            ax.plot(x, table[c], color="#8f8d87", lw=2, marker="o", label=c)
        else:
            ax.plot(x, table[c], color="#c3c2b7", lw=1, label=None)
    ax.set_xticks(x, [a.replace("1st House: ", "H1 ").replace("2nd House: ", "H2 ")
                      for a in table.index], rotation=30, ha="right")
    ax.set_ylabel("Mean absolute error on test bills (days)")
    ax.set_title("v2 against v1 and the rule of thumb, by forecast point "
                 "(thin grey = other v2 model families)", fontsize=11, loc="left")
    ax.yaxis.grid(True, color=GRID)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_models(overall: dict, ci: dict, final: str, path: Path) -> None:
    names = sorted(overall, key=lambda k: overall[k]["mae_days"], reverse=True)
    m = [overall[k]["mae_days"] for k in names]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    colours = ["#2a78d6" if k == final else "#eb6834" if k.startswith("v1") else
               "#8f8d87" if k.startswith("Rule") else "#9fc3ee" for k in names]
    ax.barh(names, m, xerr=[[v - ci[k][0] for k, v in zip(names, m, strict=True)],
                            [ci[k][1] - v for k, v in zip(names, m, strict=True)]],
            color=colours, capsize=3, height=0.6)
    for i, v in enumerate(m):
        ax.text(ci[names[i]][1] + 0.8, i, f"{v:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, max(ci[k][1] for k in names) * 1.12)
    ax.set_xlabel("Mean absolute error on test bills, days (stage forecast points; 95% CI)")
    ax.set_title("Test error by model (v2 tuned families, ensembles, v1 best, rule of thumb)",
                 fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_importance(imp: dict, final: str, path: Path) -> None:
    items = sorted(imp.items(), key=lambda kv: kv[1][0])
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.barh([k for k, _ in items], [v[0] for _, v in items], xerr=[v[1] for _, v in items],
            color="#2a78d6", capsize=3, height=0.6)
    ax.set_xlabel("Increase in test error (days) when the whole group is shuffled")
    ax.set_title(f"Which feature groups the final model relies on ({final})", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: m[k] for k in ("final_model", "final_minus_v1_xgboost_mae_95ci",
                                         "final_minus_b1_mae_95ci")}, indent=2))
