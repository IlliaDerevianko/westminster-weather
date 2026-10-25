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
"""

from __future__ import annotations

import json
import re
import warnings

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


def run() -> dict:
    src = load_source_data()
    df = build_dataset(src, TRAIN_SESSIONS, extended=True)
    lt = long_title_features(sorted(df["bill_id"].astype(int).unique()))
    df = df.merge(lt, on="bill_id", how="left")
    lt_cols = [c for c in lt.columns if c != "bill_id"]

    v2 = json.loads((V2_DIR / "metrics.json").read_text())
    base_cols = v2["design"]["features"]
    params = v2["tuned"]["XGBoost"]["params"]

    train = df[df["session_id"].isin(TRAIN_SESSIONS)]
    test = df[df["session_id"].isin(TEST_SESSIONS)]
    train_p, test_p = prepare(train, test)

    variants = {
        "v2 XGBoost": Design(base_cols),
        "+ long title": Design(base_cols + lt_cols),
        "+ long title, bill weights": Design(base_cols + lt_cols, weights="bill"),
        "+ long title, recent bills weigh more": Design(base_cols + lt_cols, weights="bill_recent"),
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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    first = out[out["anchor"] == FIRST_READING].copy()
    first["band"] = pd.cut(first["target_days"], BANDS, labels=BAND_LABELS)
    inside = (first["p10"] <= first["target_days"]) & (first["target_days"] <= first["p90"])
    result: dict = {
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
    (REPORT_DIR / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    r = run()
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
    print(f"\nwrote {REPORT_DIR}")


if __name__ == "__main__":
    main()
