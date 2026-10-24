"""Command-line entry point for ingesting all UK Parliament Bills into Neon Lakebase Postgres.

uv run python -m westminster_weather.model.cli fetch
uv run python -m westminster_weather.model.cli states [--as-of YYYY-MM-DD]
uv run python -m westminster_weather.model.cli build [--as-of YYYY-MM-DD]
uv run python -m westminster_weather.model.cli train [--as-of YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from westminster_weather.data import bills_api, calendar


def cmd_fetch(args: argparse.Namespace) -> None:
    tables, counts = bills_api.fetch_and_persist_all()
    print(
        f"Listed {counts['listed']} bills ({counts['government']} government); "
        f"downloaded {counts['downloaded']} payloads and persisted to Neon Lakebase Postgres."
    )
    for name, df in tables.items():
        print(f"  {name:<24}{len(df):>7} rows in Neon Postgres")

    bills, sessions = tables["bills"], tables["sessions"]
    print("\nOutcomes:")
    outcomes = (
        bills.groupby(["outcome", "stop_reason"], dropna=False).size().rename("bills").reset_index()
    )
    print(outcomes.sort_values("bills", ascending=False).to_string(index=False))

    print("\nSessions:")
    session_counts = calendar.session_sitting_day_counts(sessions, tables["sitting_days"])
    print(sessions.merge(session_counts, on="session_id", how="left").to_string(index=False))

    current = sessions["session_id"].max()
    in_progress = bills[bills["outcome"] == "in_progress"]
    ok = (
        in_progress["included_session_ids"].map(lambda s: current in s)
        | in_progress["carry_over_date"].notna()
    )
    print(
        f"\nIn progress: {len(in_progress)} bills; "
        f"{(~ok).sum()} break the current-session-or-carry-over rule"
    )
    print(
        in_progress[["bill_id", "short_title", "introduced_session_id", "carry_over_date"]]
        .assign(carry_over_date=lambda d: pd.to_datetime(d["carry_over_date"]).dt.date)
        .to_string(index=False)
    )


BACKEND_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = BACKEND_DIR / "reports"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
MODELS_DIR = BACKEND_DIR / "models"


def cmd_states(args: argparse.Namespace) -> None:
    from westminster_weather.model import states

    as_of = args.as_of
    report = states.sanity_report(as_of)
    print(f"Model population as of {as_of}: {report['bills']} bills (sessions >= 20)\n")
    print("Bills by final state and how they ended:")
    print(report["by_final_state"].to_string())
    print("\nSitting days spent in each completed state:")
    print(report["durations"].round(1).to_string())
    print(f"\nBills with ping-pong: {report['bills_with_ping_pong']}")

    print("\nChecks:")
    for check in states.check_transitions(as_of):
        print(f"  [{'OK' if check.ok else 'FAIL'}] {check.name}: {len(check.failures)}")
        if not check.ok:
            print(check.failures.head(10).to_string(index=False))

    chart = states.plot_durations(report["transitions"], REPORTS_DIR / "state_durations.png")
    print(f"\nChart saved: {chart}")


def cmd_build(args: argparse.Namespace) -> None:
    from westminster_weather.model import features

    training = features.build_training_rows(args.as_of)
    rows = training.rows
    print(f"Hazard training rows as of {args.as_of}: {len(rows):,}")
    print(
        f"Bills: {rows['bill_id'].nunique()}  Positive rate (moved on that day): "
        f"{training.positive_rate:.3f}"
    )
    print("\nBy state:")
    print(
        rows.groupby("state")["y"].agg(rows="size", moved="sum", rate="mean").round(3).to_string()
    )
    print("\nBy time in state (tau):")
    print(rows.groupby("tau_bucket")["y"].agg(rows="size", rate="mean").round(3).to_string())
    print("\nBy sitting days to session end:")
    print(
        rows.groupby("days_to_session_end_bucket")["y"]
        .agg(rows="size", rate="mean")
        .round(3)
        .to_string()
    )
    print(f"\nFeatures ({len(training.feature_columns)}): {', '.join(training.feature_columns)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "training_rows.parquet"
    rows.to_parquet(out, index=False)
    chart = features.plot_advance_rates(rows, REPORTS_DIR / "advance_rates.png")
    print(f"\nRows saved: {out}\nChart saved: {chart}")


def cmd_train(args: argparse.Namespace) -> None:
    from westminster_weather.model import baseline

    fitted = baseline.fit()
    points, metrics = baseline.evaluate(fitted, args.as_of)
    print(
        f"Baselines fitted on {fitted.n_train_acts} Acts from sessions "
        f"{metrics['train_sessions'][0]}-{metrics['train_sessions'][-1]} "
        f"(data as of {metrics['train_cutoff']})"
    )
    print(
        "B0 median days from 1st reading to Royal Assent:",
        {k: round(v) for k, v in fitted.b0_days.items()},
    )
    print(
        f"\nTest: {metrics['n_test_bills']} Acts from sessions "
        f"{metrics['test_sessions'][0]}-{metrics['test_sessions'][-1]}, "
        f"{metrics['n_forecasts']} forecasts (one per stage reached)"
    )
    print(f"  B0 MAE: {metrics['B0']['mae_days']} days")
    print(f"  B1 MAE: {metrics['B1']['mae_days']} days")
    by_state = pd.DataFrame(
        {"B0": metrics["B0"]["mae_days_by_state"], "B1": metrics["B1"]["mae_days_by_state"]}
    )
    print("\nMAE by stage the forecast was made from (days):")
    print(by_state.to_string())
    out = baseline.save_metrics(metrics, REPORTS_DIR / "metrics.json")
    chart = baseline.plot_mae_by_state(metrics, REPORTS_DIR / "baselines_mae_by_state.png")
    print(f"\nMetrics saved: {out}\nChart saved: {chart}")

    _train_hazard()
    _train_failure_and_pingpong(args.as_of)


def _train_hazard() -> None:
    import json

    import numpy as np

    from westminster_weather.model import hazard

    built = hazard.training_rows()
    fitted = hazard.fit_hazard(built)
    val = fitted.validation
    print(
        f"\nAdvance hazard: {len(built.rows):,} training rows from "
        f"{built.rows['bill_id'].nunique()} bills"
    )
    print(
        f"Validation (fit sessions {val['fit_sessions'][0]}-{val['fit_sessions'][-1]}, "
        f"score {val['validation_sessions'][0]}-{val['validation_sessions'][-1]}), "
        "log-loss (lower is better):"
    )
    print(f"  constant daily rate          {val['log_loss_constant_rate']}")
    print(f"  state x time-in-state only   {val['log_loss_state_and_time_only']}")
    print(f"  full logistic model (C={fitted.C})  {val['log_loss_best']}")
    lgbm = hazard.lightgbm_validation(built)
    val["lightgbm_log_loss"] = lgbm["log_loss"]
    print(f"  LightGBM (optional)          {lgbm['log_loss']}")

    coefs = fitted.coefficients()
    shared = coefs[[c for c in coefs.index if not c.startswith("cell:")]].sort_values()
    print("\nShared effects (odds ratio per day; >1 faster, <1 slower):")
    for col, value in shared.items():
        print(f"  {hazard.label(col, fitted.departments):<62}{np.exp(value):6.2f}")

    model_path = fitted.save(MODELS_DIR / "hazard.joblib")
    chart = hazard.plot_coefficients(fitted, REPORTS_DIR / "hazard_coefficients.png")
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["hazard"] = {
        **val,
        "C": fitted.C,
        "training_rows": len(built.rows),
        "odds_ratios": {c: round(float(np.exp(v)), 3) for c, v in shared.items()},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nModel saved: {model_path}\nChart saved: {chart}\nMetrics saved: {metrics_path}")


def _train_failure_and_pingpong(as_of: date) -> None:
    import json

    import joblib
    import numpy as np

    from westminster_weather.model import failure, pingpong
    from westminster_weather.model.splits import TRAIN_CUTOFF, TRAIN_SESSIONS

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for name, cutoff, sessions in (
        ("train", TRAIN_CUTOFF, TRAIN_SESSIONS),
        ("live", as_of, None),
    ):
        fail = failure.fit_failure(cutoff, sessions=sessions)
        pp = pingpong.fit_pingpong(cutoff, sessions=sessions)
        scope = "training sessions" if sessions else "all sessions"
        print(f"\nFailure routes and ping-pong ({name}: data as of {cutoff}, {scope}):")
        for line in [*fail.describe(), *pp.describe()]:
            print(f"  - {line}")
        fail.save(MODELS_DIR / f"failure_{name}.joblib")
        joblib.dump(pp, MODELS_DIR / f"pingpong_{name}.joblib")
        if name == "live":
            chart = failure.plot_components(fail, pp, REPORTS_DIR / "failure_and_pingpong.png")
            print(f"  Chart saved: {chart}")
        metrics.setdefault("failure", {})[name] = {
            "data_as_of": cutoff.isoformat(),
            "session_lengths_sitting_days": sorted(int(x) for x in fail.session_lengths.lengths),
            "dissolution_daily_rate": fail.dissolution.rates,
            "dissolution_events": fail.dissolution.events,
            "withdrawal_daily_rate": fail.withdrawal.rate,
            "defeat_probability_per_vote_stage": fail.defeat_probability,
            "ping_pong_probability": pp.probability,
            "ping_pong_median_sitting_days": float(np.median(pp.durations)),
        }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nModels saved in {MODELS_DIR}; metrics saved: {metrics_path}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="westminster_weather.model.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch", help="Download UK Government Bills and persist them to Neon Postgres"
    )
    # Kept so existing commands still work; government bills only is now always the case.
    fetch.add_argument("--gov-only", action="store_true", help=argparse.SUPPRESS)
    fetch.set_defaults(func=cmd_fetch)

    st = sub.add_parser("states", help="Sanity report and checks for model state transitions")
    st.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    st.set_defaults(func=cmd_states)

    build = sub.add_parser("build", help="Build hazard training rows (task 4)")
    build.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    build.set_defaults(func=cmd_build)

    train = sub.add_parser(
        "train", help="Fit and score the models (baselines, hazard, failure routes, ping-pong)"
    )
    train.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    train.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
