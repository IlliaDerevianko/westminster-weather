"""Command-line entry point for ingesting all UK Parliament Bills into Neon Lakebase Postgres.

uv run python -m westminster_weather.model.cli fetch
uv run python -m westminster_weather.model.cli states [--as-of YYYY-MM-DD]
uv run python -m westminster_weather.model.cli build [--as-of YYYY-MM-DD]
uv run python -m westminster_weather.model.cli train [--as-of YYYY-MM-DD]
uv run python -m westminster_weather.model.cli forecast --bill-id <BILL_ID> [--as-of YYYY-MM-DD]
uv run python -m westminster_weather.model.cli forecast --examples
uv run python -m westminster_weather.model.cli evaluate [--as-of YYYY-MM-DD]
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

    _train_hazard(args.as_of)
    _train_failure_and_pingpong(args.as_of)


def _train_hazard(as_of: date) -> None:
    import json

    import numpy as np

    from westminster_weather.model import features, hazard

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

    model_path = fitted.save(MODELS_DIR / "hazard_train.joblib")
    live_rows = features.build_training_rows(as_of)
    live = hazard.fit_hazard_with_c(live_rows, fitted.C, val)
    live_path = live.save(MODELS_DIR / "hazard_live.joblib")
    print(
        f"Live hazard refit on all {live_rows.rows['bill_id'].nunique()} bills "
        f"({len(live_rows.rows):,} rows, data as of {as_of}): {live_path}"
    )
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


# Historical bills at different stages, forecast with the training models (BUILD_PLAN task 9).
EXAMPLES: tuple[tuple[int, date], ...] = (
    (3137, date(2022, 3, 17)),  # Online Safety Bill at Commons 1st reading
    (3064, date(2022, 2, 10)),  # Commercial Rent (Coronavirus) Bill at Lords committee
    (3462, date(2024, 5, 1)),  # Renters (Reform) Bill at Lords 1st/2nd reading
)


def _print_forecast(response) -> None:
    print(
        f"\n{response.short_title} (bill {response.bill_id}) as of {response.as_of} "
        f"[{response.model_variant} models]"
    )
    stage = response.current_stage.value if response.current_stage else "-"
    print(f"  Current stage: {stage.replace('_', ' ')}")
    print(
        f"  Chance of Royal Assent: {100 * response.passage_probability:.0f}% "
        f"(before this session ends: {100 * response.p_ra_before_session_end:.0f}%)"
    )
    t = response.time_to_royal_assent
    if t:
        print(
            f"  Royal Assent if it passes: median {t.estimated_median_date} "
            f"(80% between {t.p10_date} and {t.p90_date})"
        )
    risks = ", ".join(
        f"{reason.value.replace('_', ' ')} {100 * p:.1f}%"
        for reason, p in response.primary_stop_risks.items()
    )
    print(f"  Risks of failing: {risks}")
    if response.stages:
        print("  Remaining stages (chance of reaching, median date reached):")
        for s in response.stages:
            when = s.entry_date_p50 or "-"
            print(f"    {s.model_state.value.replace('_', ' '):<32}{100 * s.p_reach:5.0f}%  {when}")


def cmd_forecast(args: argparse.Namespace) -> None:
    from westminster_weather.data import interface
    from westminster_weather.model import forecast

    if not args.examples:
        if args.bill_id is None:
            raise SystemExit("--bill-id is required (or use --examples)")
        _print_forecast(forecast.forecast_bill(args.bill_id, args.as_of))
        return

    rows = []
    today_bills = interface.get_bills(date.today()).set_index("bill_id")
    for bill_id, as_of in EXAMPLES:
        response = forecast.forecast_bill(bill_id, as_of, variant="train")
        _print_forecast(response)
        actual = today_bills.loc[bill_id]
        print(f"  WHAT HAPPENED: {actual['outcome'].replace('_', ' ')} on {actual['outcome_date']}")
        rows.append((response, actual))
    chart = forecast.plot_examples(rows, REPORTS_DIR / "historical_forecasts.png")
    print(f"\nChart saved: {chart}")


def _comparison_table(columns: dict[str, dict]) -> pd.DataFrame:
    """Before/after rows for the evaluation (one column per backtest)."""
    rows: dict[str, dict[str, str]] = {}
    for name, m in columns.items():
        mae, carried = m["mae_days"], m["mae_days_carried_over_bills"]
        rows.setdefault("Average error, all stages (days)", {})[name] = f"{mae['model']:.0f}"
        for state, v in m["mae_days_by_start_state"].items():
            label = f"  from {state.replace('_', ' ')} (n={v['n']})"
            rows.setdefault(label, {})[name] = f"{v['p50_err']:.0f}"
        rows.setdefault("Median error (days)", {})[name] = (
            f"{m['median_abs_error_days']['model']:.0f}"
        )
        rows.setdefault("80% window coverage", {})[name] = f"{100 * m['coverage_p10_p90']:.0f}%"
        if "share_median_too_early" in m:
            rows.setdefault("Median forecast too early", {})[name] = (
                f"{100 * m['share_median_too_early']:.0f}%"
            )
        rows.setdefault("Average forecast pass chance vs actual", {})[name] = (
            f"{100 * m['mean_predicted_passage']:.0f}% vs {100 * m['actual_passage_rate']:.0f}%"
        )
        rows.setdefault("Brier (pass) vs always-the-pass-rate", {})[name] = (
            f"{m['brier_passage_probability']:.3f} vs {m['brier_passage_base_rate']:.3f}"
        )
        rows.setdefault(f"Error for carried-over bills (days, n={carried['n']})", {})[name] = (
            f"{carried['model']:.0f}"
        )
        rows.setdefault("B1 average error (days)", {})[name] = f"{mae['B1']:.0f}"
    return pd.DataFrame(rows).T


SPEED_SD_GRID = (0.0, 0.3, 0.5, 0.7, 0.9, 1.2)


def cmd_tune_speed(args: argparse.Namespace) -> None:
    """Choose the per-path speed spread so 80% windows cover ~80% on training-era sessions."""
    import json

    from westminster_weather.model import baseline, evaluate, hazard
    from westminster_weather.model.splits import TRAIN_CUTOFF, TUNING_SESSIONS

    points = evaluate.backtest_points(TRAIN_CUTOFF, sessions=TUNING_SESSIONS)
    C = hazard.HazardModel.load(MODELS_DIR / "hazard_train.joblib").C
    print(
        f"Tuning on {len(points):,} forecasts for {points['bill_id'].nunique()} bills from "
        f"sessions {TUNING_SESSIONS} (data as of {TRAIN_CUTOFF}); test sessions not used"
    )
    fitted_b = baseline.fit()
    results = {}
    for sd in SPEED_SD_GRID:
        res = evaluate.run_backtest(
            points, workers=args.workers, mode="rolling", evaluated_as_of=TRAIN_CUTOFF, C=C,
            speed_sd=sd,
        )  # fmt: skip
        m = evaluate.score(res, fitted_b)
        results[sd] = {k: m[k] for k in ("coverage_p10_p90", "share_median_too_early")}
        results[sd]["mae_days"] = m["mae_days"]["model"]
        results[sd]["brier"] = m["brier_passage_probability"]
        print(
            f"  s = {sd:<4} coverage {100 * m['coverage_p10_p90']:.0f}%  MAE "
            f"{m['mae_days']['model']:.0f}  too early {100 * m['share_median_too_early']:.0f}%"
            f"  Brier {m['brier_passage_probability']:.3f}"
        )
    best = min(results, key=lambda sd: abs(results[sd]["coverage_p10_p90"] - 0.80))
    print(f"Chosen speed spread s = {best}")
    path = REPORTS_DIR / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics["speed_sd_tuning"] = {
        "sessions": list(TUNING_SESSIONS),
        "grid": {str(k): v for k, v in results.items()},
        "chosen": best,
    }
    path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")


def cmd_evaluate(args: argparse.Namespace) -> None:
    import json
    import time

    from westminster_weather.model import baseline, evaluate, hazard

    started = time.perf_counter()
    points = evaluate.backtest_points(args.as_of)
    print(
        f"Backtest: {len(points):,} forecasts for {points['bill_id'].nunique()} test bills "
        f"({int(points['ra'].sum()):,} for bills that became law)"
    )
    fitted_b = baseline.fit()
    C = hazard.HazardModel.load(MODELS_DIR / "hazard_train.joblib").C
    metrics_path = REPORTS_DIR / "metrics.json"
    saved = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    columns: dict[str, dict] = dict(saved.get("backtest_comparison", {}))
    speed_sd = saved.get("speed_sd_tuning", {}).get("chosen", 0.0)
    print(f"Per-path speed spread s = {speed_sd}")
    before = PROCESSED_DIR / "backtest_before.parquet"
    if not columns and before.exists():
        columns["Before (frozen 2019, old session end)"] = evaluate.score(
            pd.read_parquet(before), fitted_b
        )
    modes = ["frozen", "rolling"] if args.also_frozen else ["rolling"]
    for mode in modes:
        results = evaluate.run_backtest(
            points,
            workers=args.workers,
            mode=mode,
            evaluated_as_of=args.as_of,
            C=C,
            speed_sd=speed_sd,
        )
        slug = "".join(c if c.isalnum() else "_" for c in args.label.lower()).strip("_")
        name = f"backtest_{mode}" if mode == "frozen" else f"backtest_{slug}"
        results.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        label = {"frozen": "New session end, frozen 2019", "rolling": args.label}
        columns[label[mode]] = evaluate.score(results, fitted_b)
        print(f"  {mode} backtest done ({time.perf_counter() - started:.0f} s)")
    metrics = columns[label["rolling"]]
    print()
    print(_comparison_table(columns).to_string())
    print(f"\nConcordance index (after): {metrics['concordance_index']}")
    print("Calibration (after):")
    print(pd.DataFrame(metrics["calibration"]).to_string(index=False))

    evaluate.save_metrics(metrics, REPORTS_DIR / "metrics.json")
    all_metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    all_metrics["backtest_comparison"] = {k: v for k, v in columns.items()}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str))
    report = evaluate.write_report(all_metrics, REPORTS_DIR / "EVALUATION.md")
    for chart in (
        evaluate.plot_mae_by_state(metrics, REPORTS_DIR / "mae_by_state.png"),
        evaluate.plot_model_vs_baselines(metrics, REPORTS_DIR / "model_vs_baselines.png"),
        evaluate.plot_calibration(metrics, REPORTS_DIR / "calibration.png"),
    ):
        print(f"Chart saved: {chart}")
    print(f"Report saved: {report}")


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

    fc = sub.add_parser("forecast", help="Forecast one bill, or the historical examples")
    fc.add_argument("--bill-id", type=int)
    fc.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    fc.add_argument("--examples", action="store_true", help="Three historical test bills")
    fc.set_defaults(func=cmd_forecast)

    ev = sub.add_parser("evaluate", help="Backtest on test sessions, metrics and charts")
    ev.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    ev.add_argument("--workers", type=int, default=6)
    ev.add_argument(
        "--label", default="After (new + rolling)", help="Column name for this run in the table"
    )
    ev.add_argument(
        "--also-frozen", action="store_true", help="Also run the frozen-2019 models for comparison"
    )
    ev.set_defaults(func=cmd_evaluate)

    tune = sub.add_parser("tune-speed", help="Tune the per-path speed spread (training era)")
    tune.add_argument("--workers", type=int, default=6)
    tune.set_defaults(func=cmd_tune_speed)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
