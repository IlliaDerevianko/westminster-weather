"""Command-line entry point for ingesting all UK Parliament Bills into Neon Lakebase Postgres.

uv run python -m westminster_weather.model.cli fetch [--gov-only]
uv run python -m westminster_weather.model.cli states [--as-of YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from westminster_weather.data import bills_api, calendar


def cmd_fetch(args: argparse.Namespace) -> None:
    tables, counts = bills_api.fetch_and_persist_all(include_all_types=not args.gov_only)
    print(
        f"Listed {counts['listed']} bills ({counts['government']} government); "
        f"downloaded {counts['downloaded']} payloads and persisted to Neon Lakebase Postgres."
    )
    for name, df in tables.items():
        if name == "all_stage_sittings":
            continue
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


REPORTS_DIR = Path(__file__).resolve().parents[3] / "reports"


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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="westminster_weather.model.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download Bills API data and persist to Neon Postgres")
    fetch.add_argument(
        "--gov-only",
        action="store_true",
        help="Skip non-government bills (they are only used for the sitting-day calendar)",
    )
    fetch.set_defaults(func=cmd_fetch)

    st = sub.add_parser("states", help="Sanity report and checks for model state transitions")
    st.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    st.set_defaults(func=cmd_states)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
