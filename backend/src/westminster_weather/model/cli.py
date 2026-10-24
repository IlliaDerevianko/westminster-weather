"""Command-line entry point for the passage model.

uv run python -m westminster_weather.model.cli fetch [--gov-only]
uv run python -m westminster_weather.model.cli build
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from westminster_weather.data import bills_api, calendar


def cmd_fetch(args: argparse.Namespace) -> None:
    counts = bills_api.fetch_all(include_all_types=not args.gov_only)
    print(
        f"Listed {counts['listed']} bills ({counts['government']} government); "
        f"downloaded {counts['downloaded']} new or changed."
    )
    cmd_build(args)


def cmd_build(args: argparse.Namespace) -> None:
    tables = bills_api.build_tables()
    for name, df in tables.items():
        print(f"{name + '.parquet':<28}{len(df):>7} rows")

    bills, sessions = tables["bills"], tables["sessions"]
    print("\nOutcomes:")
    outcomes = (
        bills.groupby(["outcome", "stop_reason"], dropna=False).size().rename("bills").reset_index()
    )
    print(outcomes.sort_values("bills", ascending=False).to_string(index=False))

    print("\nSessions:")
    counts = calendar.session_sitting_day_counts(sessions, tables["sitting_days"])
    print(sessions.merge(counts, on="session_id", how="left").to_string(index=False))

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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="westminster_weather.model.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download Bills API data, then build")
    fetch.add_argument(
        "--gov-only",
        action="store_true",
        help="Skip non-government bills (they are only used for the sitting-day calendar)",
    )
    fetch.set_defaults(func=cmd_fetch)

    build = sub.add_parser("build", help="Rebuild parquet tables from the cache")
    build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
