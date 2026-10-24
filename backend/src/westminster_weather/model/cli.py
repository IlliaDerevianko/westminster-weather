"""Command-line entry point for ingesting all UK Parliament Bills into Neon Lakebase Postgres.

uv run python -m westminster_weather.model.cli fetch
"""

from __future__ import annotations

import argparse
import logging

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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
