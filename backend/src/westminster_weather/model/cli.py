"""Command-line entry point for the passage model.

uv run python -m westminster_weather.model.cli fetch [--gov-only]
"""

from __future__ import annotations

import argparse
import logging

from westminster_weather.data import bills_api


def cmd_fetch(args: argparse.Namespace) -> None:
    counts = bills_api.fetch_all(include_all_types=not args.gov_only)
    print(
        f"Listed {counts['listed']} bills ({counts['government']} government); "
        f"downloaded {counts['downloaded']} new or changed."
    )
    tables = bills_api.build_tables()
    bills, sittings = tables["bills"], tables["stage_sittings"]
    print(f"bills.parquet: {len(bills)} government bills")
    print(f"stage_sittings.parquet: {len(sittings)} sittings")
    print(f"all_stage_sittings.parquet: {len(tables['all_stage_sittings'])} sittings")
    print("Outcomes:")
    print(bills["outcome"].value_counts().to_string())


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="westminster_weather.model.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download and cache Bills API data, write parquet")
    fetch.add_argument(
        "--gov-only",
        action="store_true",
        help="Skip non-government bills (they are only used for the sitting-day calendar)",
    )
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
