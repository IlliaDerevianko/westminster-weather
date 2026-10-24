"""Command-line entry point for the Phase 1 → 2 take-up model.

uv run python -m westminster_weather.takeup.cli fetch-tier-b --from 2015-01-01 \
    [--sources wms spoken written members]
"""

import argparse
import logging
from datetime import date

from westminster_weather.takeup import tier_b

TIER_B_SOURCES = ["wms", "spoken", "written", "members"]


def cmd_fetch_tier_b(args: argparse.Namespace) -> None:
    tier_b.fetch_all(args.start, args.end, args.sources)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="westminster_weather.takeup.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-tier-b", help="Cache Tier B sources (no database writes)")
    fetch.add_argument("--from", dest="start", type=date.fromisoformat, default=date(2015, 1, 1))
    fetch.add_argument("--to", dest="end", type=date.fromisoformat, default=date.today())
    fetch.add_argument(
        "--sources", nargs="+", choices=TIER_B_SOURCES, default=["wms", "spoken", "members"]
    )
    fetch.set_defaults(func=cmd_fetch_tier_b)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
