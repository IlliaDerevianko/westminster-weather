"""Command-line entry point for the Phase 1 → 2 take-up model.

uv run python -m westminster_weather.takeup.cli fetch-tier-b --from 2015-01-01 \
    [--sources wms spoken written members]
uv run python -m westminster_weather.takeup.cli migrate [--dry-run]
uv run python -m westminster_weather.takeup.cli load-raw [--dry-run] [--cache-dir DIR]
uv run python -m westminster_weather.takeup.cli db-status [--dry-run]

Every database command writes only to DATABASE_URL_TAKEUP (``takeup.db``) and refuses to run
without it. ``--dry-run`` builds the frames from local files and prints counts and sizes
without creating any database engine. Connection strings are never printed.
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from westminster_weather.takeup import db as takeup_db
from westminster_weather.takeup import loaders, tier_b
from westminster_weather.takeup.tables import TABLES

TIER_B_SOURCES = ["wms", "spoken", "written", "members"]


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.1f} MB"


def _frame_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(deep=True).sum())


def _print_frame(name: str, df: pd.DataFrame) -> None:
    print(f"{name}: {len(df):,} rows, ~{_mb(_frame_bytes(df))} in memory")
    for key, value in df.attrs.get("report", {}).items():
        if key != "rows":
            print(f"  {key}: {value}")


def _engine():
    """Pooled take-up engine; exits with a message (no URL) if DATABASE_URL_TAKEUP is unset."""
    try:
        return takeup_db.get_takeup_engine()
    except takeup_db.TakeupDatabaseNotConfigured as exc:
        sys.exit(f"refused: {exc}")


def cmd_fetch_tier_b(args: argparse.Namespace) -> None:
    tier_b.fetch_all(args.start, args.end, args.sources)


def cmd_migrate(args: argparse.Namespace) -> None:
    if args.dry_run:
        print("dry run: would create (if missing):", ", ".join(sorted(TABLES)))
        return
    try:
        names = takeup_db.migrate()
    except takeup_db.TakeupDatabaseNotConfigured as exc:
        sys.exit(f"refused: {exc}")
    print("take-up tables present:", ", ".join(names))


def cmd_load_raw(args: argparse.Namespace) -> None:
    docs = loaders.build_source_documents(args.cache_dir)
    posts, fetched = loaders.build_member_government_posts(args.cache_dir)
    _print_frame("source_documents", docs)
    _print_frame("member_government_posts", posts)
    if args.dry_run:
        print(f"dry run: nothing written ({len(fetched):,} members fetched)")
        return
    engine = _engine()
    try:
        n_docs = loaders.load_source_documents(engine, docs)
        n_posts = loaders.load_member_government_posts(engine, posts)
    finally:
        engine.dispose()
    print(f"upserted {n_docs:,} source_documents and {n_posts:,} member_government_posts")


def cmd_db_status(args: argparse.Namespace) -> None:
    if args.dry_run:
        print("dry run: would count rows in", ", ".join(sorted(TABLES)), "and pg_database_size")
        return
    from sqlalchemy import func, inspect, select, text

    engine = _engine()
    try:
        with engine.connect() as conn:
            present = set(inspect(conn).get_table_names())
            for name in sorted(TABLES):
                if name in present:
                    n = conn.execute(select(func.count()).select_from(TABLES[name])).scalar_one()
                    print(f"{name}: {n:,} rows")
                else:
                    print(f"{name}: (missing; run migrate)")
            if conn.dialect.name == "postgresql":
                size = conn.execute(
                    text("SELECT pg_database_size(current_database())")
                ).scalar_one()
                print(f"database size: {_mb(int(size))}")
            conn.rollback()
    finally:
        engine.dispose()


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

    def db_command(name: str, help_text: str, func) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dry-run", action="store_true", help="Build and report only; no database")
        p.set_defaults(func=func)
        return p

    db_command("migrate", "Create the take-up tables on DATABASE_URL_TAKEUP", cmd_migrate)
    raw = db_command("load-raw", "Load source_documents and member_government_posts", cmd_load_raw)
    raw.add_argument("--cache-dir", type=Path, default=tier_b.CACHE_DIR)
    db_command("db-status", "Row counts per take-up table and database size", cmd_db_status)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
