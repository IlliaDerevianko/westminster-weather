"""Command-line entry point for the Phase 1 → 2 take-up model.

uv run python -m westminster_weather.takeup.cli fetch-tier-b --from 2015-01-01 \
    [--sources wms spoken written members]
uv run python -m westminster_weather.takeup.cli migrate [--dry-run]
uv run python -m westminster_weather.takeup.cli load-raw [--dry-run] [--cache-dir DIR]
uv run python -m westminster_weather.takeup.cli db-status [--dry-run]
uv run python -m westminster_weather.takeup.cli nlp [--dry-run] [--gemini] [--clear-events]
uv run python -m westminster_weather.takeup.cli review [--out DIR] [--seed N]
uv run python -m westminster_weather.takeup.cli score [--out DIR]

Every database command writes only to DATABASE_URL_TAKEUP (``takeup.db``) and refuses to run
without it. ``--dry-run`` builds the frames from local files and prints counts and sizes
without creating any database engine (``nlp --dry-run`` reads the database but writes
nothing). Connection strings are never printed.
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from westminster_weather.takeup import db as takeup_db
from westminster_weather.takeup import loaders, nlp, tier_b
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


def _load_nlp_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    """Raw sources (take-up DB), reference tables (production, read-only), long titles."""
    from westminster_weather.takeup import bill_text

    try:
        sources = takeup_db.load_source_tables()
    except takeup_db.TakeupDatabaseNotConfigured as exc:
        sys.exit(f"refused: {exc}")
    reference = takeup_db.load_reference_tables()
    long_titles = bill_text.bill_long_titles(reference["bills"]["bill_id"], fetch=False)
    return sources, reference, long_titles


def _print_summary(summary: dict) -> None:
    for key, value in summary.items():
        if key == "ideas_by_year":
            print("ideas_by_year (ideas, linked):")
            for year, (n, linked) in value.items():
                print(f"  {year}: {n} ({linked})")
        else:
            print(f"{key}: {value}")


def cmd_nlp(args: argparse.Namespace) -> None:
    import time

    t0 = time.perf_counter()
    sources, reference, long_titles = _load_nlp_inputs()
    fetched = nlp.fetched_members_from_cache()
    if fetched is None:
        print("note: members fetch cache not found; members with no posts count as non-ministers")
    gemini = nlp.GeminiClient() if args.gemini else None
    result = nlp.run_pipeline(
        sources["source_documents"],
        sources["member_government_posts"],
        reference["bills"],
        reference["stage_sittings"],
        long_titles,
        fetched_members=fetched,
        labeller=gemini.label if gemini else None,
        judge=gemini.same_bill if gemini else None,
    )
    _print_summary(nlp.summarize(result))
    if args.dry_run:
        print(f"dry run: nothing written ({time.perf_counter() - t0:.0f} s)")
        return
    engine = _engine()
    try:
        written = nlp.write_results(engine, result, clear_events=args.clear_events)
    except RuntimeError as exc:
        sys.exit(f"refused: {exc}")
    finally:
        engine.dispose()
    print("written:", ", ".join(f"{k} {v:,}" for k, v in written.items()))
    print(f"total runtime {time.perf_counter() - t0:.0f} s")


def cmd_review(args: argparse.Namespace) -> None:
    from westminster_weather.takeup import bill_text

    try:
        tables = takeup_db.load_takeup_tables()
    except takeup_db.TakeupDatabaseNotConfigured as exc:
        sys.exit(f"refused: {exc}")
    reference = takeup_db.load_reference_tables()
    long_titles = bill_text.bill_long_titles(reference["bills"]["bill_id"], fetch=False)
    bills = nlp.prepare_bills(reference["bills"], reference["stage_sittings"], long_titles)
    for path, status in nlp.export_review(tables, bills, args.out, seed=args.seed).items():
        print(f"{path}: {status}")
    print(
        "Fill the 'correct' column with y / n (links: correct_bill_id if another bill is "
        "right), then run: score"
    )


def cmd_score(args: argparse.Namespace) -> None:
    results = nlp.score(args.out)
    if not results:
        print(f"no review CSVs in {args.out}; run: review")
    for name, res in results.items():
        if not res.get("labelled"):
            print(f"{name}: nothing labelled yet")
            continue
        prec = res["precision"]
        print(
            f"{name}: precision {'n/a' if prec is None else f'{prec:.0%}'} "
            f"({res['positives_checked']} kept/linked checked), wrongly rejected "
            f"{res['wrongly_rejected']} of {res['rejected_checked']} rejected checked"
            + (
                f", {res['corrected_bill_ids']} corrected bill ids"
                if "corrected_bill_ids" in res
                else ""
            )
        )


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

    run = db_command("nlp", "Label, group, link and write the NLP take-up tables", cmd_nlp)
    run.add_argument("--gemini", action="store_true", help="Use Gemini (needs GEMINI_API_KEY)")
    run.add_argument(
        "--clear-events",
        action="store_true",
        help="Also clear idea_events (they refer to idea ids that a rerun renumbers)",
    )
    review = sub.add_parser("review", help="Export review_mentions.csv and review_links.csv")
    review.add_argument("--out", type=Path, default=nlp.REVIEW_DIR)
    review.add_argument("--seed", type=int, default=0)
    review.set_defaults(func=cmd_review)
    score = sub.add_parser("score", help="Precision from the hand-labelled review CSVs")
    score.add_argument("--out", type=Path, default=nlp.REVIEW_DIR)
    score.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
