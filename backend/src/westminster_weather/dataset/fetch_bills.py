"""Stage 1: fetch every UK Government Bill and its stage sittings into the dataset tables.

Run from backend/:
    uv run python -m westminster_weather.dataset.fetch_bills
    uv run python -m westminster_weather.dataset.fetch_bills --limit 20 --db sqlite:///data/sample.db

Re-running is safe: rows are upserted by primary key, never duplicated.
"""

import argparse
import asyncio
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, make_url

from westminster_weather.clients.parliament_bills import ParliamentBillsClient
from westminster_weather.dataset.build import Dataset, build_dataset
from westminster_weather.dataset.tables import bills, metadata, stage_map, stage_sittings, upsert

DEFAULT_DB_URL = "sqlite:///data/westminster_weather.db"

RawBill = tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]


async def fetch_raw(
    limit: int | None, concurrency: int
) -> tuple[list[dict[str, Any]], list[RawBill]]:
    async with ParliamentBillsClient(timeout=30.0) as client:
        stage_types = await client.list_stage_types()
        government_bills = await client.list_all_government_bills()
        if limit:
            government_bills = government_bills[:limit]
        print(f"{len(government_bills)} Government Bills listed; fetching stages and sponsors")

        gate = asyncio.Semaphore(concurrency)
        done = 0

        async def fetch_one(bill: dict[str, Any]) -> RawBill:
            nonlocal done
            async with gate:
                stages = await client.get_all_bill_stages(bill["billId"])
                detail = await client.get_bill(bill["billId"])
            done += 1
            if done % 50 == 0 or done == len(government_bills):
                print(f"  {done}/{len(government_bills)} bills fetched")
            return bill, stages, detail

        raw = await asyncio.gather(*(fetch_one(b) for b in government_bills))
    return stage_types, list(raw)


def write(db_url: str, ds: Dataset) -> None:
    url = make_url(db_url)
    if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    metadata.create_all(engine)
    with engine.begin() as conn:
        upsert(conn, stage_map, ds.stage_map)
        upsert(conn, bills, ds.bills)
        upsert(conn, stage_sittings, ds.stage_sittings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB_URL, help=f"SQLAlchemy URL (default {DEFAULT_DB_URL})"
    )
    parser.add_argument("--limit", type=int, help="only fetch the first N bills (quick test runs)")
    parser.add_argument("--concurrency", type=int, default=5, help="parallel bill requests")
    args = parser.parse_args()

    stage_types, raw = asyncio.run(fetch_raw(args.limit, args.concurrency))
    ingested_at = datetime.now(UTC).replace(tzinfo=None)
    ds = build_dataset(stage_types, raw, as_of=date.today(), ingested_at=ingested_at)
    write(args.db, ds)

    print(
        f"\nwrote {len(ds.bills)} bills, {len(ds.stage_sittings)} stage sittings, "
        f"{len(ds.stage_map)} stage types to {args.db}"
    )
    if ds.skipped_not_introduced:
        print(f"skipped {ds.skipped_not_introduced} bills with no sittings yet")
    for outcome, n in Counter(b["outcome"] for b in ds.bills).most_common():
        print(f"  {outcome:18s} {n}")
    if ds.problems:
        print(f"\n{len(ds.problems)} validation problems:")
        for problem in ds.problems[:20]:
            print("  " + problem)


if __name__ == "__main__":
    main()
