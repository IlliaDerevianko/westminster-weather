"""Take-up database access: engines, migration, upserts and loaders (DATA_CONTRACT §10).

All take-up reads and writes go to ``DATABASE_URL_TAKEUP`` (pooled) and, optionally,
``DATABASE_URL_TAKEUP_UNPOOLED`` (direct, for DDL), i.e. the Neon branch ``takeup-dev``.
If ``DATABASE_URL_TAKEUP`` is not set the engines raise, even if the unpooled take-up URL is
set; they never fall back to the production ``DATABASE_URL``.

Reference tables (bills, stage sittings, sessions, parliaments, sitting days) are read from
production through the existing ``westminster_weather.db`` adapters and never written here.
"""

from typing import Any

import pandas as pd
from sqlalchemy import Engine, Table, create_engine, select
from sqlalchemy.pool import NullPool

from westminster_weather import db as app_db
from westminster_weather.config import settings
from westminster_weather.takeup.tables import DATA_TABLES, TABLES, takeup_metadata


class TakeupDatabaseNotConfigured(RuntimeError):
    """DATABASE_URL_TAKEUP is missing, or points at the production database."""


def _checked_url(direct: bool) -> str:
    pooled = settings.sqlalchemy_takeup_pooled_url
    if not pooled:
        raise TakeupDatabaseNotConfigured(
            "DATABASE_URL_TAKEUP is not set. Create the Neon branch 'takeup-dev' and add "
            "DATABASE_URL_TAKEUP (and optionally DATABASE_URL_TAKEUP_UNPOOLED) to .env.local. "
            "The take-up code never falls back to the production DATABASE_URL."
        )
    url = settings.sqlalchemy_takeup_direct_url if direct else pooled
    production = {settings.sqlalchemy_pooled_url, settings.sqlalchemy_direct_url}
    if url in production:
        raise TakeupDatabaseNotConfigured(
            "The take-up database URL is the same as the production DATABASE_URL; "
            "point DATABASE_URL_TAKEUP at the 'takeup-dev' branch instead."
        )
    return url


def get_takeup_engine() -> Engine:
    """Sync engine on the pooled take-up URL, for reads and upserts."""
    return create_engine(
        _checked_url(direct=False),
        poolclass=NullPool,
        connect_args={"prepare_threshold": None},
    )


def get_takeup_direct_engine() -> Engine:
    """Sync engine on the direct (unpooled) take-up URL, for DDL."""
    return create_engine(_checked_url(direct=True), poolclass=NullPool)


def migrate() -> list[str]:
    """Create the take-up tables (if missing) on the take-up database; returns their names."""
    engine = get_takeup_direct_engine()
    try:
        with engine.begin() as conn:
            takeup_metadata.create_all(conn)
        return sorted(takeup_metadata.tables)
    finally:
        engine.dispose()


def _table(table: Table | str) -> Table:
    return TABLES[table] if isinstance(table, str) else table


def upsert(
    table: Table | str, rows: pd.DataFrame | list[dict[str, Any]], batch_size: int = 500
) -> int:
    """Insert or update ``rows`` by primary key (Postgres ON CONFLICT); returns the row count."""
    table = _table(table)
    if table.name not in takeup_metadata.tables:
        raise ValueError(f"{table.name} is not a take-up table")
    records = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
    records = [app_db._clean_record(r) for r in records]
    if not records:
        return 0
    engine = get_takeup_engine()
    try:
        with engine.begin() as conn:
            app_db._upsert_rows(conn, table, records, batch_size=batch_size)
        return len(records)
    finally:
        engine.dispose()


def load_takeup_tables() -> dict[str, pd.DataFrame]:
    """The six take-up data tables (§10.1–10.6) from the take-up database, by table name."""
    engine = get_takeup_engine()
    try:
        with engine.connect() as conn:
            out = {}
            for table in DATA_TABLES:
                rows = conn.execute(select(table)).mappings().all()
                out[table.name] = pd.DataFrame(
                    [dict(r) for r in rows], columns=[c.name for c in table.columns]
                )
            return out
    finally:
        engine.dispose()


PARLIAMENT_COLUMNS = ["parliament_id", "first_meeting_date", "dissolution_date", "election_date"]


def _reference_queries() -> dict[str, Any]:
    """Plain SELECTs on the five production reference tables, in the adapters' order."""
    from westminster_weather import models as m

    ss = m.StageSittingRow.__table__
    return {
        "bills": select(m.BillRow.__table__).order_by(m.BillRow.bill_id),
        "stage_sittings": select(ss).order_by(ss.c.bill_id, ss.c.event_date, ss.c.sort_order),
        "sessions": select(m.ParliamentSessionRow.__table__).order_by(
            m.ParliamentSessionRow.session_id
        ),
        "parliaments": select(m.ParliamentRow.__table__).order_by(m.ParliamentRow.parliament_id),
        "sitting_days": select(m.SittingDayRow.__table__).order_by(
            m.SittingDayRow.date, m.SittingDayRow.house
        ),
    }


def load_reference_tables() -> dict[str, pd.DataFrame]:
    """Production reference tables (bills, stage_sittings, sessions, parliaments, sitting_days).

    Read-only: plain SELECTs in a read-only transaction on the pooled production engine. The
    ``westminster_weather.db.fetch_*`` adapters are deliberately not used, because they first
    run ``ensure_reference_tables_seeded_sync()`` (``create_all`` plus merges and a commit). If
    ``parliaments`` is empty, ``DEFAULT_PARLIAMENTS`` is used in memory (nothing is written).
    """
    engine = app_db.get_sync_pooled_engine()
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(postgresql_readonly=True)
            out = {}
            for name, query in _reference_queries().items():
                result = conn.execute(query)
                out[name] = pd.DataFrame(result.mappings().all(), columns=list(result.keys()))
            conn.rollback()
    finally:
        engine.dispose()
    if out["parliaments"].empty:
        out["parliaments"] = pd.DataFrame(app_db.DEFAULT_PARLIAMENTS, columns=PARLIAMENT_COLUMNS)
    return out
