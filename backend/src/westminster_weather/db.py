"""Neon Lakebase Postgres engine, session management, and reference table DB adapters.

Follows neon-postgres skill best practices:
- Uses `DATABASE_URL` (pooled `-pooler` via PgBouncer) for application queries with
  `prepare_threshold=None` (disabling server-side prepared statements on transaction pooler).
- Uses `DATABASE_URL_UNPOOLED` (direct connection) for schema migrations / DDL.
"""

from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import Connection, Engine, Table, create_engine, delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.pool import NullPool

from westminster_weather.config import settings

DEFAULT_PARLIAMENTS: list[tuple[int, date, date | None, date | None]] = [
    (1, date(2001, 6, 13), date(2005, 4, 11), date(2005, 5, 5)),
    (2, date(2005, 5, 11), date(2010, 4, 12), date(2010, 5, 6)),
    (3, date(2010, 5, 18), date(2015, 3, 30), date(2015, 5, 7)),
    (4, date(2015, 5, 18), date(2017, 5, 3), date(2017, 6, 8)),
    (5, date(2017, 6, 13), date(2019, 11, 6), date(2019, 12, 12)),
    (6, date(2019, 12, 17), date(2024, 5, 30), date(2024, 7, 4)),
    (7, date(2024, 7, 9), None, None),
]

DEFAULT_SESSION_OVERRIDES: list[tuple[int, date | None, date | None, str | None, str | None]] = [
    (
        38,
        None,
        date(2024, 5, 24),
        "dissolution",
        "Prorogued 2024-05-24, dissolved 2024-05-30; the API has a stray 2024-06-14 sitting",
    ),
]


class Base(DeclarativeBase):
    pass


def get_pooled_engine() -> AsyncEngine:
    """Create async engine against Neon's pooled DATABASE_URL."""
    return create_async_engine(
        settings.sqlalchemy_pooled_url,
        poolclass=NullPool,
        connect_args={"prepare_threshold": None},
    )


def get_direct_engine() -> AsyncEngine:
    """Create async engine against Neon's direct DATABASE_URL_UNPOOLED for DDL/migrations."""
    return create_async_engine(
        settings.sqlalchemy_direct_url,
        poolclass=NullPool,
    )


def get_sync_pooled_engine() -> Engine:
    """Create synchronous engine against Neon's pooled DATABASE_URL for DataFrame DB adapters."""
    return create_engine(
        settings.sqlalchemy_pooled_url,
        poolclass=NullPool,
        connect_args={"prepare_threshold": None},
    )


def get_sync_direct_engine() -> Engine:
    """Create synchronous engine against Neon's direct DATABASE_URL_UNPOOLED for DDL/seeding."""
    return create_engine(
        settings.sqlalchemy_direct_url,
        poolclass=NullPool,
    )


pooled_engine = get_pooled_engine()
SessionLocal = async_sessionmaker(pooled_engine, expire_on_commit=False)
_reference_tables_initialized = False


def _seed_reference_tables(conn: Connection) -> None:
    """Create schema and idempotently seed `parliaments` and `session_overrides` in Neon."""
    from westminster_weather.models import ParliamentRow, SessionOverrideRow

    Base.metadata.create_all(conn)
    with Session(bind=conn) as session:
        for pid, first_meeting, dissolution, election in DEFAULT_PARLIAMENTS:
            session.merge(
                ParliamentRow(
                    parliament_id=pid,
                    first_meeting_date=first_meeting,
                    dissolution_date=dissolution,
                    election_date=election,
                )
            )
        for sid, start_dt, end_dt, end_type, note in DEFAULT_SESSION_OVERRIDES:
            session.merge(
                SessionOverrideRow(
                    session_id=sid,
                    start_date=start_dt,
                    end_date=end_dt,
                    end_type=end_type,
                    note=note,
                )
            )
        session.commit()


def ensure_reference_tables_seeded_sync() -> None:
    """Ensure `parliaments` and `session_overrides` exist and are seeded in Neon Postgres."""
    global _reference_tables_initialized
    if _reference_tables_initialized:
        return
    engine = get_sync_direct_engine()
    try:
        with engine.begin() as conn:
            _seed_reference_tables(conn)
        _reference_tables_initialized = True
    finally:
        engine.dispose()


async def init_db() -> None:
    """Create database tables and seed reference tables using the direct Neon connection."""
    global _reference_tables_initialized
    import westminster_weather.models  # noqa: F401

    engine = get_direct_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(_seed_reference_tables)
        _reference_tables_initialized = True
    finally:
        await engine.dispose()


async def check_db_connection() -> str:
    """Verify connectivity to Neon Lakebase Postgres and return Postgres version string."""
    async with SessionLocal() as session:
        result = await session.execute(text("SELECT version()"))
        return str(result.scalar_one())


def fetch_parliaments_df() -> pd.DataFrame:
    """DB adapter: read `parliaments` reference table from Neon Lakebase Postgres as a DataFrame."""
    from westminster_weather.models import ParliamentRow

    ensure_reference_tables_seeded_sync()
    engine = get_sync_pooled_engine()
    try:
        with Session(engine) as session:
            rows = (
                session.execute(select(ParliamentRow).order_by(ParliamentRow.parliament_id))
                .scalars()
                .all()
            )
            return pd.DataFrame(
                [
                    (
                        r.parliament_id,
                        r.first_meeting_date,
                        r.dissolution_date,
                        r.election_date,
                    )
                    for r in rows
                ],
                columns=[
                    "parliament_id",
                    "first_meeting_date",
                    "dissolution_date",
                    "election_date",
                ],
            )
    finally:
        engine.dispose()


def fetch_session_overrides_df() -> pd.DataFrame:
    """DB adapter: read `session_overrides` table from Neon Lakebase Postgres as a DataFrame."""
    from westminster_weather.models import SessionOverrideRow

    ensure_reference_tables_seeded_sync()
    engine = get_sync_pooled_engine()
    try:
        with Session(engine) as session:
            rows = (
                session.execute(
                    select(SessionOverrideRow).order_by(SessionOverrideRow.session_id)
                )
                .scalars()
                .all()
            )
            df = pd.DataFrame(
                [
                    {
                        "session_id": r.session_id,
                        "start_date": r.start_date,
                        "end_date": r.end_date,
                        "end_type": r.end_type,
                        "note": r.note,
                    }
                    for r in rows
                ],
                columns=["session_id", "start_date", "end_date", "end_type", "note"],
            )
            df["end_type"] = df["end_type"].astype("string")
            return df
    finally:
        engine.dispose()


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, list):
            cleaned[k] = v
        elif pd.isna(v):
            cleaned[k] = None
        elif isinstance(v, pd.Timestamp):
            cleaned[k] = v.to_pydatetime()
        else:
            cleaned[k] = v
    return cleaned


def _upsert_rows(
    conn: Connection, table: Table, rows: list[dict[str, Any]], batch_size: int = 500
) -> None:
    if not rows:

        return
    pk_cols = [c.name for c in table.primary_key]
    non_pk_cols = [c.name for c in table.columns if c.name not in pk_cols]
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        stmt = pg_insert(table).values(chunk)
        if non_pk_cols:
            stmt = stmt.on_conflict_do_update(
                index_elements=pk_cols,
                set_={col: stmt.excluded[col] for col in non_pk_cols},
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=pk_cols)
        conn.execute(stmt)


def save_dataset_tables_to_db(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Persist flattened UK Government Bills and calendar tables into Neon Lakebase Postgres."""
    from westminster_weather.models import (
        BillRow,
        ParliamentSessionRow,
        ScheduledSittingRow,
        SittingDayRow,
        StageMapRow,
        StageSittingRow,
    )

    ensure_reference_tables_seeded_sync()
    engine = get_sync_direct_engine()
    counts: dict[str, int] = {}
    try:
        with engine.begin() as conn:
            Base.metadata.create_all(conn)

            if "stage_map" in tables and not tables["stage_map"].empty:
                records = [_clean_record(r) for r in tables["stage_map"].to_dict("records")]
                _upsert_rows(conn, StageMapRow.__table__, records)
                counts["stage_map"] = len(records)

            if "sessions" in tables and not tables["sessions"].empty:
                records = [_clean_record(r) for r in tables["sessions"].to_dict("records")]
                _upsert_rows(conn, ParliamentSessionRow.__table__, records)
                counts["sessions"] = len(records)

            if "sitting_days" in tables and not tables["sitting_days"].empty:
                records = [_clean_record(r) for r in tables["sitting_days"].to_dict("records")]
                _upsert_rows(conn, SittingDayRow.__table__, records)
                counts["sitting_days"] = len(records)

            if "bills" in tables and not tables["bills"].empty:
                records = [_clean_record(r) for r in tables["bills"].to_dict("records")]
                _upsert_rows(conn, BillRow.__table__, records)
                counts["bills"] = len(records)

            if "stage_sittings" in tables:
                records = [_clean_record(r) for r in tables["stage_sittings"].to_dict("records")]
                _upsert_rows(conn, StageSittingRow.__table__, records)
                counts["stage_sittings"] = len(records)

            if "scheduled_sittings" in tables:
                conn.execute(delete(ScheduledSittingRow))
                records = [
                    _clean_record(r) for r in tables["scheduled_sittings"].to_dict("records")
                ]
                _upsert_rows(conn, ScheduledSittingRow.__table__, records)
                counts["scheduled_sittings"] = len(records)
        return counts
    finally:
        engine.dispose()


def fetch_bills_df() -> pd.DataFrame:
    """DB adapter: read `bills` table (~588 UK Government Bills) from Neon Lakebase Postgres."""
    from westminster_weather.models import BillRow

    ensure_reference_tables_seeded_sync()
    engine = get_sync_pooled_engine()
    try:
        with Session(engine) as session:
            rows = session.execute(select(BillRow).order_by(BillRow.bill_id)).scalars().all()
            columns = [
                "bill_id",
                "short_title",
                "originating_house",
                "introduced_session_id",
                "introduced_date",
                "department",
                "kings_speech_flag",
                "size_clauses",
                "size_pages",
                "commons_committee_type",
                "carry_over_date",
                "outcome",
                "outcome_date",
                "stop_reason",
                "last_api_update",
                "ingested_at",
                "is_act",
                "is_defeated",
                "withdrawn_date",
                "included_session_ids",
            ]
            if not rows:
                return pd.DataFrame(columns=columns)
            df = pd.DataFrame(
                [
                    {
                        "bill_id": r.bill_id,
                        "short_title": r.short_title,
                        "originating_house": r.originating_house,
                        "introduced_session_id": r.introduced_session_id,
                        "introduced_date": r.introduced_date,
                        "department": r.department,
                        "kings_speech_flag": r.kings_speech_flag,
                        "size_clauses": r.size_clauses,
                        "size_pages": r.size_pages,
                        "commons_committee_type": r.commons_committee_type,
                        "carry_over_date": r.carry_over_date,
                        "outcome": r.outcome,
                        "outcome_date": r.outcome_date,
                        "stop_reason": r.stop_reason,
                        "last_api_update": pd.Timestamp(r.last_api_update)
                        if r.last_api_update
                        else pd.NaT,
                        "ingested_at": pd.Timestamp(r.ingested_at),
                        "is_act": r.is_act,
                        "is_defeated": r.is_defeated,
                        "withdrawn_date": r.withdrawn_date,
                        "included_session_ids": list(r.included_session_ids or []),
                    }
                    for r in rows
                ],
                columns=columns,
            )
            for col in ("kings_speech_flag", "size_clauses", "size_pages"):
                df[col] = df[col].astype("object")
            return df
    finally:
        engine.dispose()


def _fetch_sittings_table_df(model_cls: Any) -> pd.DataFrame:
    ensure_reference_tables_seeded_sync()
    engine = get_sync_pooled_engine()
    try:
        with Session(engine) as session:
            rows = (
                session.execute(
                    select(model_cls).order_by(
                        model_cls.bill_id, model_cls.event_date, model_cls.sort_order
                    )
                )
                .scalars()
                .all()
            )
            columns = [
                "sitting_id",
                "bill_id",
                "bill_type_id",
                "api_stage_id",
                "bill_stage_id",
                "house",
                "session_id",
                "sort_order",
                "event_date",
                "ingested_at",
            ]
            if not rows:
                return pd.DataFrame(columns=columns)
            return pd.DataFrame(
                [
                    {
                        "sitting_id": r.sitting_id,
                        "bill_id": r.bill_id,
                        "bill_type_id": r.bill_type_id,
                        "api_stage_id": r.api_stage_id,
                        "bill_stage_id": r.bill_stage_id,
                        "house": r.house,
                        "session_id": r.session_id,
                        "sort_order": r.sort_order,
                        "event_date": r.event_date,
                        "ingested_at": pd.Timestamp(r.ingested_at),
                    }
                    for r in rows
                ],
                columns=columns,
            )
    finally:
        engine.dispose()


def fetch_stage_sittings_df() -> pd.DataFrame:
    """DB adapter: read `stage_sittings` table from Neon Lakebase Postgres."""
    from westminster_weather.models import StageSittingRow

    return _fetch_sittings_table_df(StageSittingRow)


def fetch_scheduled_sittings_df() -> pd.DataFrame:
    """DB adapter: read `scheduled_sittings` table from Neon Lakebase Postgres."""
    from westminster_weather.models import ScheduledSittingRow

    return _fetch_sittings_table_df(ScheduledSittingRow)


def fetch_sessions_df() -> pd.DataFrame:
    """DB adapter: read `sessions` table from Neon Lakebase Postgres."""
    from westminster_weather.models import ParliamentSessionRow

    ensure_reference_tables_seeded_sync()
    engine = get_sync_pooled_engine()
    try:
        with Session(engine) as session:
            rows = (
                session.execute(
                    select(ParliamentSessionRow).order_by(ParliamentSessionRow.session_id)
                )
                .scalars()
                .all()
            )
            columns = ["session_id", "parliament_id", "start_date", "end_date", "end_type"]
            if not rows:
                return pd.DataFrame(columns=columns)
            return pd.DataFrame(
                [
                    {
                        "session_id": r.session_id,
                        "parliament_id": r.parliament_id,
                        "start_date": r.start_date,
                        "end_date": r.end_date,
                        "end_type": r.end_type,
                    }
                    for r in rows
                ],
                columns=columns,
            )
    finally:
        engine.dispose()


def fetch_sitting_days_df() -> pd.DataFrame:
    """DB adapter: read `sitting_days` table from Neon Lakebase Postgres."""
    from westminster_weather.models import SittingDayRow

    ensure_reference_tables_seeded_sync()
    engine = get_sync_pooled_engine()
    try:
        with Session(engine) as session:
            rows = (
                session.execute(
                    select(SittingDayRow).order_by(SittingDayRow.date, SittingDayRow.house)
                )
                .scalars()
                .all()
            )
            columns = ["date", "house"]
            if not rows:
                return pd.DataFrame(columns=columns)
            return pd.DataFrame(
                [{"date": r.date, "house": r.house} for r in rows],
                columns=columns,
            )
    finally:
        engine.dispose()


