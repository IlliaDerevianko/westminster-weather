"""Neon Lakebase Postgres engine, session management, and reference table DB adapters.

Follows neon-postgres skill best practices:
- Uses `DATABASE_URL` (pooled `-pooler` via PgBouncer) for application queries with
  `prepare_threshold=None` (disabling server-side prepared statements on transaction pooler).
- Uses `DATABASE_URL_UNPOOLED` (direct connection) for schema migrations / DDL.
"""

from datetime import date

import pandas as pd
from sqlalchemy import Connection, Engine, create_engine, select, text
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

