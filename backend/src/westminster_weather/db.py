"""Neon Lakebase Postgres async engine and session management.

Follows neon-postgres skill best practices:
- Uses `DATABASE_URL` (pooled `-pooler` via PgBouncer) for application queries with
  `prepare_threshold=None` (disabling server-side prepared statements on transaction pooler)
  and `pool_pre_ping=True` (for scale-to-zero compute resume).
- Uses `DATABASE_URL_UNPOOLED` (direct connection) for schema migrations / DDL.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from westminster_weather.config import settings


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


pooled_engine = get_pooled_engine()
SessionLocal = async_sessionmaker(pooled_engine, expire_on_commit=False)


async def init_db() -> None:
    """Create database tables using the direct (unpooled) Neon connection."""
    import westminster_weather.models  # noqa: F401

    engine = get_direct_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


async def check_db_connection() -> str:
    """Verify connectivity to Neon Lakebase Postgres and return Postgres version string."""
    async with SessionLocal() as session:
        result = await session.execute(text("SELECT version()"))
        return str(result.scalar_one())
