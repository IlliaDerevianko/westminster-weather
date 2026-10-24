"""Bill dataset tables, as defined in DATA_CONTRACT.md sections 2.5, 3.1 and 3.2.

Standard SQL types only, so the same tables work on SQLite and PostgreSQL.
"""

from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Connection,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

metadata = MetaData()

stage_map = Table(
    "stage_map",
    metadata,
    Column("api_stage_id", Integer, primary_key=True, autoincrement=False),
    Column("api_stage_name", Text, nullable=False),
    Column("house", Text),
    Column("model_state", Text),
)

bills = Table(
    "bills",
    metadata,
    Column("bill_id", Integer, primary_key=True, autoincrement=False),
    Column("short_title", Text, nullable=False),
    Column("originating_house", Text, nullable=False),
    Column("introduced_session_id", Integer, nullable=False),
    Column("introduced_date", Date, nullable=False),
    Column("department", Text),
    Column("kings_speech_flag", Boolean),
    Column("size_clauses", Integer),
    Column("size_pages", Integer),
    Column("commons_committee_type", Text),
    Column("carry_over_date", Date),
    Column("outcome", Text, nullable=False),
    Column("outcome_date", Date),
    Column("last_api_update", DateTime),
    Column("ingested_at", DateTime, nullable=False),
)

stage_sittings = Table(
    "stage_sittings",
    metadata,
    Column("sitting_id", Integer, primary_key=True, autoincrement=False),
    Column("bill_id", Integer, ForeignKey("bills.bill_id"), nullable=False),
    Column("api_stage_id", Integer, ForeignKey("stage_map.api_stage_id"), nullable=False),
    # NULL for stages the API files under neither House (Royal Assent).
    Column("house", Text),
    Column("session_id", Integer, nullable=False),
    Column("event_date", Date, nullable=False),
    Column("ingested_at", DateTime, nullable=False),
)


def upsert(conn: Connection, table: Table, rows: list[dict[str, Any]]) -> None:
    """Insert rows, updating any that already exist by primary key (idempotent re-runs)."""
    if not rows:
        return
    insert = sqlite_insert if conn.dialect.name == "sqlite" else postgresql_insert
    stmt = insert(table)
    keys = [c.name for c in table.primary_key]
    stmt = stmt.on_conflict_do_update(
        index_elements=keys,
        set_={c.name: stmt.excluded[c.name] for c in table.columns if c.name not in keys},
    )
    conn.execute(stmt, rows)
