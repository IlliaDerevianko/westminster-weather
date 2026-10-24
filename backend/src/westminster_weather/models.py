"""SQLAlchemy ORM models for storing UK Government Bills, stages, and API evidence in Neon."""

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from westminster_weather.db import Base


class TrackedBillRow(Base):
    __tablename__ = "tracked_bills"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    parliament_bill_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    short_title: Mapped[str] = mapped_column(String(255), nullable=False)
    long_title: Mapped[str] = mapped_column(Text, nullable=False)
    sponsoring_department: Mapped[str] = mapped_column(String(255), nullable=False)
    earliest_signal_date: Mapped[date] = mapped_column(Date, nullable=False)
    first_reading_date: Mapped[date] = mapped_column(Date, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    stage_records: Mapped[list["StageRecordRow"]] = relationship(
        back_populates="bill",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    evidence_events: Mapped[list["ApiEvidenceEventRow"]] = relationship(
        back_populates="bill",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class StageRecordRow(Base):
    __tablename__ = "bill_stage_records"
    __table_args__ = (UniqueConstraint("bill_id", "stage", name="uq_bill_stage"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bill_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("tracked_bills.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    entered_date: Mapped[date] = mapped_column(Date, nullable=False)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    source_api_url: Mapped[str] = mapped_column(String(500), nullable=False)
    web_url: Mapped[str] = mapped_column(String(500), nullable=False)

    bill: Mapped["TrackedBillRow"] = relationship(back_populates="stage_records")


class ApiEvidenceEventRow(Base):
    __tablename__ = "api_evidence_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bill_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("tracked_bills.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False)
    source_api_url: Mapped[str] = mapped_column(String(500), nullable=False)
    web_url: Mapped[str] = mapped_column(String(500), nullable=False)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False)

    bill: Mapped["TrackedBillRow"] = relationship(back_populates="evidence_events")


class ParliamentRow(Base):
    """UK Parliament reference table (first meeting, dissolution, and election dates)."""

    __tablename__ = "parliaments"

    parliament_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_meeting_date: Mapped[date] = mapped_column(Date, nullable=False)
    dissolution_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    election_date: Mapped[date | None] = mapped_column(Date, nullable=True)


class SessionOverrideRow(Base):
    """Manual overrides for UK Parliamentary session boundaries and end types."""

    __tablename__ = "session_overrides"

    session_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class StageMapRow(Base):
    """Maps UK Parliament Bills API stage IDs to model states (DATA_CONTRACT §2.5)."""

    __tablename__ = "stage_map"

    api_stage_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    api_stage_name: Mapped[str] = mapped_column(Text, nullable=False)
    house: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_state: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ParliamentSessionRow(Base):
    """UK Parliamentary sessions table (DATA_CONTRACT §2.1)."""

    __tablename__ = "sessions"

    session_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    parliament_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_type: Mapped[str | None] = mapped_column(String(32), nullable=True)


class SittingDayRow(Base):
    """Sitting days per House (DATA_CONTRACT §2.4)."""

    __tablename__ = "sitting_days"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    house: Mapped[str] = mapped_column(String(32), primary_key=True)


class BillRow(Base):
    """UK Government Bills dataset table (DATA_CONTRACT §3.1)."""

    __tablename__ = "bills"

    bill_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    short_title: Mapped[str] = mapped_column(Text, nullable=False)
    originating_house: Mapped[str] = mapped_column(String(32), nullable=False)
    introduced_session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    introduced_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    department: Mapped[str | None] = mapped_column(Text, nullable=True)
    kings_speech_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    size_clauses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    commons_committee_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    carry_over_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    outcome_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_api_update: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_act: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_defeated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    withdrawn_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    included_session_ids: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)


class StageSittingRow(Base):
    """Past stage sittings for UK Government Bills (DATA_CONTRACT §3.2)."""

    __tablename__ = "stage_sittings"

    sitting_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    bill_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    bill_type_id: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    api_stage_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    bill_stage_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    house: Mapped[str | None] = mapped_column(String(32), nullable=True)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    sort_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    event_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ScheduledSittingRow(Base):
    """Future-dated scheduled stage sittings for live forecasts (DATA_CONTRACT §3.2a)."""

    __tablename__ = "scheduled_sittings"

    sitting_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    bill_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    bill_type_id: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    api_stage_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    bill_stage_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    house: Mapped[str | None] = mapped_column(String(32), nullable=True)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    sort_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    event_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ActCommencementRow(Base):
    """Phase 3 (After Royal Assent) summary & transition dates per UK Government Act."""

    __tablename__ = "act_commencements"

    bill_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    ukpga_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    ukpga_number: Mapped[int] = mapped_column(Integer, nullable=False)
    ukpga_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    act_title: Mapped[str] = mapped_column(Text, nullable=False)
    isbn: Mapped[str | None] = mapped_column(String(64), nullable=True)
    royal_assent_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    some_sections_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    first_commencement_order_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    latest_commencement_order_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    commencement_orders_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_draft_regulation_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    latest_draft_regulation_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    draft_regulations_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_laid_regulation_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    latest_laid_regulation_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    laid_regulations_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    law_fully_in_force_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_fully_in_force: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    first_amendment_or_repeal_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    latest_amendment_or_repeal_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    amendment_and_repeal_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_repealed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PostRoyalAssentEventRow(Base):
    """Phase 3 (After Royal Assent) events: Commencement Orders, Draft SIs, Laid SIs, Amendments."""

    __tablename__ = "post_royal_assent_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bill_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False)
    citation: Mapped[str | None] = mapped_column(String(128), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False)
    source_api_url: Mapped[str] = mapped_column(String(500), nullable=False)
    web_url: Mapped[str] = mapped_column(String(500), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)



