"""SQLAlchemy ORM models for storing UK Government Bills, stages, and API evidence in Neon."""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
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

