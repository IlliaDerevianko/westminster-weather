"""Take-up model tables (DATA_CONTRACT §10), as SQLAlchemy Core tables on their own MetaData.

Safety: these tables are deliberately NOT registered on ``westminster_weather.db.Base``. The
app runs ``Base.metadata.create_all`` on start-up against the production database; the
take-up tables must only ever be created by ``takeup.db.migrate()`` on the take-up database
(``TAKEUP_DATABASE_URL``, Neon branch ``takeup-dev``).

Foreign keys between take-up tables are declared. ``idea_bill_links.bill_id`` refers to the
production ``bills`` table, which lives on ``Base.metadata``; it is NOT declared as a
database-level foreign key (another MetaData cannot resolve it, and ``bills`` is maintained by
the passage-model pipeline). The check is done in Python instead (DATA_CONTRACT §10.10).
``idea_mentions.mention_id`` points at either ``signal_mentions`` or ``kings_speech_items``,
so it has no foreign key either.

The contract is a draft awaiting approval: each table is one ``Table(...)`` call below, so a
column change is a one-line edit here plus a re-run of ``migrate()`` on the branch.
JSON values are stored as TEXT (standard SQL; compatible with SQLite and PostgreSQL).
"""

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
)

takeup_metadata = MetaData()


def _ingested_at() -> Column:
    return Column("ingested_at", DateTime(timezone=True), nullable=False)


# §10.1 One row per bill announced in a King's/Queen's Speech briefing note.
kings_speech_items = Table(
    "kings_speech_items",
    takeup_metadata,
    Column("item_id", Text, primary_key=True),  # ks-<speech_date>-<nn>
    Column("speech_date", Date, nullable=False),
    Column("announced_name", Text, nullable=False),
    Column("is_draft", Boolean, nullable=False),
    Column("is_carried_over", Boolean, nullable=False),
    Column("department", Text, nullable=True),
    Column("source_url", Text, nullable=False),
    _ingested_at(),
)

# §10.2 One row per sentence in which the government commits (or seems to) to legislate.
signal_mentions = Table(
    "signal_mentions",
    takeup_metadata,
    Column("mention_id", Text, primary_key=True),  # <source>-<source_id>-<sentence index>
    Column("source", Text, nullable=False),  # wms | hansard_spoken | hansard_written
    Column("source_id", Text, nullable=False),
    Column("event_date", Date, nullable=False),
    Column("member_id", Integer, nullable=True),
    Column("is_minister", Boolean, nullable=True),  # NULL = unknown
    Column("department", Text, nullable=True),
    Column("title", Text, nullable=False),
    Column("sentence", Text, nullable=False),
    Column("snippet", Text, nullable=False),
    Column("phrase", Text, nullable=False),
    Column("firmness", Integer, nullable=False),  # 0 vague, 1 medium, 2 firm
    Column("is_commitment", Boolean, nullable=False),
    Column("topic", Text, nullable=False),
    Column("labelled_by", Text, nullable=False),  # rules | gemini | hand
    _ingested_at(),
)

# §10.3 One commitment to legislate on one topic.
ideas = Table(
    "ideas",
    takeup_metadata,
    Column("idea_id", Text, primary_key=True),  # A-<item_id> or B-<nnnnn>
    Column("tier", Text, nullable=False),  # A | B
    Column("topic", Text, nullable=False),
    Column("trigger_date", Date, nullable=False),
    Column("department", Text, nullable=True),
    Column("first_source", Text, nullable=False),
    Column("initial_firmness", Integer, nullable=False),
    Column("n_mentions", Integer, nullable=False),
    _ingested_at(),
)

# §10.4 Mentions of each idea; group_id = topic group before same-bill merges.
idea_mentions = Table(
    "idea_mentions",
    takeup_metadata,
    Column("idea_id", Text, ForeignKey("ideas.idea_id"), primary_key=True),
    Column("mention_id", Text, primary_key=True),  # signal_mentions or kings_speech_items id
    Column("group_id", Text, nullable=False),
)

# §10.5 Time-varying events for an idea (the model's x features).
idea_events = Table(
    "idea_events",
    takeup_metadata,
    Column("idea_id", Text, ForeignKey("ideas.idea_id"), primary_key=True),
    Column("event_type", Text, primary_key=True),
    Column("event_date", Date, primary_key=True),
    Column("title", Text, nullable=False),
    Column("source_url", Text, nullable=False),
    _ingested_at(),
)

# §10.6 One link per idea. bill_id -> production bills: NOT a database-level FK (see above).
idea_bill_links = Table(
    "idea_bill_links",
    takeup_metadata,
    Column("idea_id", Text, ForeignKey("ideas.idea_id"), primary_key=True),
    Column("bill_id", Integer, nullable=True),  # NULL = checked, no matching bill
    Column("similarity", Float, nullable=True),
    Column("decided_by", Text, nullable=False),  # auto | gemini | rule | hand
    Column("checked", Boolean, nullable=False),
    _ingested_at(),
)

# §10.7 One row per model fit.
takeup_runs = Table(
    "takeup_runs",
    takeup_metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("code_version", Text, nullable=False),
    Column("train_cut", Date, nullable=False),
    Column("features", Text, nullable=False),  # comma-separated z_* and x_* names
    Column("parameters", Text, nullable=False),  # JSON as TEXT
    Column("metrics", Text, nullable=False),  # JSON as TEXT
)

# §10.8 One forecast per (run, signal, as_of). Keyed on signal_id, not idea_id: before the
# bill's first reading an idea's topic groups are separate signals ('<idea_id>:<group_id>'),
# so one idea can have several forecasts at the same as_of. (Needs §10.8 updated to match.)
takeup_forecasts = Table(
    "takeup_forecasts",
    takeup_metadata,
    Column("run_id", Integer, ForeignKey("takeup_runs.run_id"), primary_key=True),
    Column("signal_id", Text, primary_key=True),  # get_takeup_signals signal_id
    Column("as_of", Date, primary_key=True),
    Column("idea_id", Text, ForeignKey("ideas.idea_id"), nullable=False),
    Column("pi", Float, nullable=False),
    Column("pi_lo", Float, nullable=False),
    Column("pi_hi", Float, nullable=False),
    Column("verdict", Text, nullable=False),  # likely | unlikely | uncertain
    Column("weeks_elapsed", Integer, nullable=False),
    Column("pi_now", Float, nullable=False),
    Column("p_bill_26w", Float, nullable=False),
    Column("p_bill_52w", Float, nullable=False),
    Column("t_p10", Integer, nullable=True),
    Column("t_p50", Integer, nullable=True),
    Column("t_p90", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# The six tables get_takeup_signals reads (runs and forecasts are model outputs).
DATA_TABLES: tuple[Table, ...] = (
    kings_speech_items,
    signal_mentions,
    ideas,
    idea_mentions,
    idea_events,
    idea_bill_links,
)
TABLES: dict[str, Table] = dict(takeup_metadata.tables)
