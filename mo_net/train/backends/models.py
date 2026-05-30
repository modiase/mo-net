from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import Self

from loguru import logger
from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from mo_net.settings import get_settings


class Base(DeclarativeBase):
    pass


class DbRun(Base):
    """One training segment.

    ``lineage_id`` groups every segment of a resume chain so cross-run
    queries don't walk ``parent_run_id`` pointers; ``parent_run_id`` keeps
    the direct predecessor for tree navigation.
    """

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    seed: Mapped[int] = mapped_column(BigInteger)
    lineage_id: Mapped[str] = mapped_column(String)
    parent_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("runs.id"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime)
    last_iteration_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    total_batches: Mapped[int] = mapped_column(Integer)
    total_epochs: Mapped[int] = mapped_column(Integer)
    build_rev: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        Index("ix_runs_lineage_id", "lineage_id"),
        Index("ix_runs_parent_run_id", "parent_run_id"),
        Index("ix_runs_name", "name"),
        Index("ix_runs_started_at", "started_at"),
    )

    @classmethod
    def unfinished_runs(cls, session: Session) -> Collection[Self]:
        return (
            session.query(cls)
            .filter(cls.completed_at.is_(None))
            .order_by(cls.started_at.desc())
            .all()
        )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        seed: int,
        lineage_id: str,
        parent_run_id: int | None,
        total_batches: int,
        total_epochs: int,
        build_rev: str | None,
        started_at: datetime | None = None,
    ) -> Self:
        ts = started_at if started_at is not None else datetime.now()
        return cls(
            name=name,
            seed=seed,
            lineage_id=lineage_id,
            parent_run_id=parent_run_id,
            started_at=ts,
            last_iteration_at=None,
            total_batches=total_batches,
            total_epochs=total_epochs,
            build_rev=build_rev,
        )


class Iteration(Base):
    """One logged training event.

    Natural-keyed by ``(run_id, step)``. ``lineage_id`` is denormalised
    from ``runs`` so chain-wide trajectory scans skip the join.
    """

    __tablename__ = "iterations"

    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True
    )
    step: Mapped[int] = mapped_column(Integer, primary_key=True)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lineage_id: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime)

    __table_args__ = (
        Index("ix_iterations_run_id_epoch", "run_id", "epoch"),
        Index("ix_iterations_lineage_id_step", "lineage_id", "step"),
        Index("ix_iterations_timestamp", "timestamp"),
    )


class Metric(Base):
    """One scalar value tagged to an iteration.

    EAV row layout: adding a metric is a row write, not a schema change.
    """

    __tablename__ = "metrics"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    step: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "step"],
            ["iterations.run_id", "iterations.step"],
            ondelete="CASCADE",
        ),
        Index("ix_metrics_run_id_name_step", "run_id", "name", "step"),
        # Covers "rank runs by metric" as an index-only scan.
        Index("ix_metrics_name_run_id_value", "name", "run_id", "value"),
    )


_VIEW_BODY = """
WITH ranked AS (
    SELECT
        run_id,
        name,
        step,
        value,
        ROW_NUMBER() OVER (PARTITION BY run_id, name ORDER BY step DESC)  AS r_latest,
        ROW_NUMBER() OVER (PARTITION BY run_id, name ORDER BY value ASC)  AS r_min,
        ROW_NUMBER() OVER (PARTITION BY run_id, name ORDER BY value DESC) AS r_max
    FROM metrics
)
SELECT
    run_id,
    name,
    MIN(value)                                  AS min_value,
    MAX(value)                                  AS max_value,
    MAX(CASE WHEN r_latest = 1 THEN value END)  AS latest_value,
    MAX(CASE WHEN r_latest = 1 THEN step  END)  AS latest_step,
    MAX(CASE WHEN r_min    = 1 THEN step  END)  AS min_value_step,
    MAX(CASE WHEN r_max    = 1 THEN step  END)  AS max_value_step,
    COUNT(*)                                    AS n_samples
FROM ranked
GROUP BY run_id, name
"""


def install_schema(engine: Engine) -> None:
    """Create tables, indexes, and the run_metrics_summary view. Idempotent."""
    Base.metadata.create_all(engine)
    # sqlite supports `IF NOT EXISTS` on views; postgres/mysql support
    # `OR REPLACE`; neither supports both.
    dialect = engine.dialect.name
    if dialect == "sqlite":
        create_view = f"CREATE VIEW IF NOT EXISTS run_metrics_summary AS {_VIEW_BODY}"
    else:
        create_view = f"CREATE OR REPLACE VIEW run_metrics_summary AS {_VIEW_BODY}"
    with engine.connect() as conn:
        conn.execute(text(create_view))
        conn.commit()


if __name__ == "__main__":
    db_path = get_settings().resolved_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    logger.info(f"Creating database at {db_path}.")
    install_schema(engine)
