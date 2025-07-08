from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from decimal import Decimal
from typing import Self

from loguru import logger
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from mo_net.db import DB_PATH


class Base(DeclarativeBase):
    pass


class Iteration(Base):
    __tablename__ = "iterations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id"))
    batch_loss: Mapped[float] = mapped_column(Float)
    batch: Mapped[int] = mapped_column(Integer)
    epoch: Mapped[int] = mapped_column(Integer)
    learning_rate: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    val_loss: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        Index("ix_iterations_run_id", "run_id"),
        Index("ix_iterations_epoch", "epoch"),
    )


class DbRun(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    current_batch: Mapped[int] = mapped_column(Integer)
    current_batch_loss: Mapped[float] = mapped_column(Float)
    current_epoch: Mapped[int] = mapped_column(Integer)
    current_learning_rate: Mapped[float] = mapped_column(Float)
    current_val_loss: Mapped[float] = mapped_column(Float)
    current_timestamp: Mapped[datetime] = mapped_column(DateTime)
    name: Mapped[str] = mapped_column(String)
    seed: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    total_batches: Mapped[int] = mapped_column(Integer)
    total_epochs: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    @property
    def is_completed(self) -> bool:
        return self.current_epoch >= self.total_epochs

    @property
    def progress(self) -> float | Decimal:
        return self.current_batch / self.total_batches

    @classmethod
    def unfinished_runs(cls, session: Session) -> Collection[Self]:
        return (
            session.query(cls)
            .filter(cls.completed_at.is_(None))
            .order_by(cls.updated_at.desc())
            .all()
        )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        seed: int,
        total_batches: int,
        total_epochs: int,
        started_at: datetime | None = None,
    ) -> Self:
        return cls(
            name=name,
            seed=seed,
            started_at=(
                _started_at := (
                    started_at if started_at is not None else datetime.now()
                )
            ),
            updated_at=_started_at,
            current_batch=0,
            current_batch_loss=0.0,
            current_epoch=0,
            current_learning_rate=0.0,
            current_val_loss=0.0,
            current_timestamp=datetime.now(),
            total_batches=total_batches,
            total_epochs=total_epochs,
        )


if __name__ == "__main__":
    engine = create_engine(f"sqlite:///{DB_PATH}")
    logger.info(f"Creating database at {DB_PATH}.")
    Base.metadata.create_all(engine)
