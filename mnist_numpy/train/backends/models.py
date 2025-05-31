from __future__ import annotations

from datetime import datetime
from typing import Self

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase

from mnist_numpy.db import DB_PATH


class Base(DeclarativeBase):
    pass


class Iteration(Base):
    __tablename__ = "iterations"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    batch_loss = Column(Float, nullable=False)
    batch = Column(Integer, nullable=False)
    epoch = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    val_loss = Column(Float, nullable=False)


class DbRun(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)

    completed_at = Column(DateTime, nullable=True)
    current_batch = Column(Integer, nullable=False)
    current_batch_loss = Column(Float, nullable=False)
    current_epoch = Column(Integer, nullable=False)
    current_learning_rate = Column(Float, nullable=False)
    current_val_loss = Column(Float, nullable=False)
    current_timestamp = Column(DateTime, nullable=False)
    name = Column(String, nullable=False)
    seed = Column(Integer, nullable=False)
    started_at = Column(DateTime, nullable=False)
    total_batches = Column(Integer, nullable=False)
    total_epochs = Column(Integer, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    @property
    def is_completed(self) -> bool:
        return self.current_epoch >= self.total_epochs

    @property
    def progress(self) -> float:
        return self.current_batch / self.total_batches

    @classmethod
    def unfinished_runs(cls, session) -> list[Self]:
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
        seed: int,
        total_batches: int,
        total_epochs: int,
        started_at: datetime.datetime | None = None,
    ) -> Self:
        return cls(
            name=str(seed),
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
