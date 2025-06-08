from contextlib import contextmanager
from pathlib import Path
from typing import Final

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mo_net import ROOT_DIR

DB_PATH: Final[Path] = ROOT_DIR / "train.db"


class _SessionMaker:
    def __init__(self):
        self._session_maker = None

    def __call__(self):
        if self._session_maker is None:
            self._session_maker = sessionmaker(
                bind=create_engine(f"sqlite:///{DB_PATH}")
            )
        return self._session_maker()


_session_maker = _SessionMaker()


@contextmanager
def with_session():
    session = _session_maker()
    try:
        yield session
    finally:
        session.close()
