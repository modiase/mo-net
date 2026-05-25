from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mo_net.settings import get_settings


def __getattr__(name: str):
    # Back-compat: defer to the live setting so existing imports keep working
    # even if the runtime value changes after CLI parsing.
    if name == "DB_PATH":
        return get_settings().resolved_db_path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _SessionMaker:
    def __init__(self):
        self._session_maker = None

    def __call__(self):
        if self._session_maker is None:
            db_path = get_settings().resolved_db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._session_maker = sessionmaker(
                bind=create_engine(f"sqlite:///{db_path}")
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
