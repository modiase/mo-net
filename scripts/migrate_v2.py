"""One-shot migration to the v2 logging schema.

Renames the legacy ``runs`` and ``iterations`` tables to ``runs_v1`` /
``iterations_v1`` (preserving the data for one-off historical queries) and
then creates the new v2 schema + ``run_metrics_summary`` view via
:func:`mo_net.train.backends.models.install_schema`.

Usage::

    # Postgres (herakles):
    MO_NET_DB_URL='postgresql+psycopg://mo_net@localhost:5432/mo_net' \
      nix develop -c python scripts/migrate_v2.py

    # SQLite (local dev):
    MO_NET_DB_URL='sqlite:////path/to/train.db' \
      nix develop -c python scripts/migrate_v2.py

Idempotent — running twice is a no-op (the rename is skipped if the
``_v1`` table already exists).
"""

from __future__ import annotations

import os
import sys

from loguru import logger
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

from mo_net.settings import get_settings
from mo_net.train.backends.models import install_schema


def resolve_db_url() -> str:
    url = os.environ.get("MO_NET_DB_URL")
    if url:
        return url
    db_path = get_settings().resolved_db_path
    if not db_path.exists():
        sys.exit(
            f"No MO_NET_DB_URL set and default sqlite path doesn't exist: {db_path}"
        )
    return f"sqlite:///{db_path}"


def drop_old_indexes(engine: Engine, table: str) -> None:
    """Drop indexes attached to ``table`` so the v2 ``create_all`` can
    recreate same-named indexes on the new tables.

    Postgres renames indexes automatically when their table is renamed;
    sqlite leaves them pointing at the old (renamed) table. Either way,
    dropping them on the old name avoids name collisions on the new
    table's index creation.
    """
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return
    for idx in inspector.get_indexes(table):
        name = idx.get("name")
        if not name or name.startswith("sqlite_"):
            continue
        with engine.begin() as conn:
            try:
                conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
                logger.info(f"  dropped index {name}")
            except Exception as exc:
                logger.warning(f"  drop index {name} failed: {exc}")


def rename_if_present(engine: Engine, old: str, new: str) -> bool:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if old not in tables:
        logger.info(f"  {old}: not present (nothing to rename)")
        return False
    if new in tables:
        logger.info(f"  {old}: already renamed to {new} (idempotent skip)")
        return False
    drop_old_indexes(engine, old)
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {old} RENAME TO {new}"))
    logger.info(f"  {old} → {new}")
    return True


def row_count(engine: Engine, table: str) -> int | None:
    try:
        with engine.connect() as conn:
            return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    except Exception as exc:
        logger.warning(f"  {table}: count failed ({exc})")
        return None


def main() -> None:
    url = resolve_db_url()
    logger.info(f"Connecting to {url}")
    engine = create_engine(url)

    logger.info("Renaming legacy tables")
    rename_if_present(engine, "iterations", "iterations_v1")
    rename_if_present(engine, "runs", "runs_v1")

    logger.info("Installing v2 schema (tables, indexes, run_metrics_summary view)")
    try:
        install_schema(engine)
    except (IntegrityError, OperationalError, ProgrammingError) as exc:
        if "already exists" not in str(exc).lower():
            raise
        logger.info("  schema already present (idempotent skip)")

    logger.info("Verification:")
    for table in ("runs", "iterations", "metrics", "runs_v1", "iterations_v1"):
        count = row_count(engine, table)
        if count is not None:
            logger.info(f"  {table}: {count} rows")

    logger.success(
        "Migration complete. New runs write into runs/iterations/metrics; "
        "_v1 tables preserved for manual inspection. Drop them when "
        "you're confident."
    )


if __name__ == "__main__":
    main()
