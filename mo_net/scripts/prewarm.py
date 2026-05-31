"""Pre-warm the resource cache for one or more URLs::

mo-net-prewarm hf://HuggingFaceFW/fineweb?config=sample-10BT
mo-net-prewarm s3://mo-net-resources/english-sentences.txt
"""

from __future__ import annotations

import sys

import click
from loguru import logger

from mo_net.log import LogLevel, setup_logging
from mo_net.resources import get_resource
from mo_net.settings import get_settings


@click.command(help=__doc__)
@click.argument("urls", nargs=-1, required=True)
def main(urls: tuple[str, ...]) -> None:
    setup_logging(LogLevel.INFO)
    logger.info(f"Resource cache: {get_settings().resource_cache}")
    failures: list[tuple[str, str]] = []
    for url in urls:
        logger.info(f"Warming {url}")
        try:
            ds = get_resource(url)
            logger.info(f"  ok — {len(ds):,} rows, columns: {ds.column_names}")
        except Exception as exc:  # noqa: BLE001 — best-effort across handlers
            logger.error(f"  failed: {exc}")
            failures.append((url, str(exc)))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
