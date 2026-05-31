"""Cache-backed resource resolution with a pluggable handler registry.

Every supported URL scheme resolves to a :class:`datasets.Dataset`.
Adding a new scheme: register a callable taking the parsed URL and
returning a Dataset::

    @register("gcs")
    def _fetch_gcs(parsed: ParseResult) -> datasets.Dataset: ...

Use :func:`head_resource` to verify reachability without paying the
download cost — important when validating a CLI argument before training
starts.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import ParseResult, parse_qs, urlparse

import requests
from loguru import logger

from mo_net.settings import get_settings

if TYPE_CHECKING:
    import datasets

type ResourceHandler = Callable[[ParseResult], "datasets.Dataset"]

_HANDLERS: dict[str, ResourceHandler] = {}


def register(*schemes: str) -> Callable[[ResourceHandler], ResourceHandler]:
    """Register ``handler`` for one or more URL schemes.

    Re-registering an existing scheme silently overrides the previous
    handler.
    """

    def decorator(handler: ResourceHandler) -> ResourceHandler:
        for scheme in schemes:
            _HANDLERS[scheme] = handler
        return handler

    return decorator


def get_resource(url: str) -> "datasets.Dataset":
    """Resolve ``url`` to a :class:`datasets.Dataset` via the registered handler."""
    parsed = urlparse(url)
    handler = _HANDLERS.get(parsed.scheme)
    if handler is None:
        raise ValueError(
            f"Unsupported URL scheme {parsed.scheme!r}; registered: {sorted(_HANDLERS)}"
        )
    return handler(parsed)


def head_resource(url: str) -> None:
    """Verify ``url`` is reachable without materialising its payload.

    For pre-flight CLI validation: :func:`get_resource` on an ``hf://``
    URL would download gigabytes just to check reachability.
    """
    parsed = urlparse(url)
    match parsed.scheme:
        case "http" | "https":
            requests.head(url, allow_redirects=True).raise_for_status()
        case "s3":
            requests.head(_s3_to_https(parsed), allow_redirects=True).raise_for_status()
        case "file":
            path = Path(parsed.path).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
        case "hf":
            from huggingface_hub import dataset_info

            dataset_info(_hf_repo_id(parsed))
        case other:
            raise ValueError(f"Unsupported URL scheme {other!r}")


def _s3_to_https(parsed: ParseResult) -> str:
    return f"https://{parsed.netloc}.s3.amazonaws.com/{parsed.path.lstrip('/')}"


def _hf_repo_id(parsed: ParseResult) -> str:
    return f"{parsed.netloc}/{parsed.path.lstrip('/')}"


def _extract_slug(url: str) -> str:
    """Filesystem-safe slug of the URL's final path component (no suffix)."""
    filename = Path(urlparse(url).path).stem
    return re.sub(r"[^a-z0-9]+", "-", filename.lower()).strip("-") or "resource"


def _extract_suffix(url: str) -> str:
    """Lowercased file suffix from a URL."""
    return Path(urlparse(url).path).suffix.lower()


_KNOWN_SUFFIXES: Final[frozenset[str]] = frozenset({".csv", ".txt", ".text"})


def _find_cached_by_hash(content_hash: str) -> Path | None:
    """Return the cached entry for ``content_hash`` if any.

    Skips entries whose suffix isn't in :data:`_KNOWN_SUFFIXES` —
    :func:`_wrap_path_as_dataset` couldn't dispatch on them anyway.
    """
    cache = get_settings().resource_cache
    if not cache.exists():
        return None
    return next(
        (
            p
            for p in cache.iterdir()
            if p.name.startswith(content_hash) and p.suffix.lower() in _KNOWN_SUFFIXES
        ),
        None,
    )


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive cross-process file lock on ``lock_path``. Host-local."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _download_with_etag(download_url: str) -> Path:
    """Fetch ``download_url`` into the resource cache, keyed by ETag.

    SHA-256 of the body is the fallback key when the server omits an
    ETag. Concurrent callers serialise on a per-resource lock; writes
    are atomic via ``.tmp`` + ``os.replace``.
    """
    cache = get_settings().resource_cache
    cache.mkdir(parents=True, exist_ok=True)

    logger.info(f"resolving {download_url}")
    head = requests.head(download_url, allow_redirects=True)
    head.raise_for_status()
    etag = head.headers.get("ETag", "").strip('"')
    if etag and (existing := _find_cached_by_hash(etag)):
        logger.info(f"cache hit: {existing}")
        return existing

    lock_key = etag or hashlib.sha256(download_url.encode()).hexdigest()[:32]
    with _file_lock(cache / f".{lock_key}.lock"):
        if etag and (existing := _find_cached_by_hash(etag)):
            logger.info(f"cache hit (after waiting on peer): {existing}")
            return existing

        logger.info(f"downloading {download_url}")
        response = requests.get(download_url)
        response.raise_for_status()
        body = response.content
        content_hash = (
            etag
            or response.headers.get("ETag", "").strip('"')
            or hashlib.sha256(body).hexdigest()[:32]
        )
        cache_path = (
            cache
            / f"{content_hash}-{_extract_slug(download_url)}{_extract_suffix(download_url)}"
        )
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_bytes(body)
        os.replace(tmp_path, cache_path)
        logger.info(f"cached {len(body):,} bytes → {cache_path}")
    return cache_path


def _wrap_path_as_dataset(path: Path) -> "datasets.Dataset":
    """Materialise a local file as a :class:`datasets.Dataset` by suffix.

    ``.csv`` → ``from_csv`` (columns from header). ``.txt`` / ``.text``
    → ``from_text`` (one row per line, column named ``text``).
    """
    import datasets

    suffix = path.suffix.lower()
    match suffix:
        case ".csv":
            return datasets.Dataset.from_csv(str(path))
        case ".txt" | ".text":
            return datasets.Dataset.from_text(str(path))
        case _:
            raise ValueError(
                f"Cannot wrap {suffix!r} as a Dataset (supported: .csv, .txt, .text)"
            )


@register("http", "https")
def _fetch_http(parsed: ParseResult) -> "datasets.Dataset":
    return _wrap_path_as_dataset(_download_with_etag(parsed.geturl()))


@register("s3")
def _fetch_s3(parsed: ParseResult) -> "datasets.Dataset":
    return _wrap_path_as_dataset(_download_with_etag(_s3_to_https(parsed)))


@register("file")
def _resolve_file(parsed: ParseResult) -> "datasets.Dataset":
    path = Path(parsed.path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    logger.info(f"resolving file://{path}")
    return _wrap_path_as_dataset(path)


@register("hf")
def _fetch_hf(parsed: ParseResult) -> "datasets.Dataset":
    """Load a Hugging Face dataset into the resource cache.

    URL: ``hf://<owner>/<name>?config=<cfg>&split=<split>&text_field=<col>``.
    Defaults: ``split=train``, ``text_field=text``, no config. The
    text-bearing column is renamed to ``text`` if not already.
    """
    import datasets

    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    repo_id = _hf_repo_id(parsed)
    config = params.get("config")
    split = params.get("split", "train")
    text_field = params.get("text_field", "text")

    cache_dir = get_settings().resource_cache / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"loading hf://{repo_id} (config={config!r}, split={split!r}, "
        f"cache_dir={cache_dir})"
    )
    ds = datasets.load_dataset(repo_id, config, split=split, cache_dir=str(cache_dir))
    if text_field != "text" and text_field in ds.column_names:
        ds = ds.rename_column(text_field, "text")
    return ds
