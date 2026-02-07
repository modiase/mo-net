import re
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import requests

from mo_net import PROJECT_ROOT_DIR

RESOURCE_CACHE: Final[Path] = PROJECT_ROOT_DIR / ".resource_cache"

MNIST_TRAIN_URL: Final[str] = "s3://mo-net-resources/mnist_train.csv"
MNIST_TEST_URL: Final[str] = "s3://mo-net-resources/mnist_test.csv"


def _extract_slug(url: str) -> str:
    """Extract a slug from the URL's final path component, stripping suffixes."""
    parsed = urlparse(url)
    filename = Path(parsed.path).stem
    slug = re.sub(r"[^a-z0-9]+", "-", filename.lower()).strip("-")
    return slug or "resource"


def _find_cached_by_hash(content_hash: str) -> Path | None:
    """Find a cached file by its content hash prefix."""
    if not RESOURCE_CACHE.exists():
        return None
    for path in RESOURCE_CACHE.iterdir():
        if path.name.startswith(content_hash):
            return path
    return None


def get_resource(url: str) -> Path:
    parsed = urlparse(url)

    match parsed.scheme:
        case "http" | "https":
            download_url = url
        case "s3":
            download_url = (
                f"https://{parsed.netloc}.s3.amazonaws.com/{parsed.path.lstrip('/')}"
            )
        case "file":
            path = Path(parsed.path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return path
        case _:
            raise ValueError(f"Unsupported protocol: {parsed.scheme}")

    RESOURCE_CACHE.mkdir(exist_ok=True)

    # HEAD request to get ETag (content hash) without downloading
    head_response = requests.head(download_url)
    head_response.raise_for_status()
    etag = head_response.headers.get("ETag", "").strip('"')

    if etag and (existing := _find_cached_by_hash(etag)):
        return existing

    # Download if not cached
    response = requests.get(download_url)
    response.raise_for_status()

    # Use ETag if available, otherwise fall back to computing hash
    content_hash = etag if etag else response.headers.get("ETag", "").strip('"')
    if not content_hash:
        import hashlib

        content_hash = hashlib.sha256(response.content).hexdigest()[:32]

    slug = _extract_slug(url)
    cache_path = RESOURCE_CACHE / f"{content_hash}-{slug}"

    with open(cache_path, "wb") as f:
        f.write(response.content)

    return cache_path
