import hashlib
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import requests

from mo_net import PROJECT_ROOT

RESOURCE_CACHE: Final[Path] = PROJECT_ROOT / ".resource_cache"

MNIST_TRAIN_URL: Final[str] = "s3://mo-net-resources/mnist_train.csv"
MNIST_TEST_URL: Final[str] = "s3://mo-net-resources/mnist_test.csv"


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

    cache_path = RESOURCE_CACHE / hashlib.md5(url.encode()).hexdigest()

    if cache_path.exists():
        return cache_path

    response = requests.get(download_url)
    response.raise_for_status()

    with open(cache_path, "wb") as f:
        f.write(response.content)

    return cache_path
