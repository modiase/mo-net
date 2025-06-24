import os
from pathlib import Path
from typing import Final

__version__: Final[str] = "0.0.13"

ROOT_DIR: Final[Path] = Path(__file__).parent.resolve()

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
