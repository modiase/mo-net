from pathlib import Path
from typing import Final

__version__: Final[str] = "0.0.13"

PACKAGE_DIR: Final[Path] = Path(__file__).parent.resolve()
PROJECT_ROOT_DIR: Final[Path] = PACKAGE_DIR.parent.resolve()
