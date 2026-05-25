"""Layered runtime path configuration.

Precedence (highest first):
    1. Constructor kwargs (when ``Settings(...)`` is built explicitly).
    2. Environment variables (``MO_NET_*`` prefix).
    3. TOML config file at ``$XDG_CONFIG_HOME/mo-net/config.toml``.
    4. Compile-time defaults (writable-checkout heuristic, otherwise XDG).

Single accessor: :func:`get_settings` returns a cached singleton. CLI flag
handlers should mutate ``os.environ`` then call ``get_settings.cache_clear()``
so subsequent reads pick up the new values.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _xdg(env_var: str, fallback_subdir: str) -> Path:
    raw = os.environ.get(env_var)
    return Path(raw) if raw else Path.home() / fallback_subdir


def _default_data_dir() -> Path:
    """Use ``PROJECT_ROOT_DIR/data`` for writable dev checkouts; XDG otherwise.

    The check is a single heuristic: nix-store install paths are treated as
    read-only and trigger the XDG fallback.
    """
    from mo_net import PROJECT_ROOT_DIR

    legacy = PROJECT_ROOT_DIR / "data"
    in_nix_store = str(PROJECT_ROOT_DIR).startswith("/nix/store/")
    if legacy.exists() or (PROJECT_ROOT_DIR.is_dir() and not in_nix_store):
        return legacy
    return _xdg("XDG_DATA_HOME", ".local/share") / "mo-net"


def _default_resource_cache() -> Path:
    return _xdg("XDG_CACHE_HOME", ".cache") / "mo-net"


def _config_toml_path() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / "mo-net" / "config.toml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MO_NET_",
        env_file=None,
        toml_file=_config_toml_path(),
        extra="ignore",
    )

    data_dir: Path = Field(default_factory=_default_data_dir)
    db_path: Path | None = None
    resource_cache: Path = Field(default_factory=_default_resource_cache)

    @property
    def resolved_db_path(self) -> Path:
        return self.db_path if self.db_path is not None else self.data_dir / "train.db"

    @property
    def run_dir(self) -> Path:
        return self.data_dir / "run"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
