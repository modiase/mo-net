import os
from pathlib import Path

import pytest

from mo_net.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MO_NET_") or key.startswith("XDG_"):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_env_var_overrides_default(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MO_NET_DATA_DIR", str(tmp_path / "data"))
    assert Settings().data_dir == tmp_path / "data"


def test_resource_cache_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MO_NET_RESOURCE_CACHE", str(tmp_path / "cache"))
    assert Settings().resource_cache == tmp_path / "cache"


def test_db_path_derived_from_data_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MO_NET_DATA_DIR", str(tmp_path))
    s = Settings()
    assert s.db_path is None
    assert s.resolved_db_path == tmp_path / "train.db"


def test_db_path_explicit_overrides_derivation(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MO_NET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MO_NET_DB_PATH", str(tmp_path / "elsewhere.db"))
    assert Settings().resolved_db_path == tmp_path / "elsewhere.db"


def test_run_and_output_dirs_derive_from_data_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MO_NET_DATA_DIR", str(tmp_path))
    s = Settings()
    assert s.run_dir == tmp_path / "run"
    assert s.output_dir == tmp_path / "output"


def test_init_kwarg_wins_over_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MO_NET_DATA_DIR", str(tmp_path / "from-env"))
    s = Settings(data_dir=tmp_path / "from-init")
    assert s.data_dir == tmp_path / "from-init"


def test_get_settings_caches_until_cleared(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MO_NET_DATA_DIR", str(tmp_path / "a"))
    first = get_settings()
    assert first.data_dir == tmp_path / "a"

    monkeypatch.setenv("MO_NET_DATA_DIR", str(tmp_path / "b"))
    cached = get_settings()
    assert cached.data_dir == tmp_path / "a", "should still be cached"

    get_settings.cache_clear()
    fresh = get_settings()
    assert fresh.data_dir == tmp_path / "b"


def test_default_data_dir_falls_back_to_xdg_when_root_is_nix_store(
    monkeypatch, tmp_path: Path
):
    from mo_net import settings as settings_module

    fake_xdg = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(fake_xdg))
    monkeypatch.setattr("mo_net.PROJECT_ROOT_DIR", Path("/nix/store/abc-mo-net-0.1.0"))
    assert settings_module._default_data_dir() == fake_xdg / "mo-net"


def test_resource_cache_default_uses_xdg(monkeypatch, tmp_path: Path):
    from mo_net.settings import _default_resource_cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    assert _default_resource_cache() == tmp_path / "xdg-cache" / "mo-net"
