"""Tests for the resource registry and the shipped handlers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mo_net import resources


@pytest.fixture(autouse=True)
def _resource_cache_in_tmp(monkeypatch, tmp_path: Path):
    """Point the resource cache at a tmp dir so tests don't hit ~/.cache."""
    monkeypatch.setenv("MO_NET_RESOURCE_CACHE", str(tmp_path / "cache"))
    from mo_net.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_register_adds_handler_for_each_scheme():
    seen: list[str] = []

    def handler(parsed):
        seen.append(parsed.scheme)
        return SimpleNamespace(rows=())

    resources.register("scheme-a", "scheme-b")(handler)  # type: ignore[arg-type]

    try:
        resources.get_resource("scheme-a://x")
        resources.get_resource("scheme-b://y")
    finally:
        del resources._HANDLERS["scheme-a"]
        del resources._HANDLERS["scheme-b"]

    assert seen == ["scheme-a", "scheme-b"]


def test_get_resource_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported URL scheme 'definitely-not'"):
        resources.get_resource("definitely-not://anywhere")


def test_wrap_path_as_dataset_rejects_unknown_suffix(tmp_path: Path):
    junk = tmp_path / "thing.bin"
    junk.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="Cannot wrap '.bin'"):
        resources._wrap_path_as_dataset(junk)


def test_file_handler_loads_text_one_row_per_line(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("first line\nsecond line\nthird\n")
    ds = resources.get_resource(f"file://{corpus}")
    assert ds.column_names == ["text"]
    assert list(ds["text"]) == ["first line", "second line", "third"]


def test_file_handler_loads_csv_with_header(tmp_path: Path):
    csv = tmp_path / "rows.csv"
    csv.write_text("label,a,b\n0,1,2\n1,3,4\n")
    ds = resources.get_resource(f"file://{csv}")
    assert ds.column_names == ["label", "a", "b"]
    assert list(ds["label"]) == [0, 1]


def test_file_handler_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resources.get_resource(f"file://{tmp_path / 'nope.txt'}")


def test_http_download_caches_by_etag(monkeypatch, tmp_path: Path):
    body = b"alpha\nbeta\ngamma\n"

    head_mock = MagicMock()
    head_mock.headers = {"ETag": '"etag-xyz"'}
    head_mock.raise_for_status = MagicMock()

    get_mock = MagicMock()
    get_mock.headers = {"ETag": '"etag-xyz"'}
    get_mock.content = body
    get_mock.raise_for_status = MagicMock()

    head_calls: list[str] = []
    get_calls: list[str] = []

    def _head(url, **_):
        head_calls.append(url)
        return head_mock

    def _get(url, **_):
        get_calls.append(url)
        return get_mock

    monkeypatch.setattr(resources.requests, "head", _head)
    monkeypatch.setattr(resources.requests, "get", _get)

    ds1 = resources.get_resource("https://example.invalid/corpus.txt")
    assert list(ds1["text"]) == ["alpha", "beta", "gamma"]
    assert len(get_calls) == 1

    ds2 = resources.get_resource("https://example.invalid/corpus.txt")
    assert list(ds2["text"]) == ["alpha", "beta", "gamma"]
    assert len(get_calls) == 1, "second call should hit the on-disk cache"
    assert len(head_calls) == 2


def test_s3_rewrites_to_https(monkeypatch):
    captured: dict[str, str] = {}

    def _head(url, **_):
        captured["head"] = url
        return MagicMock(headers={"ETag": '"e"'}, raise_for_status=MagicMock())

    def _get(url, **_):
        captured["get"] = url
        return MagicMock(
            headers={"ETag": '"e"'},
            content=b"x\ny\n",
            raise_for_status=MagicMock(),
        )

    monkeypatch.setattr(resources.requests, "head", _head)
    monkeypatch.setattr(resources.requests, "get", _get)

    resources.get_resource("s3://my-bucket/path/to/file.txt")
    assert captured["head"] == "https://my-bucket.s3.amazonaws.com/path/to/file.txt"


def test_hf_url_parses_query_and_loads(monkeypatch):
    captured: dict = {}
    rename_calls: list[tuple[str, str]] = []

    class _Stub:
        column_names = ["content"]

        def rename_column(self, old: str, new: str):
            rename_calls.append((old, new))
            return self

    def _load_dataset(repo_id, config, *, split, cache_dir):
        captured["repo_id"] = repo_id
        captured["config"] = config
        captured["split"] = split
        captured["cache_dir"] = cache_dir
        return _Stub()

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    resources.get_resource(
        "hf://acme/widget-corpus?config=v1&split=validation&text_field=content"
    )
    assert captured["repo_id"] == "acme/widget-corpus"
    assert captured["config"] == "v1"
    assert captured["split"] == "validation"
    assert captured["cache_dir"].endswith("/huggingface")
    assert rename_calls == [("content", "text")]


def test_hf_defaults_split_train_no_rename_when_already_text(monkeypatch):
    captured: dict = {}
    rename_calls: list[tuple[str, str]] = []

    class _Stub:
        column_names = ["text"]

        def rename_column(self, old: str, new: str):
            rename_calls.append((old, new))
            return self

    def _load_dataset(repo_id, config, *, split, cache_dir):
        captured["repo_id"] = repo_id
        captured["config"] = config
        captured["split"] = split
        return _Stub()

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    resources.get_resource("hf://owner/name")
    assert captured["split"] == "train"
    assert captured["config"] is None
    assert rename_calls == []


def test_head_resource_http_only_does_head(monkeypatch):
    calls: list[tuple[str, str]] = []

    def _head(url, **_):
        calls.append(("head", url))
        return MagicMock(raise_for_status=MagicMock())

    def _get(*_, **__):  # pragma: no cover — should never run
        calls.append(("get", ""))
        raise AssertionError("probe must not call GET")

    monkeypatch.setattr(resources.requests, "head", _head)
    monkeypatch.setattr(resources.requests, "get", _get)

    resources.head_resource("https://example.invalid/thing.txt")
    assert calls == [("head", "https://example.invalid/thing.txt")]


def test_head_resource_hf_uses_dataset_info(monkeypatch):
    seen: list[str] = []

    from huggingface_hub import dataset_info as real_dataset_info  # noqa: F401

    def _info(repo_id):
        seen.append(repo_id)
        return MagicMock()

    monkeypatch.setattr("huggingface_hub.dataset_info", _info)

    resources.head_resource("hf://acme/widget-corpus?split=train")
    assert seen == ["acme/widget-corpus"]


def test_head_resource_file_raises_for_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resources.head_resource(f"file://{tmp_path / 'ghost.csv'}")


def test_download_falls_back_to_sha256_when_etag_missing(monkeypatch):
    body = b"no-etag-here\n"

    def _head(url, **_):
        return MagicMock(headers={}, raise_for_status=MagicMock())

    def _get(url, **_):
        return MagicMock(headers={}, content=body, raise_for_status=MagicMock())

    monkeypatch.setattr(resources.requests, "head", _head)
    monkeypatch.setattr(resources.requests, "get", _get)

    path = resources._download_with_etag("https://example.invalid/x.txt")
    import hashlib

    expected = hashlib.sha256(body).hexdigest()[:32]
    assert path.name.startswith(expected)


def test_extract_slug_strips_suffix_and_normalises():
    assert resources._extract_slug("s3://b/path/to/My File.csv") == "my-file"
    assert resources._extract_slug("https://x/?q=1") == "resource"


def test_iter_text_rows_file_yields_lines(tmp_path: Path):
    corpus = tmp_path / "doc.txt"
    corpus.write_text("first\nsecond line\nthird\n")
    assert list(resources.iter_text_rows(f"file://{corpus}")) == [
        "first",
        "second line",
        "third",
    ]


def test_iter_text_rows_hf_uses_streaming(monkeypatch):
    seen_kwargs: dict = {}

    def _load_dataset(repo_id, config, *, split, cache_dir, streaming):
        seen_kwargs.update(
            repo_id=repo_id,
            config=config,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
        return iter([{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    rows = list(resources.iter_text_rows("hf://acme/corpus?config=v1&split=train"))
    assert rows == ["alpha", "beta", "gamma"]
    assert seen_kwargs["streaming"] is True
    assert seen_kwargs["repo_id"] == "acme/corpus"
    assert seen_kwargs["config"] == "v1"
    assert seen_kwargs["split"] == "train"


def test_iter_text_rows_hf_honors_text_field(monkeypatch):
    def _load_dataset(*_, **__):
        return iter([{"content": "x"}, {"content": "y"}])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _load_dataset)

    rows = list(resources.iter_text_rows("hf://o/n?text_field=content"))
    assert rows == ["x", "y"]


def test_iter_text_rows_csv_rejected(tmp_path: Path):
    csv = tmp_path / "x.csv"
    csv.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Cannot stream"):
        list(resources.iter_text_rows(f"file://{csv}"))


def test_download_writes_atomically_no_tmp_left_on_success(monkeypatch, tmp_path):
    body = b"a,b\n1,2\n"

    monkeypatch.setattr(
        resources.requests,
        "head",
        lambda url, **_: MagicMock(
            headers={"ETag": '"abc"'}, raise_for_status=MagicMock()
        ),
    )
    monkeypatch.setattr(
        resources.requests,
        "get",
        lambda url, **_: MagicMock(
            headers={"ETag": '"abc"'}, content=body, raise_for_status=MagicMock()
        ),
    )

    resources.get_resource("https://example.invalid/sample.csv")

    cache = Path(resources.get_settings().resource_cache)
    leftover_tmps = list(cache.glob("*.tmp"))
    assert leftover_tmps == []
    cached = list(cache.glob("abc-*.csv"))
    assert len(cached) == 1


def test_download_lock_creates_and_keeps_lock_file(monkeypatch, tmp_path):
    body = b"text\n"
    monkeypatch.setattr(
        resources.requests,
        "head",
        lambda url, **_: MagicMock(
            headers={"ETag": '"locked"'}, raise_for_status=MagicMock()
        ),
    )
    monkeypatch.setattr(
        resources.requests,
        "get",
        lambda url, **_: MagicMock(
            headers={"ETag": '"locked"'}, content=body, raise_for_status=MagicMock()
        ),
    )

    resources.get_resource("https://example.invalid/x.txt")

    cache = Path(resources.get_settings().resource_cache)
    lock_files = list(cache.glob(".locked.lock"))
    assert len(lock_files) == 1, "lock file should remain (zero-byte sentinel)"
