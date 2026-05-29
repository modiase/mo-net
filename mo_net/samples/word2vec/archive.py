"""Word2Vec ``.mar`` archive: subclass of :class:`mo_net.archive.ModelArchive`.

The base class owns all zip + manifest plumbing. This module supplies the
word2vec-specific bits:

- ``data/vocab.msgpack`` is the only data payload.
- Layer pickles are :class:`CBOWModel` or :class:`SkipGramModel`,
  dispatched via the manifest's ``metadata.type``.
- The legacy layout had ``model.pkl`` / ``vocab.msgpack`` / ``metadata.json``
  at the zip root; the synthesised v1 manifest copies the metadata fields
  verbatim and sets ``training_state=None``.

Module-level functions (:func:`save_word2vec_archive`,
:func:`load_word2vec_archive`, :func:`peek_manifest`) are thin
delegations to a default :class:`Word2VecArchive` instance for ergonomic
call sites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any, Final
from zipfile import ZipFile

import jax

from mo_net.archive import (
    ArchiveError,
    LayerPreference,
    Manifest,
    ModelArchive,
    ModelMetadata,
)
from mo_net.samples.word2vec.models import CBOWModel, SkipGramModel
from mo_net.samples.word2vec.vocab import Vocab

type Word2VecModel = CBOWModel | SkipGramModel

DATA_VOCAB_PATH: Final = "data/vocab.msgpack"

LEGACY_MODEL_PATH: Final = "model.pkl"
LEGACY_VOCAB_PATH: Final = "vocab.msgpack"
LEGACY_METADATA_PATH: Final = "metadata.json"


class Word2VecArchive(ModelArchive[Word2VecModel, Vocab]):
    # The base's `**payload` exists so subclasses can declare their own data
    # schema; mypy flags the narrower override but the contract is preserved
    # by the module-level wrapper which restates the typed kwargs.
    def _write_data(  # type: ignore[override]
        self, zf: ZipFile, /, *, vocab: Vocab, **_: Any
    ) -> None:
        zf.writestr(DATA_VOCAB_PATH, vocab.serialize())

    def _read_data(
        self, zf: ZipFile, manifest: Manifest, *, has_manifest: bool
    ) -> Vocab:
        path = DATA_VOCAB_PATH if has_manifest else LEGACY_VOCAB_PATH
        return Vocab.from_bytes(zf.read(path))

    def _instantiate_model(
        self,
        *,
        source: IO[bytes],
        manifest: Manifest,
        data: Vocab,
        training: bool = False,
        key: jax.Array | None = None,
        **_: Any,
    ) -> Word2VecModel:
        resolved_key = key if key is not None else jax.random.PRNGKey(0)
        neg_sampling_dist = data.get_negative_sampling_distribution()

        if manifest.metadata.type == "skipgram":
            return SkipGramModel.load(
                source,
                training=training,
                key=resolved_key,
                negative_sampling_dist=neg_sampling_dist,
            )
        if manifest.metadata.type == "cbow":
            return CBOWModel.load(
                source,
                training=training,
                key=resolved_key,
                negative_sampling_dist=neg_sampling_dist,
            )
        raise ArchiveError(
            f"Unknown model type in manifest: {manifest.metadata.type!r}"
        )

    def _synthesise_legacy_manifest(self, zf: ZipFile) -> Manifest:
        try:
            raw = zf.read(LEGACY_METADATA_PATH)
        except KeyError as exc:
            raise ArchiveError(
                "Archive has neither manifest.json nor legacy "
                f"{LEGACY_METADATA_PATH}; not a recognisable word2vec archive."
            ) from exc
        legacy = json.loads(raw.decode("utf-8"))
        return Manifest(
            metadata=ModelMetadata(
                type=legacy["type"],
                softmax_strategy=legacy.get("softmax_strategy", "full"),
            ),
            training_state=None,
        )

    def _legacy_layer_path(self) -> str:
        return LEGACY_MODEL_PATH


_default_archive: Final = Word2VecArchive()


def save_word2vec_archive(
    path: Path,
    *,
    model: Word2VecModel,
    vocab: Vocab,
    manifest: Manifest,
    best_model: Word2VecModel | None = None,
) -> None:
    """Write a word2vec ``.mar`` to ``path``."""
    _default_archive.save(
        path,
        model=model,
        manifest=manifest,
        best_model=best_model,
        vocab=vocab,
    )


def load_word2vec_archive(
    path: Path,
    *,
    training: bool = False,
    key: jax.Array | None = None,
    prefer: LayerPreference = "best",
) -> tuple[Word2VecModel, Vocab, Manifest]:
    """Load (model, vocab, manifest) from a word2vec ``.mar`` or legacy zip."""
    return _default_archive.load(path, prefer=prefer, training=training, key=key)


def peek_manifest(path: Path) -> Manifest:
    """Read just the manifest (cheap UI probe)."""
    return _default_archive.peek_manifest(path)
