"""Tests for the `.mar` model-archive format."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import jax
import pytest

from mo_net.archive import (
    LAYERS_BEST_PATH,
    LAYERS_LAST_PATH,
    LAYERS_MODEL_PATH,
    MANIFEST_FILENAME,
    ArchiveError,
    Manifest,
    ModelMetadata,
    TrainingState,
    UnsupportedManifestVersionError,
    read_manifest,
    write_manifest,
)
from mo_net.samples.word2vec import CBOWModel
from mo_net.samples.word2vec.archive import (
    DATA_VOCAB_PATH,
    LEGACY_METADATA_PATH,
    LEGACY_MODEL_PATH,
    LEGACY_VOCAB_PATH,
    load_word2vec_archive,
    peek_manifest,
    save_word2vec_archive,
)
from mo_net.samples.word2vec.softmax_strategy import SoftmaxConfig
from mo_net.samples.word2vec.vocab import Vocab


@pytest.fixture
def tiny_model_and_vocab() -> tuple[CBOWModel, Vocab]:
    sentences = [["the", "quick", "brown", "fox", "jumps", "lazy", "dog"]]
    vocab, _ = Vocab.from_sentences(sentences, max_size=100)
    model = CBOWModel.create(
        vocab=vocab,
        embedding_dim=8,
        context_size=2,
        softmax_config=SoftmaxConfig.negative_sampling(k=3),
        key=jax.random.PRNGKey(42),
    )
    return model, vocab


def _full_manifest() -> Manifest:
    return Manifest(
        metadata=ModelMetadata(type="cbow", softmax_strategy="full"),
        training_state=TrainingState(
            completed_epoch=3,
            current_iteration=600,
            current_learning_rate=4.2e-5,
            lr_schedule_phase="cosine",
            checkpoint_strategy="min-val",
            seed=1234567890,
            run_name="test-run",
            parent_run_name=None,
            best_val_loss=5.123,
            best_val_loss_epoch=2,
            build_rev="abc1234",
        ),
    )


class TestManifestRoundTrip:
    """Pydantic serialisation preserves every manifest field exactly."""

    def test_round_trip_recovers_identical_values(self, tmp_path: Path) -> None:
        """A fully-populated manifest survives write→read with byte-for-byte equality."""
        manifest = _full_manifest()
        archive = tmp_path / "round_trip.mar"
        with zipfile.ZipFile(archive, "w") as zf:
            write_manifest(zf, manifest)

        with zipfile.ZipFile(archive, "r") as zf:
            loaded = read_manifest(zf)

        assert loaded == manifest

    def test_round_trip_with_none_training_state(self, tmp_path: Path) -> None:
        """A manifest with no training_state (a fresh, never-trained archive) round-trips cleanly."""
        manifest = Manifest(
            metadata=ModelMetadata(type="skipgram", softmax_strategy="full")
        )
        archive = tmp_path / "no_state.mar"
        with zipfile.ZipFile(archive, "w") as zf:
            write_manifest(zf, manifest)

        with zipfile.ZipFile(archive, "r") as zf:
            loaded = read_manifest(zf)

        assert loaded.training_state is None
        assert loaded.metadata.type == "skipgram"


class TestManifestVersionGuard:
    """Future-format archives fail loudly instead of being misread."""

    def test_unknown_version_raises(self, tmp_path: Path) -> None:
        """A manifest_version we don't understand raises UnsupportedManifestVersionError."""
        archive = tmp_path / "future.mar"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(
                MANIFEST_FILENAME,
                json.dumps(
                    {
                        "manifest_version": 2,
                        "metadata": {"type": "cbow", "softmax_strategy": "full"},
                        "training_state": None,
                    }
                ),
            )

        with zipfile.ZipFile(archive, "r") as zf:
            with pytest.raises(UnsupportedManifestVersionError):
                read_manifest(zf)


class TestLegacyFallback:
    """Zips produced before the manifest format still load via legacy synthesis."""

    def _write_legacy_zip(self, path: Path, vocab: Vocab, model: CBOWModel) -> None:
        from io import BytesIO

        with zipfile.ZipFile(path, "w") as zf:
            buf = BytesIO()
            model.dump(buf)
            zf.writestr(LEGACY_MODEL_PATH, buf.getvalue())
            zf.writestr(LEGACY_VOCAB_PATH, vocab.serialize())
            zf.writestr(
                LEGACY_METADATA_PATH,
                json.dumps({"type": "cbow", "softmax_strategy": "negative-sampling"}),
            )

    def test_peek_manifest_synthesises_v1(
        self, tmp_path: Path, tiny_model_and_vocab: tuple[CBOWModel, Vocab]
    ) -> None:
        """`peek_manifest` on a legacy zip returns a v1 manifest with `training_state=None`."""
        model, vocab = tiny_model_and_vocab
        legacy = tmp_path / "legacy.zip"
        self._write_legacy_zip(legacy, vocab, model)

        manifest = peek_manifest(legacy)

        assert manifest.manifest_version == 1
        assert manifest.metadata.type == "cbow"
        assert manifest.metadata.softmax_strategy == "negative-sampling"
        assert manifest.training_state is None

    def test_load_legacy_zip(
        self, tmp_path: Path, tiny_model_and_vocab: tuple[CBOWModel, Vocab]
    ) -> None:
        """Full model + vocab load from a legacy zip works transparently."""
        model, vocab = tiny_model_and_vocab
        legacy = tmp_path / "legacy.zip"
        self._write_legacy_zip(legacy, vocab, model)

        loaded_model, loaded_vocab, manifest = load_word2vec_archive(
            legacy, training=False
        )

        assert isinstance(loaded_model, CBOWModel)
        assert len(loaded_vocab) == len(vocab)
        assert manifest.training_state is None


class TestSaveLoadRoundTrip:
    """Full save→load cycle preserves the model, vocab, and training state."""

    def test_single_checkpoint_round_trip(
        self, tmp_path: Path, tiny_model_and_vocab: tuple[CBOWModel, Vocab]
    ) -> None:
        """Single-snapshot strategies (`min-val`/`last`) write to `layers/model.pkl` and survive a round-trip."""
        model, vocab = tiny_model_and_vocab
        manifest = _full_manifest()
        archive = tmp_path / "single.mar"

        save_word2vec_archive(archive, model=model, vocab=vocab, manifest=manifest)

        with zipfile.ZipFile(archive, "r") as zf:
            names = set(zf.namelist())
        assert MANIFEST_FILENAME in names
        assert DATA_VOCAB_PATH in names
        assert LAYERS_MODEL_PATH in names
        assert LAYERS_BEST_PATH not in names
        assert LAYERS_LAST_PATH not in names

        loaded_model, loaded_vocab, loaded_manifest = load_word2vec_archive(
            archive, training=False
        )
        assert isinstance(loaded_model, CBOWModel)
        assert len(loaded_vocab) == len(vocab)
        assert loaded_manifest.training_state is not None
        assert loaded_manifest.training_state.completed_epoch == 3
        assert loaded_manifest.training_state.best_val_loss == pytest.approx(5.123)

    def test_both_strategy_writes_best_and_last(
        self, tmp_path: Path, tiny_model_and_vocab: tuple[CBOWModel, Vocab]
    ) -> None:
        """`--checkpoint-strategy=both` writes both `best.pkl` and `last.pkl`; the loader's `prefer` arg picks between them."""
        model, vocab = tiny_model_and_vocab
        manifest = _full_manifest()
        archive = tmp_path / "both.mar"

        save_word2vec_archive(
            archive,
            model=model,
            vocab=vocab,
            manifest=manifest,
            best_model=model,
        )

        with zipfile.ZipFile(archive, "r") as zf:
            names = set(zf.namelist())
        assert LAYERS_BEST_PATH in names
        assert LAYERS_LAST_PATH in names
        assert LAYERS_MODEL_PATH not in names

        loaded_best, _, _ = load_word2vec_archive(archive, prefer="best")
        loaded_last, _, _ = load_word2vec_archive(archive, prefer="last")
        assert isinstance(loaded_best, CBOWModel)
        assert isinstance(loaded_last, CBOWModel)


class TestExtensionAgnostic:
    """File extension is cosmetic; format detection is by `manifest.json` presence."""

    def test_load_works_with_zip_extension(
        self, tmp_path: Path, tiny_model_and_vocab: tuple[CBOWModel, Vocab]
    ) -> None:
        """A `.mar` payload saved with a `.zip` suffix loads identically."""
        model, vocab = tiny_model_and_vocab
        manifest = _full_manifest()
        as_zip = tmp_path / "archive.zip"

        # Save uses the path as given — extension doesn't drive format selection.
        save_word2vec_archive(as_zip, model=model, vocab=vocab, manifest=manifest)
        loaded_model, _, loaded_manifest = load_word2vec_archive(as_zip)

        assert isinstance(loaded_model, CBOWModel)
        assert loaded_manifest.training_state is not None


class TestErrorPaths:
    """Corrupt or incomplete archives surface a typed error rather than crashing."""

    def test_missing_payload_raises_archive_error(self, tmp_path: Path) -> None:
        """A manifest-only archive (no `layers/*.pkl`) raises `ArchiveError` on load."""
        archive = tmp_path / "incomplete.mar"
        with zipfile.ZipFile(archive, "w") as zf:
            write_manifest(
                zf,
                Manifest(metadata=ModelMetadata(type="cbow", softmax_strategy="full")),
            )

        with pytest.raises(ArchiveError):
            load_word2vec_archive(archive)

    def test_unrecognisable_zip_raises_archive_error(self, tmp_path: Path) -> None:
        """A zip that's neither new format nor legacy raises `ArchiveError` on peek."""
        archive = tmp_path / "junk.mar"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("readme.txt", "not a model")

        with pytest.raises(ArchiveError):
            peek_manifest(archive)
