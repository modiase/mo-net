"""Generic model-archive (`.mar`) format.

A `.mar` is a zip with a top-level ``manifest.json`` describing both
immutable model identity (``metadata``) and mutable training progress
(``training_state``). Payload bytes live under ``layers/`` (model weights)
and ``data/`` (sample-specific assets such as vocabularies, label
encoders, normalisation stats).

Sample integrations subclass :class:`ModelArchive`, implementing the four
hooks (:meth:`ModelArchive._write_data`, :meth:`ModelArchive._read_data`,
:meth:`ModelArchive._instantiate_model`,
:meth:`ModelArchive._synthesise_legacy_manifest`) and reusing the base
class's zip-and-manifest plumbing — including the legacy-zip fallback,
the ``best`` / ``last`` / ``model`` layer-pickle selection, and manifest
version validation.
"""

from __future__ import annotations

import json
import zipfile
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import IO, Any, Final, Generic, Literal, TypeVar
from zipfile import ZipFile

from pydantic import BaseModel

MAR_EXTENSION: Final = ".mar"
MANIFEST_FILENAME: Final = "manifest.json"

LAYERS_MODEL_PATH: Final = "layers/model.pkl"
LAYERS_BEST_PATH: Final = "layers/best.pkl"
LAYERS_LAST_PATH: Final = "layers/last.pkl"

type ModelType = Literal["cbow", "skipgram"]
type SoftmaxStrategyName = Literal["full", "negative-sampling", "hierarchical"]
type LRSchedulePhase = Literal["warmup", "cosine", "constant"]
type CheckpointStrategyName = Literal["min-val", "last", "both"]
type LayerPreference = Literal["best", "last"]


class ModelMetadata(BaseModel):
    """Immutable model identity — what the archive contains."""

    type: ModelType
    softmax_strategy: SoftmaxStrategyName


class TrainingState(BaseModel):
    """Mutable training progress — where in the schedule the model is.

    `None`-valued fields are tolerated for archives built without a full
    training context (legacy zips, freshly-bundled bases).
    """

    completed_epoch: int
    current_iteration: int
    current_learning_rate: float
    lr_schedule_phase: LRSchedulePhase
    checkpoint_strategy: CheckpointStrategyName
    seed: int
    run_name: str | None = None
    parent_run_name: str | None = None
    # Shared across every segment of a resume chain; minted by the logging
    # backend on the first run and inherited via the manifest on resume.
    lineage_id: str | None = None
    best_val_loss: float | None = None
    best_val_loss_epoch: int | None = None
    build_rev: str | None = None


class Manifest(BaseModel):
    manifest_version: Literal[1] = 1
    metadata: ModelMetadata
    training_state: TrainingState | None = None


class ArchiveError(Exception):
    """Raised when an archive can't be parsed (corrupt zip, missing payloads)."""


class UnsupportedManifestVersionError(ArchiveError):
    """Raised when the manifest's `manifest_version` is not understood."""


def read_manifest(zf: ZipFile) -> Manifest:
    """Read a manifest from an already-open zip.

    Raises :class:`KeyError` if ``manifest.json`` is not present, and
    :class:`UnsupportedManifestVersionError` if the version is unknown.
    """
    raw = zf.read(MANIFEST_FILENAME)
    payload = json.loads(raw.decode("utf-8"))
    version = payload.get("manifest_version")
    if version != 1:
        raise UnsupportedManifestVersionError(
            f"Unsupported manifest_version: {version!r}. This build only knows v1."
        )
    return Manifest.model_validate(payload)


def write_manifest(zf: ZipFile, manifest: Manifest) -> None:
    """Write a manifest into an already-open zip."""
    zf.writestr(
        MANIFEST_FILENAME,
        manifest.model_dump_json(indent=2).encode("utf-8"),
    )


ModelT = TypeVar("ModelT")
DataT = TypeVar("DataT")


class ModelArchive(ABC, Generic[ModelT, DataT]):
    """Read/write the ``.mar`` model-archive format.

    Owns all zip-and-manifest plumbing — layer-pickle selection across the
    three checkpoint strategies, legacy-zip fallback, manifest version
    validation, archive-error wrapping. Subclasses implement four hooks
    describing the sample-specific payload schema:

    - :meth:`_write_data`              — what to write under ``data/``
    - :meth:`_read_data`               — how to read it back
    - :meth:`_instantiate_model`       — how to turn ``layers/*.pkl`` bytes
                                         into a model
    - :meth:`_synthesise_legacy_manifest` — how to recover a v1 manifest
                                            from a pre-format zip

    A second sample (e.g. an MNIST CNN archive) just defines those four
    hooks; the save/load/peek surface is inherited unchanged.
    """

    def save(
        self,
        path: Path,
        *,
        model: ModelT,
        manifest: Manifest,
        best_model: ModelT | None = None,
        **payload: Any,
    ) -> None:
        """Write a ``.mar`` archive to ``path``.

        ``best_model`` is only used for ``--checkpoint-strategy=both``: when
        passed, ``model`` is treated as the "last" snapshot and ``best_model``
        as the "best" one. Single-checkpoint strategies pass only ``model``.

        ``**payload`` is forwarded to :meth:`_write_data` so subclasses can
        accept arbitrary sample-specific arguments (e.g. ``vocab=`` for
        word2vec).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = BytesIO()
        with ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            write_manifest(zf, manifest)
            self._write_data(zf, **payload)
            if best_model is not None:
                zf.writestr(LAYERS_LAST_PATH, self._dump_model(model))
                zf.writestr(LAYERS_BEST_PATH, self._dump_model(best_model))
            else:
                zf.writestr(LAYERS_MODEL_PATH, self._dump_model(model))
        path.write_bytes(buf.getvalue())

    def load(
        self,
        path: Path,
        *,
        prefer: LayerPreference = "best",
        **kwargs: Any,
    ) -> tuple[ModelT, DataT, Manifest]:
        """Load a model, its data payload, and the manifest from ``path``.

        Accepts both the new ``.mar`` layout and legacy zips (detected by
        the absence of ``manifest.json`` and routed through
        :meth:`_synthesise_legacy_manifest`).

        ``**kwargs`` is forwarded to :meth:`_instantiate_model` for
        sample-specific load options (``training=``, ``key=``).
        """
        try:
            with ZipFile(path, "r") as zf:
                manifest = self._read_manifest_or_synthesise(zf)
                has_manifest = MANIFEST_FILENAME in zf.namelist()
                data = self._read_data(zf, manifest, has_manifest=has_manifest)
                layer_path = self._select_layer_path(
                    zf, has_manifest=has_manifest, prefer=prefer
                )
                with zf.open(layer_path) as source:
                    model = self._instantiate_model(
                        source=source, manifest=manifest, data=data, **kwargs
                    )
        except UnsupportedManifestVersionError:
            raise
        except KeyError as exc:
            raise ArchiveError(
                f"Missing entry {exc.args[0]!r} in archive: {path}"
            ) from exc
        except zipfile.BadZipFile as exc:
            raise ArchiveError(f"Invalid archive: {path}") from exc
        return model, data, manifest

    def peek_manifest(self, path: Path) -> Manifest:
        """Read just the manifest from ``path`` without loading the model.

        Cheap probe for UI / lineage callers. Handles both layouts via
        :meth:`_synthesise_legacy_manifest`.
        """
        try:
            with ZipFile(path, "r") as zf:
                return self._read_manifest_or_synthesise(zf)
        except zipfile.BadZipFile as exc:
            raise ArchiveError(f"Invalid archive: {path}") from exc

    @abstractmethod
    def _write_data(self, zf: ZipFile, /, **payload: Any) -> None:
        """Write the sample-specific ``data/*`` entries into ``zf``."""

    @abstractmethod
    def _read_data(
        self, zf: ZipFile, manifest: Manifest, *, has_manifest: bool
    ) -> DataT:
        """Read the sample-specific data payload from ``zf``.

        ``has_manifest`` lets subclasses pick between the new
        ``data/<file>`` location and the legacy at-the-root layout.
        """

    @abstractmethod
    def _instantiate_model(
        self,
        *,
        source: IO[bytes],
        manifest: Manifest,
        data: DataT,
        **kwargs: Any,
    ) -> ModelT:
        """Construct a model from a layer pickle byte stream.

        ``data`` is the result of :meth:`_read_data` and is passed in case
        the model construction needs it (e.g. word2vec uses the vocab to
        derive the negative-sampling distribution).
        """

    @abstractmethod
    def _synthesise_legacy_manifest(self, zf: ZipFile) -> Manifest:
        """Build a v1 manifest from a pre-format zip.

        Called when ``manifest.json`` is absent. Subclasses know what
        legacy metadata file to read; typically returns a manifest with
        ``training_state=None``.
        """

    def _dump_model(self, model: ModelT) -> bytes:
        """Serialise ``model`` to bytes. Default uses ``model.dump(buf)``."""
        buf = BytesIO()
        model.dump(buf)  # type: ignore[attr-defined]
        return buf.getvalue()

    def _read_manifest_or_synthesise(self, zf: ZipFile) -> Manifest:
        if MANIFEST_FILENAME in zf.namelist():
            return read_manifest(zf)
        return self._synthesise_legacy_manifest(zf)

    def _select_layer_path(
        self, zf: ZipFile, *, has_manifest: bool, prefer: LayerPreference
    ) -> str:
        if not has_manifest:
            return self._legacy_layer_path()
        names = set(zf.namelist())
        # Per-preference search order: try the preferred snapshot first,
        # then fall back through the single-checkpoint canonical path,
        # then the other named snapshot. Covers min-val / last / both
        # without the caller needing to know which is on disk.
        candidates_by_preference: dict[LayerPreference, tuple[str, ...]] = {
            "best": (LAYERS_BEST_PATH, LAYERS_MODEL_PATH, LAYERS_LAST_PATH),
            "last": (LAYERS_LAST_PATH, LAYERS_MODEL_PATH, LAYERS_BEST_PATH),
        }
        for candidate in candidates_by_preference[prefer]:
            if candidate in names:
                return candidate
        raise ArchiveError(
            "Archive has a manifest but no layer pickle "
            f"(looked for {candidates_by_preference[prefer]})."
        )

    def _legacy_layer_path(self) -> str:
        """Path inside a legacy zip where the model pickle lives.

        Defaults to ``model.pkl`` (matches every legacy w2v zip). Subclasses
        with a different legacy convention can override.
        """
        return "model.pkl"
