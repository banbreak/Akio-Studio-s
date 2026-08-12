"""Portable ``.synthetic_actor`` bundle export for Akio Studio characters.

A synthetic actor is the studio's portable character asset: identity
embeddings, LoRA weights, pose rigs, and a voice model packed into a single
zip archive with a verifiable manifest. Bundles are self-describing — the
manifest carries a streamed SHA-256 for every payload file so a bundle copied
between machines (or restored from backup) can be integrity-checked without
any external state.

Design notes honoring the architecture audit:

* Hashes are computed by streaming in 1 MiB chunks — model weights are
  multi-GiB and must never be slurped into the orchestrator's memory.
* Weight/binary payloads (``.safetensors``, ``.gguf``, ...) are archived with
  ``ZIP_STORED``: fp16/quantized tensors are high-entropy, so deflating
  gigabytes burns minutes of CPU for roughly 5% savings.
* The archive is written to a ``.tmp`` sibling in the destination directory
  and moved into place with :func:`os.replace`, flushed to stable storage
  first, so a crash can never leave a truncated bundle behind the final name
  (audit finding A9).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from akio_studio._io import _flush_to_stable_storage
from akio_studio.exceptions import ActorExportError, MissingAssetError

logger = logging.getLogger(__name__)

#: Asset categories a bundle may contain; each becomes a top-level archive dir.
BUNDLE_CATEGORIES: tuple[str, ...] = (
    "identity_embeds",
    "lora_weights",
    "pose_rigs",
    "voice_model",
)

#: Safe identifier: starts alphanumeric, then alphanumerics/dot/underscore/dash.
#: The character class cannot express a path separator on any platform.
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Extensions archived uncompressed — weight formats do not deflate usefully.
_STORED_EXTENSIONS: frozenset[str] = frozenset(
    {".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".onnx", ".npz", ".zip"}
)

#: Streaming granularity for hashing and verification.
_CHUNK_SIZE: int = 1024 * 1024

_MANIFEST_NAME = "manifest.json"


def _sha256_stream(handle: Any) -> str:
    """Return the hex SHA-256 of a readable binary stream, 1 MiB at a time."""
    digest = hashlib.sha256()
    while chunk := handle.read(_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


class SyntheticActorSDKExporter:
    """Assembles and verifies ``.synthetic_actor`` character bundles.

    The exporter owns a single output directory; every exported bundle lands
    there as ``{character_id}.synthetic_actor``.
    """

    def __init__(self, output_dir: Path) -> None:
        """Remember ``output_dir``; it is created lazily on first export."""
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------ export

    def export_synthetic_actor(
        self,
        character_id: str,
        asset_paths: dict[str, Path | list[Path]],
        manifest_extra: dict[str, Any] | None = None,
    ) -> Path:
        """Pack character assets into an atomic ``.synthetic_actor`` zip.

        Args:
            character_id: Safe slug naming the actor (and the bundle file).
            asset_paths: Mapping of bundle category (a subset of
                :data:`BUNDLE_CATEGORIES`) to one source file or a list of
                source files. At least one category must contribute a file.
            manifest_extra: Optional shallow overrides merged into the
                manifest (e.g. ``danbooru_tag_anchors``, ``power_level``,
                ``model_specs``). The computed ``files`` list and the
                ``character_id`` argument stay authoritative.

        Returns:
            Path of the finished bundle.

        Raises:
            ActorExportError: Invalid slug, unknown category, no assets at
                all, or duplicate archive entry names.
            MissingAssetError: A source path does not exist or is not a file.
        """
        self._validate_character_id(character_id)
        normalized = self._normalize_asset_paths(asset_paths)

        entries: dict[str, Path] = {}
        file_records: list[dict[str, Any]] = []
        for category, sources in normalized.items():
            for source in sources:
                arcname = f"{category}/{source.name}"
                if arcname in entries:
                    raise ActorExportError(
                        f"duplicate archive entry {arcname!r}: "
                        f"{entries[arcname]} and {source} share a filename"
                    )
                entries[arcname] = source
                with source.open("rb") as handle:
                    sha256 = _sha256_stream(handle)
                file_records.append(
                    {
                        "category": category,
                        "name": source.name,
                        "sha256": sha256,
                        "size_bytes": source.stat().st_size,
                    }
                )
        file_records.sort(key=lambda record: (record["category"], record["name"]))

        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "character_id": character_id,
            "created_at": datetime.now(UTC).isoformat(),
            "danbooru_tag_anchors": [],
            "power_level": None,
            "model_specs": {},
        }
        if manifest_extra:
            manifest.update(manifest_extra)
        # Computed/derived fields always win over manifest_extra.
        manifest["character_id"] = character_id
        manifest["files"] = file_records

        self.output_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = self.output_dir / f"{character_id}.synthetic_actor"
        # Unique temp name: a fixed ".tmp" sibling would let two concurrent
        # exports of the same character interleave writes into one inode and
        # install a corrupt bundle.
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=self.output_dir, prefix=f".{bundle_path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "wb") as raw:
                with zipfile.ZipFile(raw, mode="w") as archive:
                    archive.writestr(
                        _MANIFEST_NAME,
                        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                    for arcname in sorted(entries, key=self._arcname_sort_key):
                        source = entries[arcname]
                        compress_type = (
                            zipfile.ZIP_STORED
                            if source.suffix.lower() in _STORED_EXTENSIONS
                            else zipfile.ZIP_DEFLATED
                        )
                        # ZipFile.write streams from disk internally — the
                        # multi-GiB weights never enter this process whole.
                        archive.write(source, arcname=arcname, compress_type=compress_type)
                raw.flush()
                _flush_to_stable_storage(raw.fileno())
            os.replace(tmp_path, bundle_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

        total_bytes = sum(record["size_bytes"] for record in file_records)
        logger.info(
            "exported synthetic actor %s: %d files, %d bytes -> %s",
            character_id,
            len(file_records),
            total_bytes,
            bundle_path,
        )
        return bundle_path

    # ----------------------------------------------------------------- inspect

    def inspect_bundle(self, bundle_path: Path) -> dict[str, Any]:
        """Parse a bundle's manifest and verify every listed SHA-256.

        Each manifest ``files`` entry is re-hashed by streaming the archived
        bytes in 1 MiB chunks and compared against its recorded digest.

        Args:
            bundle_path: Path to a ``.synthetic_actor`` archive.

        Returns:
            The parsed manifest dictionary.

        Raises:
            ActorExportError: Missing/unreadable bundle, corrupt zip, absent
                or malformed manifest, missing archive entries, or a SHA-256
                mismatch.
        """
        bundle_path = Path(bundle_path)
        if not bundle_path.is_file():
            raise ActorExportError(f"bundle does not exist: {bundle_path}")

        try:
            with zipfile.ZipFile(bundle_path) as archive:
                manifest = self._read_manifest(archive, bundle_path)
                records = manifest.get("files")
                if not isinstance(records, list):
                    raise ActorExportError(
                        f"{bundle_path}: manifest 'files' is missing or not a list"
                    )
                listed = {_MANIFEST_NAME}
                available = set(archive.namelist())
                for record in records:
                    arcname = self._record_arcname(record, bundle_path)
                    listed.add(arcname)
                    if arcname not in available:
                        raise ActorExportError(
                            f"{bundle_path}: manifest lists {arcname!r} "
                            "but the archive has no such entry"
                        )
                    with archive.open(arcname) as handle:
                        actual = _sha256_stream(handle)
                    if actual != record["sha256"]:
                        raise ActorExportError(
                            f"{bundle_path}: sha256 mismatch for {arcname!r} "
                            f"(manifest {record['sha256']}, archive {actual})"
                        )
                unlisted = available - listed
                if unlisted:
                    logger.warning(
                        "%s contains %d entries absent from its manifest: %s",
                        bundle_path,
                        len(unlisted),
                        sorted(unlisted),
                    )
        except zipfile.BadZipFile as exc:
            raise ActorExportError(f"not a valid bundle zip: {bundle_path}") from exc

        logger.info("verified bundle %s: %d files intact", bundle_path, len(records))
        return manifest

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _validate_character_id(character_id: str) -> None:
        """Reject identifiers that are not safe filename slugs."""
        if not isinstance(character_id, str) or not _SLUG_RE.fullmatch(character_id):
            raise ActorExportError(
                f"character_id {character_id!r} is not a safe slug "
                "(expected ^[A-Za-z0-9][A-Za-z0-9._-]*$ with no path separators)"
            )

    @staticmethod
    def _normalize_asset_paths(
        asset_paths: dict[str, Path | list[Path]],
    ) -> dict[str, list[Path]]:
        """Validate categories/sources; return category -> list[Path].

        Unknown categories raise :class:`ActorExportError`; nonexistent or
        non-file sources raise :class:`MissingAssetError` naming the path.
        """
        normalized: dict[str, list[Path]] = {}
        for category, value in asset_paths.items():
            if category not in BUNDLE_CATEGORIES:
                raise ActorExportError(
                    f"unknown bundle category {category!r}; "
                    f"expected one of {BUNDLE_CATEGORIES}"
                )
            sources = list(value) if isinstance(value, (list, tuple)) else [value]
            paths: list[Path] = []
            for source in sources:
                path = Path(source)
                if not path.exists():
                    raise MissingAssetError(f"actor asset does not exist: {path}")
                if not path.is_file():
                    raise MissingAssetError(f"actor asset is not a file: {path}")
                paths.append(path)
            normalized[category] = paths
        if not any(normalized.values()):
            raise ActorExportError(
                "asset_paths contributed no files; at least one category must be non-empty"
            )
        return normalized

    @staticmethod
    def _arcname_sort_key(arcname: str) -> tuple[str, str]:
        """Sort key matching the manifest ordering: (category, name)."""
        category, _, name = arcname.partition("/")
        return (category, name)

    @staticmethod
    def _read_manifest(archive: zipfile.ZipFile, bundle_path: Path) -> dict[str, Any]:
        """Load and parse ``manifest.json`` from an open archive."""
        try:
            with archive.open(_MANIFEST_NAME) as handle:
                manifest = json.load(handle)
        except KeyError as exc:
            raise ActorExportError(f"{bundle_path}: archive has no {_MANIFEST_NAME}") from exc
        except json.JSONDecodeError as exc:
            raise ActorExportError(f"{bundle_path}: {_MANIFEST_NAME} is not valid JSON") from exc
        if not isinstance(manifest, dict):
            raise ActorExportError(f"{bundle_path}: {_MANIFEST_NAME} is not a JSON object")
        return manifest

    @staticmethod
    def _record_arcname(record: Any, bundle_path: Path) -> str:
        """Validate a manifest file record and derive its archive name."""
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("category"), str)
            or not isinstance(record.get("name"), str)
            or not isinstance(record.get("sha256"), str)
        ):
            raise ActorExportError(
                f"{bundle_path}: malformed manifest file record: {record!r}"
            )
        return f"{record['category']}/{record['name']}"
