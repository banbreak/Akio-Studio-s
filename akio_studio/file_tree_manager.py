"""Production file-tree management for Akio Studio concepts.

Every concept (a show/series candidate) owns one directory under the studio's
production base, with a fixed set of subdirectories for masters, vertical
shorts, per-asset metadata, actor bundles, lore-graph snapshots, and logs.
This module is the single authority for that layout: other modules receive
resolved :class:`~pathlib.Path` objects from it instead of assembling paths
themselves, so identifier validation (safe slugs, no path traversal) happens
in exactly one place.

Metadata documents are written through :func:`akio_studio._io.atomic_write_json`
— the ``.tmp`` + ``os.replace`` pattern the architecture audit mandates — so a
crash mid-write can never leave a truncated JSON file behind a final name.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from akio_studio._io import atomic_write_json
from akio_studio.config import StudioConfig
from akio_studio.exceptions import FileTreeError

logger = logging.getLogger(__name__)

#: Subdirectories created inside every concept's root directory.
CONCEPT_SUBDIRS: tuple[str, ...] = (
    "master_16_9",
    "shorts_9_16",
    "metadata",
    "actors",
    "lore_graph",
    "logs",
)

#: Safe identifier: starts alphanumeric, then alphanumerics/dot/underscore/dash.
#: The character class cannot express a path separator on any platform.
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_METADATA_DIR = "metadata"


class ProductionFileTreeManager:
    """Creates and navigates the on-disk production tree for concepts."""

    def __init__(self, base_dir: Path | None = None) -> None:
        """Root the tree at ``base_dir`` (default: ``StudioConfig().base_dir``)."""
        self.base_dir = Path(base_dir) if base_dir is not None else StudioConfig().base_dir

    # ------------------------------------------------------------------- trees

    def initialize_concept_tree(self, concept_id: str) -> dict[str, Path]:
        """Create (idempotently) the full directory tree for a concept.

        Args:
            concept_id: Safe slug naming the concept.

        Returns:
            Mapping with ``"root"`` plus one entry per name in
            :data:`CONCEPT_SUBDIRS`, each a resolved :class:`~pathlib.Path`.

        Raises:
            FileTreeError: Invalid ``concept_id`` or the directories cannot
                be created.
        """
        root = self._concept_root(concept_id)
        paths: dict[str, Path] = {"root": root}
        try:
            root.mkdir(parents=True, exist_ok=True)
            for subdir in CONCEPT_SUBDIRS:
                path = root / subdir
                path.mkdir(exist_ok=True)
                paths[subdir] = path
        except OSError as exc:
            raise FileTreeError(f"cannot create concept tree for {concept_id!r}: {exc}") from exc
        logger.debug("concept tree ready for %s at %s", concept_id, root)
        return paths

    # ---------------------------------------------------------------- metadata

    def write_asset_metadata(
        self,
        concept_id: str,
        asset_filename: str,
        metadata_payload: dict[str, Any],
    ) -> Path:
        """Atomically write an asset's metadata envelope as JSON.

        The document lands at ``{base}/{concept_id}/metadata/{asset}.json``
        (``.json`` appended only when not already present) wrapped in an
        envelope recording the asset name and a timezone-aware UTC write
        timestamp. The concept tree is initialized automatically if missing.

        Args:
            concept_id: Safe slug naming the concept.
            asset_filename: Bare filename of the asset the metadata describes.
            metadata_payload: JSON-serializable payload to store.

        Returns:
            Path of the written JSON document.

        Raises:
            FileTreeError: Invalid identifiers or a filesystem/serialization
                failure.
        """
        target = self._metadata_path(concept_id, asset_filename)
        self.initialize_concept_tree(concept_id)
        envelope = {
            "asset": asset_filename,
            "written_at": datetime.now(UTC).isoformat(),
            "payload": metadata_payload,
        }
        try:
            atomic_write_json(target, envelope)
        except (OSError, TypeError, ValueError) as exc:
            raise FileTreeError(
                f"cannot write metadata for {asset_filename!r} in concept {concept_id!r}: {exc}"
            ) from exc
        logger.debug("wrote asset metadata %s", target)
        return target

    def read_asset_metadata(self, concept_id: str, asset_filename: str) -> dict[str, Any]:
        """Read back a previously written metadata envelope.

        Args:
            concept_id: Safe slug naming the concept.
            asset_filename: Asset filename as passed to
                :meth:`write_asset_metadata`.

        Returns:
            The stored envelope (``asset`` / ``written_at`` / ``payload``).

        Raises:
            FileTreeError: Invalid identifiers, no such document, or a
                corrupt/unreadable file.
        """
        target = self._metadata_path(concept_id, asset_filename)
        if not target.is_file():
            raise FileTreeError(
                f"no metadata for asset {asset_filename!r} in concept {concept_id!r} ({target})"
            )
        try:
            document = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileTreeError(f"cannot read metadata document {target}: {exc}") from exc
        if not isinstance(document, dict):
            raise FileTreeError(f"metadata document {target} is not a JSON object")
        return document

    # ----------------------------------------------------------------- listing

    def list_concepts(self) -> list[str]:
        """Return the sorted names of all concept directories under the base."""
        if not self.base_dir.is_dir():
            return []
        return sorted(entry.name for entry in self.base_dir.iterdir() if entry.is_dir())

    def list_assets(self, concept_id: str) -> list[str]:
        """Return sorted asset names with metadata documents for a concept.

        Names are the ``.json`` stems, i.e. the original asset filenames as
        passed to :meth:`write_asset_metadata`. A concept whose tree (or
        metadata directory) does not exist yet lists as empty.

        Raises:
            FileTreeError: Invalid ``concept_id``.
        """
        metadata_dir = self._concept_root(concept_id) / _METADATA_DIR
        if not metadata_dir.is_dir():
            return []
        return sorted(entry.stem for entry in metadata_dir.glob("*.json") if entry.is_file())

    # ----------------------------------------------------------------- helpers

    def _concept_root(self, concept_id: str) -> Path:
        """Validate ``concept_id`` and return its root directory path."""
        if not isinstance(concept_id, str) or not _SLUG_RE.fullmatch(concept_id):
            raise FileTreeError(
                f"concept_id {concept_id!r} is not a safe slug "
                "(expected ^[A-Za-z0-9][A-Za-z0-9._-]*$ with no path separators)"
            )
        return self.base_dir / concept_id

    def _metadata_path(self, concept_id: str, asset_filename: str) -> Path:
        """Validate identifiers and return the metadata document path."""
        root = self._concept_root(concept_id)
        self._validate_asset_filename(asset_filename)
        json_name = (
            asset_filename if asset_filename.endswith(".json") else f"{asset_filename}.json"
        )
        return root / _METADATA_DIR / json_name

    @staticmethod
    def _validate_asset_filename(asset_filename: str) -> None:
        """Reject filenames that could escape the metadata directory."""
        if (
            not isinstance(asset_filename, str)
            or not asset_filename
            or asset_filename in (".", "..")
            or "/" in asset_filename
            or "\\" in asset_filename
            or "\x00" in asset_filename
        ):
            raise FileTreeError(
                f"asset_filename {asset_filename!r} is invalid: must be a bare "
                "filename with no path separators and not '..'"
            )
