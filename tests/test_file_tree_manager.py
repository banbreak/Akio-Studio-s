"""Unit tests for :mod:`akio_studio.file_tree_manager`."""

from __future__ import annotations

from pathlib import Path

import pytest

from akio_studio.exceptions import FileTreeError
from akio_studio.file_tree_manager import CONCEPT_SUBDIRS, ProductionFileTreeManager


class TestTreeCreation:
    """Idempotent concept-tree initialization."""

    def test_tree_created_with_all_subdirs(self, tmp_path: Path) -> None:
        manager = ProductionFileTreeManager(base_dir=tmp_path)
        tree = manager.initialize_concept_tree("ronin-arc-001")
        assert tree["root"] == tmp_path / "ronin-arc-001"
        for subdir in CONCEPT_SUBDIRS:
            assert tree[subdir].is_dir()
            assert tree[subdir].parent == tree["root"]

    def test_initialization_is_idempotent(self, tmp_path: Path) -> None:
        manager = ProductionFileTreeManager(base_dir=tmp_path)
        first = manager.initialize_concept_tree("ronin-arc-001")
        (first["metadata"] / "keep.json").write_text("{}", encoding="utf-8")
        second = manager.initialize_concept_tree("ronin-arc-001")
        assert first == second
        assert (second["metadata"] / "keep.json").exists()  # nothing clobbered

    @pytest.mark.parametrize("slug", ["../escape", "a/b", "", ".hidden", "a\\b"])
    def test_invalid_concept_slug_raises(self, tmp_path: Path, slug: str) -> None:
        manager = ProductionFileTreeManager(base_dir=tmp_path)
        with pytest.raises(FileTreeError):
            manager.initialize_concept_tree(slug)


class TestAssetMetadata:
    """Atomic metadata write + read round-trip."""

    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        manager = ProductionFileTreeManager(base_dir=tmp_path)
        payload = {"seed": 41, "prompt_hash": "abc123", "quality_gate": {"pulid": 0.6}}
        target = manager.write_asset_metadata("concept-a", "shot012_seed41.mp4", payload)
        assert target == tmp_path / "concept-a" / "metadata" / "shot012_seed41.mp4.json"
        envelope = manager.read_asset_metadata("concept-a", "shot012_seed41.mp4")
        assert envelope["payload"] == payload
        assert envelope["asset"] == "shot012_seed41.mp4"
        assert "written_at" in envelope
        # Atomic writer must leave no .tmp siblings behind.
        assert list(target.parent.glob("*.tmp")) == []

    def test_json_suffix_not_doubled(self, tmp_path: Path) -> None:
        manager = ProductionFileTreeManager(base_dir=tmp_path)
        target = manager.write_asset_metadata("concept-a", "notes.json", {"k": 1})
        assert target.name == "notes.json"
        assert manager.read_asset_metadata("concept-a", "notes.json")["payload"] == {"k": 1}

    def test_read_missing_metadata_raises(self, tmp_path: Path) -> None:
        manager = ProductionFileTreeManager(base_dir=tmp_path)
        manager.initialize_concept_tree("concept-a")
        with pytest.raises(FileTreeError):
            manager.read_asset_metadata("concept-a", "ghost.mp4")

    @pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "nul\x00byte"])
    def test_invalid_asset_filename_raises(self, tmp_path: Path, name: str) -> None:
        manager = ProductionFileTreeManager(base_dir=tmp_path)
        with pytest.raises(FileTreeError):
            manager.write_asset_metadata("concept-a", name, {})

    def test_unserializable_payload_raises_file_tree_error(self, tmp_path: Path) -> None:
        manager = ProductionFileTreeManager(base_dir=tmp_path)
        with pytest.raises(FileTreeError):
            manager.write_asset_metadata("concept-a", "bad.mp4", {"oops": {1, 2}})


class TestListing:
    """Concept and asset listings."""

    def test_list_concepts_and_assets(self, tmp_path: Path) -> None:
        manager = ProductionFileTreeManager(base_dir=tmp_path)
        assert manager.list_concepts() == []
        manager.initialize_concept_tree("b-concept")
        manager.initialize_concept_tree("a-concept")
        assert manager.list_concepts() == ["a-concept", "b-concept"]
        assert manager.list_assets("a-concept") == []
        manager.write_asset_metadata("a-concept", "shot2.mp4", {})
        manager.write_asset_metadata("a-concept", "shot1.mp4", {})
        assert manager.list_assets("a-concept") == ["shot1.mp4", "shot2.mp4"]
