"""Unit tests for :mod:`akio_studio.actor_sdk_exporter`."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from akio_studio.actor_sdk_exporter import SyntheticActorSDKExporter
from akio_studio.exceptions import ActorExportError, MissingAssetError


def _make_assets(tmp_path: Path) -> dict[str, Path]:
    """Create a small fake weights file, pose rig, and voice config."""
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    lora = src / "kael_identity.safetensors"
    lora.write_bytes(b"FAKE-TENSOR-\x00\x01\x02" * 300)
    pose = src / "pose_default.json"
    pose.write_text(json.dumps({"rig": "openpose-25", "keyframes": 12}), encoding="utf-8")
    voice = src / "voice_config.json"
    voice.write_text(json.dumps({"engine": "gpt-sovits", "lang": "ja"}), encoding="utf-8")
    return {"lora_weights": lora, "pose_rigs": pose, "voice_model": voice}


def _export(tmp_path: Path) -> tuple[SyntheticActorSDKExporter, Path]:
    """Export a standard three-asset bundle into ``tmp_path/out``."""
    exporter = SyntheticActorSDKExporter(tmp_path / "out")
    bundle = exporter.export_synthetic_actor("ronin_kael", _make_assets(tmp_path))
    return exporter, bundle


class TestExportRoundTrip:
    """Bundle assembly, manifest checksums, and verification."""

    def test_round_trip_and_checksum_verify(self, tmp_path: Path) -> None:
        exporter, bundle = _export(tmp_path)
        assert bundle.name == "ronin_kael.synthetic_actor"
        assert bundle.is_file()
        manifest = exporter.inspect_bundle(bundle)  # re-hashes every entry
        assert manifest["character_id"] == "ronin_kael"
        records = manifest["files"]
        assert [record["category"] for record in records] == [
            "lora_weights",
            "pose_rigs",
            "voice_model",
        ]
        assert all(len(record["sha256"]) == 64 for record in records)
        # No stray .tmp sibling may survive a successful export.
        assert list(bundle.parent.glob("*.tmp")) == []

    def test_manifest_extra_merges_but_identity_is_authoritative(self, tmp_path: Path) -> None:
        exporter = SyntheticActorSDKExporter(tmp_path / "out")
        bundle = exporter.export_synthetic_actor(
            "ronin_kael",
            _make_assets(tmp_path),
            manifest_extra={"character_id": "evil", "power_level": 6, "files": "bogus"},
        )
        manifest = exporter.inspect_bundle(bundle)
        assert manifest["character_id"] == "ronin_kael"  # extra cannot override
        assert manifest["power_level"] == 6
        assert isinstance(manifest["files"], list)  # computed list wins

    def test_compression_types_per_entry(self, tmp_path: Path) -> None:
        _, bundle = _export(tmp_path)
        with zipfile.ZipFile(bundle) as archive:
            stored = archive.getinfo("lora_weights/kael_identity.safetensors")
            assert stored.compress_type == zipfile.ZIP_STORED
            assert archive.getinfo("pose_rigs/pose_default.json").compress_type == (
                zipfile.ZIP_DEFLATED
            )
            assert archive.getinfo("manifest.json").compress_type == zipfile.ZIP_DEFLATED

    def test_tampered_checksum_is_detected(self, tmp_path: Path) -> None:
        exporter, bundle = _export(tmp_path)
        with zipfile.ZipFile(bundle) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            entries = {
                name: archive.read(name)
                for name in archive.namelist()
                if name != "manifest.json"
            }
        manifest["files"][0]["sha256"] = "0" * 64
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name, data in entries.items():
                archive.writestr(name, data)
        with pytest.raises(ActorExportError, match="sha256 mismatch"):
            exporter.inspect_bundle(bundle)


class TestExportValidation:
    """Slug, category, and source-path validation."""

    def test_missing_asset_raises(self, tmp_path: Path) -> None:
        exporter = SyntheticActorSDKExporter(tmp_path / "out")
        with pytest.raises(MissingAssetError):
            exporter.export_synthetic_actor(
                "kael", {"lora_weights": tmp_path / "absent.safetensors"}
            )

    @pytest.mark.parametrize("slug", ["../evil", "a/b", "", ".hidden", "-dash"])
    def test_bad_slug_raises(self, tmp_path: Path, slug: str) -> None:
        exporter = SyntheticActorSDKExporter(tmp_path / "out")
        with pytest.raises(ActorExportError):
            exporter.export_synthetic_actor(slug, _make_assets(tmp_path))

    def test_unknown_category_raises(self, tmp_path: Path) -> None:
        exporter = SyntheticActorSDKExporter(tmp_path / "out")
        assets = _make_assets(tmp_path)
        with pytest.raises(ActorExportError):
            exporter.export_synthetic_actor("kael", {"textures": assets["lora_weights"]})

    def test_empty_asset_map_raises(self, tmp_path: Path) -> None:
        exporter = SyntheticActorSDKExporter(tmp_path / "out")
        with pytest.raises(ActorExportError):
            exporter.export_synthetic_actor("kael", {"lora_weights": []})

    def test_inspect_missing_bundle_raises(self, tmp_path: Path) -> None:
        exporter = SyntheticActorSDKExporter(tmp_path / "out")
        with pytest.raises(ActorExportError):
            exporter.inspect_bundle(tmp_path / "nope.synthetic_actor")
