"""Unit tests for :mod:`akio_studio._io` atomic write helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import akio_studio._io as io_mod
from akio_studio._io import atomic_write_bytes, atomic_write_json, atomic_write_text


class TestAtomicWriteJson:
    """Write-then-replace semantics: never a partial file behind the name."""

    def test_round_trip_with_trailing_newline(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.json"
        payload = {"b": 2, "a": [1, 2, 3]}
        assert atomic_write_json(target, payload) == target
        text = target.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text) == payload
        assert list(tmp_path.glob("*.tmp")) == []

    def test_unserializable_payload_leaves_nothing_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.json"
        with pytest.raises(TypeError):
            atomic_write_json(target, {"oops": {1, 2}})  # sets are not JSON
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []  # no target, no orphan tmp

    def test_failure_mid_write_cleans_tmp_and_preserves_old_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "doc.json"
        atomic_write_json(target, {"version": 1})

        def _explode(fileno: int) -> None:
            raise OSError("simulated disk failure before rename")

        monkeypatch.setattr(io_mod, "_flush_to_stable_storage", _explode)
        with pytest.raises(OSError, match="simulated disk failure"):
            atomic_write_json(target, {"version": 2})
        # The previous content survives untouched and the tmp file is gone.
        assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
        assert list(tmp_path.glob("*.tmp*")) == []

    def test_overwrite_replaces_content(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.json"
        atomic_write_json(target, {"version": 1})
        atomic_write_json(target, {"version": 2})
        assert json.loads(target.read_text(encoding="utf-8")) == {"version": 2}


class TestAtomicWriteBytesAndText:
    """Byte/text variants share the same atomic machinery."""

    def test_bytes_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "blob.bin"
        assert atomic_write_bytes(target, b"\x00\x01payload") == target
        assert target.read_bytes() == b"\x00\x01payload"

    def test_text_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        atomic_write_text(target, "こんにちは\n")
        assert target.read_text(encoding="utf-8") == "こんにちは\n"
