"""Shared filesystem helpers.

Atomic writes follow the ``.tmp`` + :func:`os.replace` pattern. The temp file
is created in the *destination directory* (not the system temp dir) because
``os.replace`` is only atomic within a single filesystem, and it is fsynced
before the rename so a crash cannot leave a truncated file behind the final
name.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def _flush_to_stable_storage(fileno: int) -> None:
    """Force written bytes to stable storage.

    On Darwin, ``fsync`` only pushes data to the drive, not through its write
    cache; Apple documents ``fcntl(F_FULLFSYNC)`` as the real barrier. Fall
    back to plain fsync if the filesystem rejects it (e.g. SMB mounts).
    """
    if sys.platform == "darwin":
        import fcntl

        try:
            fcntl.fcntl(fileno, fcntl.F_FULLFSYNC)
            return
        except OSError:
            pass
    os.fsync(fileno)


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Atomically write ``payload`` to ``path``; returns ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            _flush_to_stable_storage(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    """Atomically write ``text`` to ``path``; returns ``path``."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, payload: Any, indent: int = 2) -> Path:
    """Atomically serialize ``payload`` as pretty-printed JSON at ``path``."""
    text = json.dumps(payload, indent=indent, ensure_ascii=False, sort_keys=True)
    return atomic_write_text(path, text + "\n")
