"""Streaming checksums and atomic, verified file copies for large artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Copy one file without mutating its source, verify SHA-256, then atomically replace.

    A different existing destination is protected unless ``overwrite`` is set.
    Partial files stay beside the destination and are deleted after failures.
    """
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"copy source does not exist: {src}")
    if src == dst:
        digest = sha256_file(src, chunk_size=chunk_size)
        return {
            "source": str(src), "destination": str(dst),
            "size_bytes": src.stat().st_size, "sha256": digest,
            "copied": False, "verified": True,
        }

    source_hash: str | None = None
    if dst.exists():
        if not dst.is_file():
            raise IsADirectoryError(f"copy destination is not a file: {dst}")
        source_hash = sha256_file(src, chunk_size=chunk_size)
        destination_hash = sha256_file(dst, chunk_size=chunk_size)
        if source_hash == destination_hash:
            return {
                "source": str(src), "destination": str(dst),
                "size_bytes": src.stat().st_size, "sha256": source_hash,
                "copied": False, "verified": True,
            }
        if not overwrite:
            raise FileExistsError(
                f"destination has different contents: {dst}; pass overwrite=True to replace it"
            )

    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".part", dir=dst.parent)
    temp_path = Path(temp_name)
    transfer_digest = hashlib.sha256()
    size_bytes = 0
    try:
        # Open the destination descriptor first so it is still closed if opening
        # the source fails after mkstemp succeeds.
        with os.fdopen(fd, "wb") as target_stream, src.open("rb") as source_stream:
            for chunk in iter(lambda: source_stream.read(chunk_size), b""):
                target_stream.write(chunk)
                transfer_digest.update(chunk)
                size_bytes += len(chunk)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        copied_hash = transfer_digest.hexdigest()
        if source_hash is not None and copied_hash != source_hash:
            raise IOError("source changed while it was being copied; checksum differs")
        verified_hash = sha256_file(temp_path, chunk_size=chunk_size)
        if copied_hash != verified_hash:
            raise IOError("temporary copy checksum verification failed")
        os.replace(temp_path, dst)
        return {
            "source": str(src), "destination": str(dst),
            "size_bytes": size_bytes, "sha256": verified_hash,
            "copied": True, "verified": True,
        }
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise
