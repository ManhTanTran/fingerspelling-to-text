"""Extract one trusted processed NPZ member from the source ZIP archive."""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def extract_npz_member(
    archive_path: str | Path,
    member_name: str,
    destination: str | Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Stream one named ZIP member to a local file and verify its size/hash/CRC.

    ``member_name`` may be the full archive path or a unique basename. ZIP
    traversal paths are never materialized: only the requested member's bytes
    are streamed into the caller-selected destination.
    """
    source = Path(archive_path).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source archive not found: {source}")
    if source == target:
        raise ValueError("NPZ output must not overwrite its source archive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    requested = member_name.replace("\\", "/").lstrip("/")
    with zipfile.ZipFile(source, "r") as archive:
        exact = [info for info in archive.infolist() if info.filename == requested]
        matches = exact or [
            info for info in archive.infolist()
            if PurePosixPath(info.filename).name == PurePosixPath(requested).name
        ]
        if not matches:
            available = [
                info.filename for info in archive.infolist()
                if info.filename.lower().endswith(".npz")
            ]
            raise FileNotFoundError(
                f"ZIP member {member_name!r} not found; NPZ members: {available[:20]}"
            )
        if len(matches) != 1:
            raise ValueError(
                f"ZIP member {member_name!r} is ambiguous: {[info.filename for info in matches]}"
            )
        info = matches[0]
        if info.is_dir():
            raise ValueError(f"requested ZIP member is a directory: {info.filename}")

        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent
        )
        temp_path = Path(temp_name)
        digest = hashlib.sha256()
        bytes_written = 0
        try:
            with os.fdopen(fd, "wb") as output, archive.open(info, "r") as member:
                while True:
                    chunk = member.read(chunk_size)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            # ZipExtFile checks the member CRC when read through EOF.
            if bytes_written != info.file_size:
                raise IOError(
                    f"extracted member size mismatch: expected {info.file_size}, got {bytes_written}"
                )
            if target.exists() and not overwrite:
                raise FileExistsError(f"extraction destination already exists: {target}")
            os.replace(temp_path, target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        return {
            "archive": str(source),
            "member": info.filename,
            "destination": str(target),
            "size_bytes": bytes_written,
            "sha256": digest.hexdigest(),
            "crc32": f"{info.CRC:08x}",
            "compressed_size_bytes": info.compress_size,
            "compression_type": info.compress_type,
        }
