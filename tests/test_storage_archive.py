import hashlib
import zipfile

import pytest

from fingerspelling.storage.archive import extract_npz_member


def test_extract_npz_member_streams_nested_member_and_records_integrity(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    payload = b"processed test artifact" * 4096
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nested/splits/fsboard_test_proc_clean.npz", payload)
        archive.writestr("nested/splits/train.npz", b"train")

    output = tmp_path / "cache" / "fsboard_test_proc_clean.npz"
    result = extract_npz_member(archive_path, "fsboard_test_proc_clean.npz", output)

    assert output.read_bytes() == payload
    assert result["member"] == "nested/splits/fsboard_test_proc_clean.npz"
    assert result["size_bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["crc32"]


def test_extract_npz_member_requires_unambiguous_member(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train/fsboard_test_proc_clean.npz", b"train")
        archive.writestr("test/fsboard_test_proc_clean.npz", b"test")

    with pytest.raises(ValueError, match="ambiguous"):
        extract_npz_member(
            archive_path,
            "fsboard_test_proc_clean.npz",
            tmp_path / "output.npz",
        )


def test_extract_npz_member_does_not_replace_source_archive(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("fsboard_test_proc_clean.npz", b"npz")

    with pytest.raises(ValueError, match="must not overwrite"):
        extract_npz_member(archive_path, "fsboard_test_proc_clean.npz", archive_path)
