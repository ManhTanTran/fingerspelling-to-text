import hashlib

import pytest

from fingerspelling.storage.integrity import copy_verified, sha256_file


def test_copy_verified_checks_contents_and_preserves_source(tmp_path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "cache" / "source.bin"
    payload = bytes(range(256)) * 1024
    source.write_bytes(payload)

    result = copy_verified(source, destination, chunk_size=997)

    expected_hash = hashlib.sha256(payload).hexdigest()
    assert result["verified"] is True
    assert result["copied"] is True
    assert result["sha256"] == expected_hash
    assert sha256_file(destination) == expected_hash
    assert source.read_bytes() == payload


def test_copy_verified_refuses_different_existing_destination_without_overwrite(tmp_path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"new artifact")
    destination.write_bytes(b"old artifact")

    with pytest.raises(FileExistsError, match="different contents"):
        copy_verified(source, destination)

    assert destination.read_bytes() == b"old artifact"
    result = copy_verified(source, destination, overwrite=True)
    assert result["verified"] is True
    assert destination.read_bytes() == source.read_bytes()


def test_copy_verified_same_path_is_a_verified_noop(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"artifact")

    result = copy_verified(source, source)

    assert result["copied"] is False
    assert result["verified"] is True
