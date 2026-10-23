import numpy as np
import pytest

from fingerspelling.data.fsboard_dataset import load_processed_npz
from scripts.export_npz_subset import export_subset


def _make_processed_npz(path, sample_count=40):
    left = np.empty(sample_count, dtype=object)
    right = np.empty(sample_count, dtype=object)
    hands = np.empty(sample_count, dtype=np.int64)
    words = np.empty(sample_count, dtype=object)
    for index in range(sample_count):
        pose = np.full((8, 21, 3), index / 100, dtype=np.float32)
        hands[index] = index % 2
        left[index] = {0: pose} if hands[index] == 1 else None
        right[index] = {0: pose} if hands[index] == 0 else None
        words[index] = f"WORD{index}"
    np.savez_compressed(
        path,
        left_hand_poses=left,
        right_hand_poses=right,
        signing_hand=hands,
        words=words,
    )


def test_export_subset_writes_deterministic_reloadable_24_sample_artifact(tmp_path):
    source = tmp_path / "processed.npz"
    first_output = tmp_path / "subset-a.npz"
    second_output = tmp_path / "subset-b.npz"
    _make_processed_npz(source)
    original_bytes = source.read_bytes()

    first = export_subset(source, first_output, count=24, seed=12, trusted_local=True)
    second = export_subset(source, second_output, count=24, seed=12, trusted_local=True)
    examples, stats = load_processed_npz(
        first_output, trusted_local=True, filter_ctc_impossible=False
    )

    assert len(examples) == 24
    assert stats["kept"] == 24
    assert first["source_indices"] == second["source_indices"]
    assert first_output.read_bytes() == second_output.read_bytes()
    assert source.read_bytes() == original_bytes
    assert first["sample_count"] == 24


@pytest.mark.parametrize("count", [15, 33])
def test_export_subset_rejects_sizes_outside_16_to_32(tmp_path, count):
    source = tmp_path / "processed.npz"
    _make_processed_npz(source)

    with pytest.raises(ValueError, match="between 16 and 32"):
        export_subset(source, tmp_path / "subset.npz", count=count, trusted_local=True)
