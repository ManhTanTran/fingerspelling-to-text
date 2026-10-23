import numpy as np
import pytest

from fingerspelling.data.fsboard_dataset import (
    FSboardDataset,
    FSboardExample,
    collate_batch,
    load_processed_npz,
)
from fingerspelling.data.vocabulary import Vocabulary


def _entry(pose):
    return {0: pose}


def test_loader_requires_explicit_trust_for_pickle_backed_npz(tmp_path):
    path = tmp_path / "processed.npz"
    np.savez(path)
    with pytest.raises(ValueError, match="trusted_local"):
        load_processed_npz(path)


def test_loader_selects_signing_hand_and_filters_ctc_impossible(tmp_path):
    pose = np.zeros((2, 21, 3), dtype=np.float32)
    right = np.empty(2, dtype=object)
    left = np.empty(2, dtype=object)
    right[:] = [_entry(np.zeros((0, 21, 3), dtype=np.float32)), _entry(pose)]
    left[:] = [_entry(pose), _entry(np.zeros((0, 21, 3), dtype=np.float32))]
    path = tmp_path / "processed.npz"
    np.savez(
        path,
        right_hand_poses=right,
        left_hand_poses=left,
        signing_hand=np.array([1, 0], dtype=object),
        words=np.array(["ab", "aa"], dtype=object),
    )

    examples, stats = load_processed_npz(path, trusted_local=True)

    assert len(examples) == 1
    assert examples[0].source_index == 0
    assert examples[0].signing_hand == 1
    assert stats["too_short_for_ctc"] == 1
    assert stats["kept"] == 1


def test_dataset_and_collate_return_notebook_batch_layout():
    pose = np.zeros((3, 21, 3), dtype=np.float32)
    examples = [
        FSboardExample(0, pose, "ab", 0),
        FSboardExample(1, pose[:2], "a", 1),
    ]
    vocabulary = Vocabulary.from_examples(examples)
    dataset = FSboardDataset(examples, vocabulary, feature_mode="motion")

    x, targets, input_lengths, target_lengths, texts = collate_batch(
        [dataset[0], dataset[1]]
    )

    assert x.shape == (2, 3, 126)
    assert targets.tolist() == [vocabulary.char_to_idx["a"], vocabulary.char_to_idx["b"], vocabulary.char_to_idx["a"]]
    assert input_lengths.tolist() == [3, 2]
    assert target_lengths.tolist() == [2, 1]
    assert texts == ["ab", "a"]

