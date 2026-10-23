import numpy as np
import pytest

from fingerspelling.features.normalization import normalize_hand


def test_normalization_matches_wrist_anchor_3d_scale_and_clip():
    pose = np.zeros((1, 21, 3), dtype=np.float32)
    pose[0, 9] = [0.0, 0.0, 2.0]
    pose[0, 1] = [20.0, 0.0, 0.0]

    normalized = normalize_hand(pose)

    assert normalized.shape == (1, 63)
    assert normalized.dtype == np.float32
    assert normalized[0, 3] == pytest.approx(5.0)
    assert normalized[0, 27:30] == pytest.approx([0.0, 0.0, 1.0])
    assert np.isfinite(normalized).all()


def test_normalization_uses_minimum_scale_one():
    pose = np.zeros((1, 21, 3), dtype=np.float32)
    pose[0, 9, 2] = 0.25
    pose[0, 1, 0] = 0.5

    normalized = normalize_hand(pose)
    assert normalized[0, 3] == pytest.approx(0.5)


@pytest.mark.parametrize("pose", [np.zeros((0, 21, 3)), np.zeros((2, 20, 3))])
def test_normalization_rejects_empty_or_wrong_shape(pose):
    with pytest.raises(ValueError):
        normalize_hand(pose)


def test_normalization_rejects_non_finite_values():
    pose = np.zeros((1, 21, 3), dtype=np.float32)
    pose[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        normalize_hand(pose)

