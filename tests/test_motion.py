import numpy as np
import pytest

from fingerspelling.features.motion import add_motion_features
from fingerspelling.features.normalization import normalize_hand


def test_motion_features_are_pose_then_clipped_adjacent_differences():
    pose = np.zeros((3, 21, 3), dtype=np.float32)
    pose[:, 9, 2] = 2.0
    pose[1, 4, 0] = 20.0
    pose[2, 4, 0] = -20.0

    features = add_motion_features(pose)
    hand = normalize_hand(pose)
    expected_velocity = np.zeros_like(hand)
    expected_velocity[1:] = np.clip(hand[1:] - hand[:-1], -5.0, 5.0)

    assert features.shape == (3, 126)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(features[:, :63], hand)
    np.testing.assert_array_equal(features[:, 63:], expected_velocity)
    np.testing.assert_array_equal(features[0, 63:], np.zeros(63, dtype=np.float32))
    assert np.isfinite(features).all()


def test_motion_rejects_non_finite_pose():
    pose = np.zeros((2, 21, 3), dtype=np.float32)
    pose[0, 3, 1] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        add_motion_features(pose)

