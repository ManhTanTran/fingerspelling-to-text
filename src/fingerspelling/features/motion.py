"""Notebook-compatible 126D hand-pose and finite-difference features."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .normalization import normalize_hand


def add_motion_features(pose: ArrayLike) -> NDArray[np.float32]:
    """Concatenate normalized pose and clipped frame-to-frame differences.

    The first velocity row is all zero. Later rows are differences between
    adjacent stored pose rows, without timestamp division or interpolation,
    then clipped to [-5, 5], matching notebook cell 40 / cell 42.
    """
    hand = normalize_hand(pose)
    velocity = np.zeros_like(hand)
    velocity[1:] = hand[1:] - hand[:-1]
    velocity = np.clip(velocity, -5.0, 5.0)
    return np.ascontiguousarray(np.concatenate([hand, velocity], axis=1), dtype=np.float32)

