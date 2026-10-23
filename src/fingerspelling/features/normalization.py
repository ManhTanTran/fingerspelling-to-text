"""Notebook-compatible FSboard hand-pose normalization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_hand(pose: ArrayLike) -> NDArray[np.float32]:
    """Normalize ``(T, 21, 3)`` poses into the checkpoint's 63D features.

    Mirrors notebook cell 33 / cell 40 exactly: translate each frame by
    landmark 0, divide by the 3D distance from landmark 0 to landmark 9 with
    a minimum scale of 1.0, clip coordinates to [-5, 5], flatten landmark-major
    to 63 dimensions, and return contiguous float32 data.
    """
    array = np.asarray(pose, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (21, 3):
        raise ValueError(f"pose must have shape [time, 21, 3], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("pose must contain at least one frame")
    if not np.isfinite(array).all():
        raise ValueError("pose contains NaN or infinite coordinates")

    wrist = array[:, 0:1, :]
    scale = np.linalg.norm(array[:, 9, :] - array[:, 0, :], axis=1)
    scale = np.maximum(scale, 1.0)
    normalized = (array - wrist) / scale[:, None, None]
    normalized = np.clip(normalized, -5.0, 5.0)
    return np.ascontiguousarray(normalized.reshape(len(array), 63), dtype=np.float32)

