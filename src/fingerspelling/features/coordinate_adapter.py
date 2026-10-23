"""Adapter from MediaPipe normalized landmarks to FSboard pixel coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


COORDINATE_ADAPTERS = ("pixel_xy_preserve_z", "normalized_xyz")


def mediapipe_to_fsboard_coordinates(
    landmarks: ArrayLike,
    image_width: int,
    image_height: int,
) -> NDArray[np.float32]:
    """Scale normalized MediaPipe x/y to image pixels; preserve z.

    The saved notebook sample has x/y magnitudes in the hundreds/thousands
    while z remains small (approximately [-0.22, 0.03]). This candidate adapter
    therefore maps x to ``x * width`` and y to ``y * height`` while preserving
    MediaPipe's relative z. MediaPipe documents z as roughly the same scale as
    normalized x, so scaling x/y but not z intentionally follows the mixed-scale
    convention observed in the FSboard sample; it is not a verified physical
    coordinate transform. It does not mirror, center, or normalize the points;
    ``normalize_hand`` then applies notebook-compatible wrist translation and
    scale normalization. A same-sample FSboard comparison is still required.

    Input may be one frame ``[21, 3]`` or a sequence ``[T, 21, 3]``. The caller
    must supply coordinates from the unmirrored image used for inference.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    array = np.asarray(landmarks, dtype=np.float32)
    if array.ndim not in (2, 3) or array.shape[-2:] != (21, 3):
        raise ValueError(
            f"landmarks must have shape [21, 3] or [time, 21, 3], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("landmarks contain NaN or infinite coordinates")

    adapted = array.copy()
    adapted[..., 0] *= image_width
    adapted[..., 1] *= image_height
    return np.ascontiguousarray(adapted, dtype=np.float32)


def adapt_mediapipe_coordinates(
    landmarks: ArrayLike,
    image_width: int,
    image_height: int,
    *,
    adapter: str = "pixel_xy_preserve_z",
) -> NDArray[np.float32]:
    """Apply an explicitly selected MediaPipe coordinate convention.

    ``pixel_xy_preserve_z`` is the existing candidate adapter. ``normalized_xyz``
    keeps MediaPipe's normalized coordinates unchanged. Neither option is
    asserted to match the FSboard training representation without a matched
    source-video comparison.
    """
    if adapter == "pixel_xy_preserve_z":
        return mediapipe_to_fsboard_coordinates(landmarks, image_width, image_height)
    if adapter == "normalized_xyz":
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        array = np.asarray(landmarks, dtype=np.float32)
        if array.ndim not in (2, 3) or array.shape[-2:] != (21, 3):
            raise ValueError(
                f"landmarks must have shape [21, 3] or [time, 21, 3], got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("landmarks contain NaN or infinite coordinates")
        return np.ascontiguousarray(array.copy(), dtype=np.float32)
    raise ValueError(
        f"unsupported coordinate adapter {adapter!r}; choose one of {COORDINATE_ADAPTERS}"
    )

