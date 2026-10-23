"""Convert a video to notebook-shaped hand or hand-plus-motion features."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from fingerspelling.features.coordinate_adapter import adapt_mediapipe_coordinates
from fingerspelling.features.motion import add_motion_features
from fingerspelling.features.normalization import normalize_hand

from .decoder import VideoDecoder, VideoMetadata
from .hand_tracker import SigningHandTracker


@dataclass(frozen=True)
class VideoFeatureTimings:
    decode_ms: float
    tracking_ms: float
    coordinate_adapter_ms: float
    feature_ms: float
    total_ms: float
    hand_tracker_init_ms: float = 0.0


@dataclass(frozen=True)
class VideoFeatures:
    pose: NDArray[np.float32]
    features: NDArray[np.float32]
    frame_indices: NDArray[np.int64]
    detection_rate: float
    selected_hand: str
    metadata: VideoMetadata
    timings: VideoFeatureTimings
    coordinate_adapter: str


def video_to_features(
    video_path: str | Path,
    hand_model_path: str | Path,
    *,
    input_dim: int = 126,
    forced_side: str | None = None,
    coordinate_adapter: str = "pixel_xy_preserve_z",
    max_duration_seconds: float = 120.0,
    max_upload_bytes: int = 100_000_000,
    max_pixels: int = 16_777_216,
    decoder_factory: Callable[..., VideoDecoder] = VideoDecoder,
    tracker_factory: Callable[..., SigningHandTracker] = SigningHandTracker,
) -> VideoFeatures:
    """Decode and track a clip; missing hand frames are omitted like FSboard rows.

    ``frame_indices`` preserves the decoded frame number for diagnostics. As in
    the notebook, the model sees the ordered rows where a pose exists and motion
    is computed between adjacent stored poses; no interpolation is performed.
    """
    if input_dim not in (63, 126):
        raise ValueError("input_dim must match a supported checkpoint: 63 or 126")

    total_start = time.perf_counter()
    decoder = decoder_factory(
        video_path,
        max_duration_seconds=max_duration_seconds,
        max_upload_bytes=max_upload_bytes,
        max_pixels=max_pixels,
    )
    tracker_start = time.perf_counter()
    tracked = tracker_factory(hand_model_path, forced_side=forced_side)
    tracker_init_ms = (time.perf_counter() - tracker_start) * 1000
    poses: list[NDArray[np.float32]] = []
    frame_indices: list[int] = []
    decode_ms = 0.0
    tracking_ms = 0.0
    coordinate_adapter_ms = 0.0
    try:
        with decoder, tracked:
            frames = iter(decoder)
            try:
                while True:
                    stage_start = time.perf_counter()
                    try:
                        frame_index, frame = next(frames)
                    except StopIteration:
                        break
                    decode_ms += (time.perf_counter() - stage_start) * 1000

                    timestamp_ms = int(round(frame_index * 1000.0 / decoder.metadata.fps))
                    stage_start = time.perf_counter()
                    detection = tracked.detect(frame, timestamp_ms)
                    tracking_ms += (time.perf_counter() - stage_start) * 1000
                    if detection is None:
                        continue
                    stage_start = time.perf_counter()
                    adapted = adapt_mediapipe_coordinates(
                        detection.landmarks,
                        decoder.metadata.width,
                        decoder.metadata.height,
                        adapter=coordinate_adapter,
                    )
                    coordinate_adapter_ms += (time.perf_counter() - stage_start) * 1000
                    poses.append(adapted)
                    frame_indices.append(frame_index)
            finally:
                # Closing the generator finalizes decoder metadata on all exits.
                close = getattr(frames, "close", None)
                if close is not None:
                    close()
            detection_rate = tracked.detection_rate
            selected_hand = tracked.locked_side or tracked.forced_side or "unknown"
    finally:
        total_ms = (time.perf_counter() - total_start) * 1000

    if not poses:
        raise ValueError(
            "MediaPipe found no usable signing-hand landmarks in the video; "
            "try a clearer clip or choose --hand left/right explicitly"
        )

    feature_start = time.perf_counter()
    pose_array = np.ascontiguousarray(np.stack(poses), dtype=np.float32)
    feature_array = normalize_hand(pose_array) if input_dim == 63 else add_motion_features(pose_array)
    feature_ms = (time.perf_counter() - feature_start) * 1000
    # Add feature calculation to total processing time; total excludes checkpoint load.
    total_ms += feature_ms
    metadata = decoder.metadata
    if metadata is None:
        raise RuntimeError("video metadata unavailable after decoding")
    return VideoFeatures(
        pose=pose_array,
        features=feature_array,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        detection_rate=detection_rate,
        selected_hand=selected_hand,
        metadata=metadata,
        timings=VideoFeatureTimings(
            decode_ms,
            tracking_ms,
            coordinate_adapter_ms,
            feature_ms,
            total_ms,
            tracker_init_ms,
        ),
        coordinate_adapter=coordinate_adapter,
    )
