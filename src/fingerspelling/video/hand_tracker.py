"""MediaPipe Tasks hand tracking with a stable signing-hand selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HandDetection:
    """One selected hand's normalized MediaPipe landmarks for a video frame."""

    landmarks: NDArray[np.float32]
    side: str
    score: float
    timestamp_ms: int


class SigningHandTracker:
    """Run MediaPipe VIDEO mode and lock onto one hand for the whole clip.

    With no forced side, the first detected hand with the highest handedness
    score is selected. Later frames only accept detections with that same side;
    the tracker never silently switches hands when the selected hand disappears.
    """

    SIDES = {"left", "right"}

    def __init__(
        self,
        model_asset_path: str | Path,
        *,
        forced_side: str | None = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_asset_path = Path(model_asset_path)
        if not self.model_asset_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe HandLandmarker model not found: {self.model_asset_path}. "
                "Pass --hand-model with a local hand_landmarker.task asset."
            )
        if forced_side is not None:
            forced_side = forced_side.lower()
            if forced_side not in self.SIDES:
                raise ValueError("forced_side must be 'left', 'right', or None")
        if num_hands < 1:
            raise ValueError("num_hands must be positive")

        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "video inference requires optional dependencies; install with "
                "python -m pip install -e '.[video]'"
            ) from exc

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_asset_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._cv2 = cv2
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.forced_side = forced_side
        self.locked_side: str | None = forced_side
        self.processed_frames = 0
        self.detected_frames = 0
        self._last_timestamp_ms: int | None = None

    @property
    def detection_rate(self) -> float:
        if self.processed_frames == 0:
            return 0.0
        return self.detected_frames / self.processed_frames

    def detect(self, bgr_frame: Any, timestamp_ms: int) -> HandDetection | None:
        """Return selected normalized [21, 3] landmarks or None for a miss."""
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be nonnegative")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("MediaPipe video timestamps must increase strictly")
        frame = np.asarray(bgr_frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("video frame must be a nonempty BGR image with 3 channels")

        self._last_timestamp_ms = timestamp_ms
        self.processed_frames += 1
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        selected = self._select_detection(result, timestamp_ms)
        if selected is not None:
            self.detected_frames += 1
        return selected

    def _select_detection(self, result: Any, timestamp_ms: int) -> HandDetection | None:
        landmarks_by_hand = getattr(result, "hand_landmarks", None) or []
        handedness_by_hand = getattr(result, "handedness", None) or []
        candidates: list[tuple[float, str, NDArray[np.float32]]] = []

        for index, landmarks in enumerate(landmarks_by_hand):
            if index >= len(handedness_by_hand) or not handedness_by_hand[index]:
                continue
            category = handedness_by_hand[index][0]
            side = str(
                getattr(category, "category_name", "")
                or getattr(category, "display_name", "")
            ).lower()
            if side not in self.SIDES:
                continue
            if self.locked_side is not None and side != self.locked_side:
                continue
            coords = np.asarray(
                [[point.x, point.y, point.z] for point in landmarks], dtype=np.float32
            )
            if coords.shape != (21, 3) or not np.isfinite(coords).all():
                continue
            score = float(getattr(category, "score", 0.0))
            if not np.isfinite(score):
                continue
            candidates.append((score, side, coords))

        if not candidates:
            return None
        score, side, coords = max(candidates, key=lambda item: item[0])
        if self.locked_side is None:
            self.locked_side = side
        return HandDetection(
            landmarks=np.ascontiguousarray(coords),
            side=side,
            score=score,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "SigningHandTracker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
