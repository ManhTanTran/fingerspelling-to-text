"""Bounded MP4/WebM frame decoding with OpenCV."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    width: int
    height: int
    reported_frame_count: int
    estimated_duration_seconds: float | None
    decoded_frame_count: int = 0


class VideoDecoder:
    """Stream frames in file order and release capture resources on completion."""

    SUPPORTED_SUFFIXES = {".mp4", ".webm"}

    def __init__(
        self,
        path: str | Path,
        *,
        max_duration_seconds: float = 120.0,
        max_upload_bytes: int = 100_000_000,
        max_pixels: int = 16_777_216,
    ) -> None:
        self.path = Path(path)
        self.max_duration_seconds = float(max_duration_seconds)
        self.max_upload_bytes = int(max_upload_bytes)
        self.max_pixels = int(max_pixels)
        self._cv2 = None
        self._capture = None
        self.metadata: VideoMetadata | None = None

        if self.max_duration_seconds <= 0 or self.max_upload_bytes <= 0 or self.max_pixels <= 0:
            raise ValueError("video limits must be positive")
        if self.path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError("video format must be MP4 or WebM")
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        if self.path.stat().st_size <= 0:
            raise ValueError("video file is empty")
        if self.path.stat().st_size > self.max_upload_bytes:
            raise ValueError(f"video exceeds upload limit of {self.max_upload_bytes} bytes")

    def __enter__(self) -> "VideoDecoder":
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "video decoding requires the optional opencv-python dependency"
            ) from exc
        self._cv2 = cv2
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"cannot open video or codec is unavailable: {self.path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reported_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            capture.release()
            raise
        reported = (
            int(reported_value)
            if math.isfinite(reported_value) and reported_value > 0
            else 0
        )
        if not math.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError("video reports an invalid FPS")
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("video reports invalid frame dimensions")
        if width * height > self.max_pixels:
            capture.release()
            raise ValueError(
                f"video resolution {width}x{height} exceeds limit of "
                f"{self.max_pixels} pixels"
            )
        estimated_duration = reported / fps if reported else None
        if estimated_duration is not None and estimated_duration > self.max_duration_seconds:
            capture.release()
            raise ValueError(
                f"video duration {estimated_duration:.2f}s exceeds limit "
                f"{self.max_duration_seconds:.2f}s"
            )

        self._capture = capture
        self.metadata = VideoMetadata(
            str(self.path.resolve()), fps, width, height, reported, estimated_duration
        )
        return self

    def __iter__(self) -> Iterator[tuple[int, object]]:
        if self._capture is None or self._cv2 is None or self.metadata is None:
            raise RuntimeError("use VideoDecoder inside a with statement")
        frame_limit = math.ceil(self.max_duration_seconds * self.metadata.fps)
        count = 0
        try:
            while count < frame_limit:
                ok, frame = self._capture.read()
                if not ok:
                    if count == 0:
                        raise ValueError("video is empty, corrupt, or cannot be decoded")
                    break
                if (
                    frame is None
                    or getattr(frame, "ndim", 0) != 3
                    or frame.shape[2] != 3
                ):
                    raise ValueError(f"decoded frame {count} must be a 3-channel color image")
                yield count, frame
                count += 1
            else:
                more, _ = self._capture.read()
                if more:
                    raise ValueError(
                        f"video exceeds duration limit of {self.max_duration_seconds:.2f}s"
                    )
        finally:
            if self.metadata is not None:
                self.metadata = replace(self.metadata, decoded_frame_count=count)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

