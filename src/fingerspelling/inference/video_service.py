"""Shared video-to-text inference service for CLI and HTTP applications."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from fingerspelling.features.coordinate_adapter import COORDINATE_ADAPTERS
from fingerspelling.inference.predictor import Predictor
from fingerspelling.video.video_to_features import VideoFeatures, video_to_features


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInferenceResult:
    status: str
    prediction: str
    needs_review: None
    processing_time_ms: float
    latency_ms: float
    model_version: str
    device: str
    input_duration_seconds: float
    input_fps: float
    input_width: int
    input_height: int
    input_frames: int
    landmark_frames: int
    hand_detection_rate: float
    selected_hand: str
    feature_dim: int
    coordinate_adapter: str
    coordinate_compatibility_verified: bool
    stage_timings_ms: dict[str, float]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VideoInferenceService:
    """Run the same real video pipeline from both the CLI and FastAPI."""

    def __init__(
        self,
        predictor: Predictor,
        hand_model_path: str | Path,
        *,
        coordinate_adapter: str = "pixel_xy_preserve_z",
        max_duration_seconds: float = 120.0,
        max_upload_bytes: int = 100_000_000,
        max_pixels: int = 16_777_216,
        video_processor: Callable[..., VideoFeatures] = video_to_features,
    ) -> None:
        if coordinate_adapter not in COORDINATE_ADAPTERS:
            raise ValueError(
                f"coordinate_adapter must be one of {COORDINATE_ADAPTERS}"
            )
        if max_duration_seconds <= 0 or max_upload_bytes <= 0 or max_pixels <= 0:
            raise ValueError("upload and duration limits must be positive")
        self.predictor = predictor
        self.hand_model_path = Path(hand_model_path)
        self.coordinate_adapter = coordinate_adapter
        self.max_duration_seconds = max_duration_seconds
        self.max_upload_bytes = max_upload_bytes
        self.max_pixels = max_pixels
        self.video_processor = video_processor

    def predict_path(
        self, path: str | Path, *, forced_side: str | None = None
    ) -> VideoInferenceResult:
        started = time.perf_counter()
        video = self.video_processor(
            path,
            self.hand_model_path,
            input_dim=self.predictor.checkpoint.input_dim,
            forced_side=forced_side,
            coordinate_adapter=self.coordinate_adapter,
            max_duration_seconds=self.max_duration_seconds,
            max_upload_bytes=self.max_upload_bytes,
            max_pixels=self.max_pixels,
        )
        prediction = self.predictor.predict_features(video.features)
        processing_time_ms = (time.perf_counter() - started) * 1000
        duration = video.metadata.decoded_frame_count / video.metadata.fps
        timings = {
            "video_decode": round(video.timings.decode_ms, 3),
            "hand_tracker_init": round(video.timings.hand_tracker_init_ms, 3),
            "hand_landmark_extraction": round(video.timings.tracking_ms, 3),
            "coordinate_adapter": round(video.timings.coordinate_adapter_ms, 3),
            "feature_extraction": round(video.timings.feature_ms, 3),
            "model_forward": round(prediction.model_forward_ms, 3),
            "ctc_decode": round(prediction.ctc_decode_ms, 3),
            "model_inference": round(prediction.latency_ms, 3),
            "video_pipeline_total": round(video.timings.total_ms, 3),
            "backend_processing_total": round(processing_time_ms, 3),
        }
        result = VideoInferenceResult(
            status="success",
            prediction=prediction.text,
            needs_review=None,
            processing_time_ms=round(processing_time_ms, 3),
            latency_ms=round(processing_time_ms, 3),
            model_version=f"bigru_ctc_{self.predictor.checkpoint.input_dim}d",
            device=str(self.predictor.device),
            input_duration_seconds=round(duration, 3),
            input_fps=video.metadata.fps,
            input_width=video.metadata.width,
            input_height=video.metadata.height,
            input_frames=video.metadata.decoded_frame_count,
            landmark_frames=len(video.frame_indices),
            hand_detection_rate=round(video.detection_rate, 6),
            selected_hand=video.selected_hand,
            feature_dim=int(video.features.shape[1]),
            coordinate_adapter=video.coordinate_adapter,
            coordinate_compatibility_verified=False,
            stage_timings_ms=timings,
            note=(
                "MediaPipe-to-FSboard coordinate compatibility has not been validated "
                "on a matched source-video sample. Treat this prediction as experimental."
            ),
        )
        LOGGER.info(
            "Video inference completed: model=%s device=%s frames=%d landmarks=%d "
            "detection_rate=%.4f processing_ms=%.3f adapter=%s",
            result.model_version,
            result.device,
            result.input_frames,
            result.landmark_frames,
            result.hand_detection_rate,
            result.processing_time_ms,
            result.coordinate_adapter,
        )
        return result
