"""FastAPI wrapper around the shared local video-to-text service."""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Sequence

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

from fingerspelling.features.coordinate_adapter import COORDINATE_ADAPTERS
from fingerspelling.inference import Predictor, VideoInferenceService
from fingerspelling.video import video_to_features


LOGGER = logging.getLogger(__name__)


class PredictResponse(BaseModel):
    prediction: str
    status: str
    needs_review: bool | None = None
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
    upload_size_bytes: int
    note: str


class APIError(BaseModel):
    status: str = "error"
    message: str
    details: Any | None = None


def _configured_origins(origins: Sequence[str] | None = None) -> list[str]:
    if origins is not None:
        return [origin.strip() for origin in origins if origin.strip()]
    raw = os.environ.get(
        "FSBOARD_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(
    *,
    checkpoint_path: str | Path | None = None,
    hand_model_path: str | Path | None = None,
    device: str = "auto",
    max_duration_seconds: float = 120.0,
    max_upload_bytes: int = 100_000_000,
    max_pixels: int = 16_777_216,
    predictor: Any | None = None,
    video_processor: Callable[..., Any] = video_to_features,
    coordinate_adapter: str | None = None,
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    if max_duration_seconds <= 0 or max_upload_bytes <= 0 or max_pixels <= 0:
        raise ValueError("upload and duration limits must be positive")
    resolved_checkpoint = Path(
        checkpoint_path
        or os.environ.get("FSBOARD_CHECKPOINT", "checkpoints/bigru_ctc_motion_best.pt")
    )
    resolved_hand_model = Path(
        hand_model_path
        or os.environ.get("FSBOARD_HAND_MODEL", "models/hand_landmarker.task")
    )
    resolved_adapter = coordinate_adapter or os.environ.get(
        "FSBOARD_COORDINATE_ADAPTER", "pixel_xy_preserve_z"
    )
    if resolved_adapter not in COORDINATE_ADAPTERS:
        raise ValueError(f"FSBOARD_COORDINATE_ADAPTER must be one of {COORDINATE_ADAPTERS}")
    if predictor is not None:
        initial_service = VideoInferenceService(
            predictor,
            resolved_hand_model,
            coordinate_adapter=resolved_adapter,
            max_duration_seconds=max_duration_seconds,
            max_upload_bytes=max_upload_bytes,
            max_pixels=max_pixels,
            video_processor=video_processor,
        )
    else:
        initial_service = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            loaded = Predictor.from_checkpoint(resolved_checkpoint, device=device)
            app.state.service = VideoInferenceService(
                loaded,
                resolved_hand_model,
                coordinate_adapter=resolved_adapter,
                max_duration_seconds=max_duration_seconds,
                max_upload_bytes=max_upload_bytes,
                max_pixels=max_pixels,
                video_processor=video_processor,
            )
        yield

    app = FastAPI(
        title="FSboard Fingerspelling Inference",
        version="0.2.0",
        lifespan=lifespan,
        responses={
            400: {"model": APIError},
            413: {"model": APIError},
            415: {"model": APIError},
            422: {"model": APIError},
            503: {"model": APIError},
        },
    )
    app.state.service = initial_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_configured_origins(cors_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        allow_credentials=False,
    )

    @app.exception_handler(HTTPException)
    async def handle_http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder(APIError(message=str(exc.detail))),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                APIError(message="Request validation failed.", details=exc.errors())
            ),
        )

    @app.middleware("http")
    async def enforce_content_length(request: Request, call_next):
        if request.url.path == "/predict":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError:
                    return _json_error(400, "invalid Content-Length header")
                # Permit multipart headers around the bounded video payload.
                if declared > max_upload_bytes + 1_048_576:
                    return _json_error(413, "request exceeds the configured upload limit")
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        service = app.state.service
        blockers = []
        if service is None:
            blockers.append("checkpoint is not loaded")
        else:
            if not service.hand_model_path.is_file():
                blockers.append(f"HandLandmarker model missing: {service.hand_model_path}")
            for module_name, package_name in (
                ("cv2", "opencv-python"),
                ("mediapipe", "mediapipe"),
            ):
                if importlib.util.find_spec(module_name) is None:
                    blockers.append(f"install optional dependency {package_name}")
        ready = service is not None and not blockers
        return {
            "status": "ok" if ready else "not_ready",
            "model_loaded": service is not None,
            "video_inference_ready": ready,
            "coordinate_compatibility_verified": False,
            "coordinate_adapter": resolved_adapter,
            "blockers": blockers,
        }

    @app.get("/model-info")
    async def model_info() -> dict[str, Any]:
        service = app.state.service
        if service is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        checkpoint = service.predictor.checkpoint
        return {
            "model_version": f"bigru_ctc_{checkpoint.input_dim}d",
            "input_dim": checkpoint.input_dim,
            "hidden_size": checkpoint.hidden_size,
            "num_layers": checkpoint.num_layers,
            "num_classes": checkpoint.num_classes,
            "blank_index": service.predictor.vocabulary.blank_index,
            "epoch": checkpoint.epoch,
            "device": str(service.predictor.device),
            "coordinate_adapter": resolved_adapter,
            "coordinate_compatibility_verified": False,
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(
        file: UploadFile = File(...),
        hand: str | None = Form(default=None),
    ) -> PredictResponse:
        service = app.state.service
        if service is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        if not service.hand_model_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"HandLandmarker model missing: {service.hand_model_path}",
            )
        if hand not in (None, "", "left", "right"):
            raise HTTPException(status_code=422, detail="hand must be 'left', 'right', or omitted")
        original_name = file.filename or ""
        suffix = Path(original_name).suffix.lower()
        if suffix not in {".mp4", ".webm"}:
            await file.close()
            raise HTTPException(status_code=415, detail="video format must be MP4 or WebM")

        total_bytes = 0
        try:
            with tempfile.TemporaryDirectory(prefix="fsboard-inference-") as temp_dir:
                temp_path = Path(temp_dir) / f"upload{suffix}"
                with temp_path.open("wb") as target:
                    while True:
                        chunk = await file.read(1024 * 1024)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > service.max_upload_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail="video exceeds the configured upload limit",
                            )
                        target.write(chunk)
                if total_bytes == 0:
                    raise HTTPException(status_code=400, detail="uploaded video is empty")
                try:
                    result = await run_in_threadpool(
                        service.predict_path,
                        temp_path,
                        forced_side=hand or None,
                    )
                except (ValueError, FileNotFoundError) as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except RuntimeError as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except Exception as exc:
                    LOGGER.exception("Unexpected error during video inference")
                    raise HTTPException(
                        status_code=500,
                        detail="Video processing failed unexpectedly. Check the server log for details.",
                    ) from exc
                payload = result.to_dict()
                payload["upload_size_bytes"] = total_bytes
                return PredictResponse(**payload)
        finally:
            await file.close()

    return app


def _json_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(APIError(message=detail)),
    )


app = create_app()
