from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fingerspelling.video.decoder import VideoMetadata
from fingerspelling.video.video_to_features import VideoFeatureTimings, VideoFeatures
from backend.app.main import create_app


class _FakePredictor:
    def __init__(self):
        self.checkpoint = SimpleNamespace(
            input_dim=126,
            hidden_size=128,
            num_layers=2,
            num_classes=60,
            epoch=3,
        )
        self.vocabulary = SimpleNamespace(blank_index=0)
        self.device = "cpu"

    def predict_features(self, features):
        assert features.shape == (2, 126)
        return SimpleNamespace(
            text="hello", latency_ms=1.2, model_forward_ms=0.8, ctc_decode_ms=0.4
        )


def _fake_video_processor(path, hand_model, **kwargs):
    return VideoFeatures(
        pose=np.zeros((2, 21, 3), dtype=np.float32),
        features=np.zeros((2, 126), dtype=np.float32),
        frame_indices=np.asarray([0, 2], dtype=np.int64),
        detection_rate=2 / 3,
        selected_hand=kwargs.get("forced_side") or "right",
        metadata=VideoMetadata(str(path), 30.0, 200, 100, 3, 0.1, 3),
        timings=VideoFeatureTimings(1.0, 2.0, 0.5, 0.2, 4.0, 0.1),
        coordinate_adapter=kwargs.get("coordinate_adapter", "pixel_xy_preserve_z"),
    )


def test_api_health_model_info_cors_and_ephemeral_upload(tmp_path):
    received = []

    def video_processor(path, hand_model, **kwargs):
        assert path.is_file()
        received.append((path, kwargs.get("forced_side")))
        return _fake_video_processor(path, hand_model, **kwargs)

    hand_model = tmp_path / "hand.task"
    hand_model.write_bytes(b"test model marker")
    app = create_app(
        predictor=_FakePredictor(),
        hand_model_path=hand_model,
        video_processor=video_processor,
        cors_origins=["http://localhost:5173"],
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is True
        assert health["coordinate_compatibility_verified"] is False
        model = client.get("/model-info").json()
        assert model["input_dim"] == 126
        assert model["blank_index"] == 0

        preflight = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"

        response = client.post(
            "/predict",
            data={"hand": "left"},
            files={"file": ("sample.webm", b"fake video payload", "video/webm")},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["prediction"] == "hello"
    assert payload["status"] == "success"
    assert payload["needs_review"] is None
    assert payload["latency_ms"] > 0
    assert payload["processing_time_ms"] > 0
    assert payload["model_version"] == "bigru_ctc_126d"
    assert payload["hand_detection_rate"] == pytest.approx(2 / 3)
    assert payload["input_frames"] == 3
    assert payload["landmark_frames"] == 2
    assert payload["feature_dim"] == 126
    assert payload["selected_hand"] == "left"
    assert payload["stage_timings_ms"]["model_forward"] == 0.8
    assert payload["stage_timings_ms"]["ctc_decode"] == 0.4
    assert received and received[0][1] == "left"
    assert not received[0][0].exists()


def test_api_rejects_unsupported_oversized_and_invalid_hand(tmp_path):
    hand_model = tmp_path / "hand.task"
    hand_model.write_bytes(b"test model marker")
    app = create_app(
        predictor=_FakePredictor(),
        hand_model_path=hand_model,
        max_upload_bytes=4,
        video_processor=_fake_video_processor,
    )
    with TestClient(app) as client:
        unsupported = client.post(
            "/predict", files={"file": ("sample.avi", b"abc", "video/x-msvideo")}
        )
        oversized = client.post(
            "/predict", files={"file": ("sample.mp4", b"12345", "video/mp4")}
        )
        invalid_hand = client.post(
            "/predict",
            data={"hand": "both"},
            files={"file": ("sample.mp4", b"1234", "video/mp4")},
        )
    assert unsupported.status_code == 415
    assert unsupported.json()["status"] == "error"
    assert oversized.status_code == 413
    assert invalid_hand.status_code == 422


def test_api_fails_clearly_when_mediapipe_model_is_missing(tmp_path):
    app = create_app(
        predictor=_FakePredictor(),
        hand_model_path=tmp_path / "missing.task",
        video_processor=_fake_video_processor,
    )
    with TestClient(app) as client:
        response = client.post(
            "/predict", files={"file": ("sample.mp4", b"video", "video/mp4")}
        )
    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert "HandLandmarker model missing" in response.json()["message"]


def test_api_returns_structured_error_for_unexpected_inference_failure(tmp_path):
    hand_model = tmp_path / "hand.task"
    hand_model.write_bytes(b"test model marker")

    def broken_video_processor(path, model, **kwargs):
        raise TypeError("unexpected processor failure")

    app = create_app(
        predictor=_FakePredictor(),
        hand_model_path=hand_model,
        video_processor=broken_video_processor,
    )
    with TestClient(app) as client:
        response = client.post(
            "/predict", files={"file": ("sample.mp4", b"video", "video/mp4")}
        )
    assert response.status_code == 500
    assert response.json() == {
        "status": "error",
        "message": "Video processing failed unexpectedly. Check the server log for details.",
        "details": None,
    }
