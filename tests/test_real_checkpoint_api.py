"""Real-checkpoint/API integration on valid features, without claiming video E2E."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from fingerspelling.inference import Predictor, VideoInferenceService
from fingerspelling.video.decoder import VideoMetadata
from fingerspelling.video.video_to_features import VideoFeatureTimings, VideoFeatures


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "checkpoints" / "bigru_ctc_motion_best.pt"


def _valid_feature_video(path, hand_model, **kwargs):
    return VideoFeatures(
        pose=np.zeros((8, 21, 3), dtype=np.float32),
        features=np.zeros((8, 126), dtype=np.float32),
        frame_indices=np.arange(8, dtype=np.int64),
        detection_rate=1.0,
        selected_hand=kwargs.get("forced_side") or "right",
        metadata=VideoMetadata(str(path), 30.0, 640, 480, 8, 8 / 30, 8),
        timings=VideoFeatureTimings(0.3, 1.0, 0.2, 0.1, 2.0, 0.1),
        coordinate_adapter=kwargs.get("coordinate_adapter", "pixel_xy_preserve_z"),
    )


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="local ignored 126D checkpoint is unavailable")
def test_real_checkpoint_api_matches_shared_video_service(tmp_path):
    predictor = Predictor.from_checkpoint(CHECKPOINT, device="cpu")
    hand_model = tmp_path / "hand.task"
    hand_model.write_bytes(b"test model marker")
    service = VideoInferenceService(
        predictor,
        hand_model,
        video_processor=_valid_feature_video,
    )
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fixture bytes; processor supplies valid features")
    cli_result = service.predict_path(video_path, forced_side="left")

    app = create_app(
        predictor=predictor,
        hand_model_path=hand_model,
        video_processor=_valid_feature_video,
    )
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data={"hand": "left"},
            files={"file": ("sample.mp4", video_path.read_bytes(), "video/mp4")},
        )

    assert response.status_code == 200, response.text
    api_result = response.json()
    assert api_result["prediction"] == cli_result.prediction
    assert api_result["model_version"] == "bigru_ctc_126d"
    assert api_result["feature_dim"] == 126
    assert api_result["coordinate_compatibility_verified"] is False
    assert api_result["hand_detection_rate"] == 1.0
    assert set(api_result["stage_timings_ms"]) >= {
        "video_decode", "hand_landmark_extraction", "feature_extraction",
        "model_forward", "ctc_decode", "backend_processing_total",
    }
