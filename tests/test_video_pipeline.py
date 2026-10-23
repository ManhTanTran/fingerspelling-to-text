import numpy as np
import pytest
from types import SimpleNamespace
import sys

from fingerspelling.features.coordinate_adapter import mediapipe_to_fsboard_coordinates
from fingerspelling.video.hand_tracker import SigningHandTracker
from fingerspelling.video.video_to_features import video_to_features
from fingerspelling.video.decoder import VideoMetadata
from fingerspelling.video.decoder import VideoDecoder


def test_coordinate_adapter_scales_xy_but_preserves_depth():
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.5, 0.25, -0.1]

    adapted = mediapipe_to_fsboard_coordinates(landmarks, 1920, 1080)

    assert adapted.shape == (21, 3)
    assert adapted.dtype == np.float32
    assert adapted[0] == pytest.approx([960.0, 270.0, -0.1])


def test_coordinate_adapter_handles_sequences_and_validates_dimensions():
    sequence = np.ones((4, 21, 3), dtype=np.float32) * 0.25
    adapted = mediapipe_to_fsboard_coordinates(sequence, 640, 480)
    assert adapted.shape == (4, 21, 3)
    assert adapted[0, 0] == pytest.approx([160.0, 120.0, 0.25])
    with pytest.raises(ValueError, match="positive"):
        mediapipe_to_fsboard_coordinates(sequence, 0, 480)


def _landmarks(offset=0.0):
    return [SimpleNamespace(x=0.1 + offset, y=0.2, z=-0.05) for _ in range(21)]


def _tracker_for_selection():
    tracker = object.__new__(SigningHandTracker)
    tracker.locked_side = None
    return tracker


def test_tracker_locks_first_detected_hand_and_does_not_switch():
    tracker = _tracker_for_selection()
    first = SimpleNamespace(
        hand_landmarks=[_landmarks(), _landmarks(0.2)],
        handedness=[
            [SimpleNamespace(category_name="Left", score=0.9)],
            [SimpleNamespace(category_name="Right", score=0.8)],
        ],
    )
    detected = tracker._select_detection(first, 0)
    assert detected is not None
    assert detected.side == "left"
    assert tracker.locked_side == "left"

    other_hand_only = SimpleNamespace(
        hand_landmarks=[_landmarks(0.2)],
        handedness=[[SimpleNamespace(category_name="Right", score=1.0)]],
    )
    assert tracker._select_detection(other_hand_only, 33) is None
    assert tracker.locked_side == "left"


class _FakeDecoder:
    def __init__(self, path, **kwargs):
        self.metadata = None

    def __enter__(self):
        self.metadata = VideoMetadata("fake.mp4", 30.0, 200, 100, 3, 0.1)
        return self

    def __iter__(self):
        try:
            for index in range(3):
                frame = np.full((100, 200, 3), index, dtype=np.uint8)
                yield index, frame
        finally:
            from dataclasses import replace

            self.metadata = replace(self.metadata, decoded_frame_count=3)

    def __exit__(self, exc_type, exc, traceback):
        return None


class _FakeTracker:
    def __init__(self, model_path, forced_side=None):
        self.locked_side = forced_side or "right"
        self.forced_side = forced_side
        self.processed = 0

    @property
    def detection_rate(self):
        return 2 / 3

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def detect(self, frame, timestamp_ms):
        self.processed += 1
        assert timestamp_ms == [0, 33, 67][self.processed - 1]
        if int(frame[0, 0, 0]) == 1:
            return None
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = [0.1, 0.2, -0.05]
        landmarks[9] = [0.2, 0.2, -0.05]
        return SimpleNamespace(landmarks=landmarks, side="right", score=0.9)


def test_video_pipeline_omits_missing_frames_and_builds_126d_features():
    video = video_to_features(
        "ignored.mp4",
        "ignored.task",
        input_dim=126,
        decoder_factory=_FakeDecoder,
        tracker_factory=_FakeTracker,
    )
    assert video.features.shape == (2, 126)
    assert video.pose.shape == (2, 21, 3)
    assert video.frame_indices.tolist() == [0, 2]
    assert video.detection_rate == pytest.approx(2 / 3)
    assert video.selected_hand == "right"
    assert video.metadata.decoded_frame_count == 3
    assert np.isfinite(video.features).all()


class _FakeCapture:
    def __init__(self, frames, fps=30.0, width=16, height=12):
        self.frames = list(frames)
        self.fps = fps
        self.width = width
        self.height = height
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            1: self.fps,
            2: self.width,
            3: self.height,
            4: len(self.frames),
        }[prop]

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def _fake_cv2(capture):
    return SimpleNamespace(
        VideoCapture=lambda path: capture,
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3,
        CAP_PROP_FRAME_COUNT=4,
    )


def test_video_decoder_preserves_order_and_releases_capture(tmp_path, monkeypatch):
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"video")
    frames = [np.full((12, 16, 3), i, dtype=np.uint8) for i in range(3)]
    capture = _FakeCapture(frames)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(capture))

    with VideoDecoder(path) as decoder:
        decoded = list(decoder)
        assert [index for index, _ in decoded] == [0, 1, 2]
        assert [int(frame[0, 0, 0]) for _, frame in decoded] == [0, 1, 2]
    assert capture.released
    assert decoder.metadata.decoded_frame_count == 3


def test_video_decoder_reports_empty_and_over_duration_videos(tmp_path, monkeypatch):
    path = tmp_path / "empty.webm"
    path.write_bytes(b"video")
    empty_capture = _FakeCapture([])
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(empty_capture))
    with VideoDecoder(path) as decoder:
        with pytest.raises(ValueError, match="empty, corrupt"):
            list(decoder)

    long_capture = _FakeCapture([np.zeros((12, 16, 3), dtype=np.uint8)] * 4, fps=2)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(long_capture))
    with pytest.raises(ValueError, match="exceeds limit"):
        with VideoDecoder(path, max_duration_seconds=1.0):
            pass


def test_video_decoder_rejects_bad_fps_resolution_size_and_format(tmp_path, monkeypatch):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"12345")

    with pytest.raises(ValueError, match="upload limit"):
        VideoDecoder(path, max_upload_bytes=4)
    with pytest.raises(ValueError, match="format must be MP4 or WebM"):
        VideoDecoder(tmp_path / "bad.avi")

    video_path = tmp_path / "bad.webm"
    video_path.write_bytes(b"video")
    bad_fps = _FakeCapture([np.zeros((12, 16, 3), dtype=np.uint8)], fps=0)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(bad_fps))
    with pytest.raises(ValueError, match="invalid FPS"):
        with VideoDecoder(video_path):
            pass
    assert bad_fps.released

    too_large = _FakeCapture([np.zeros((12, 16, 3), dtype=np.uint8)])
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(too_large))
    with pytest.raises(ValueError, match="resolution"):
        with VideoDecoder(video_path, max_pixels=100):
            pass
    assert too_large.released

