"""Predict fingerspelled text from one local MP4 or WebM video."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from fingerspelling.features.coordinate_adapter import COORDINATE_ADAPTERS
from fingerspelling.inference import Predictor, VideoInferenceService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("FSBOARD_CHECKPOINT", "checkpoints/bigru_ctc_motion_best.pt")),
        help="trusted 126D checkpoint (default: FSBOARD_CHECKPOINT or local 126D checkpoint)",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="local MediaPipe HandLandmarker .task model (default: models/hand_landmarker.task)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda")
    parser.add_argument("--hand", choices=("left", "right"), default=None)
    parser.add_argument(
        "--coordinate-adapter",
        choices=COORDINATE_ADAPTERS,
        default=os.environ.get("FSBOARD_COORDINATE_ADAPTER", "pixel_xy_preserve_z"),
        help="candidate MediaPipe-to-FSboard coordinates (default: pixel_xy_preserve_z)",
    )
    parser.add_argument("--max-duration-seconds", type=float, default=120.0)
    parser.add_argument("--max-upload-bytes", type=int, default=100_000_000)
    parser.add_argument("--max-pixels", type=int, default=16_777_216)
    args = parser.parse_args()

    try:
        load_start = time.perf_counter()
        predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
        checkpoint_load_ms = (time.perf_counter() - load_start) * 1000
        service = VideoInferenceService(
            predictor,
            args.hand_model,
            coordinate_adapter=args.coordinate_adapter,
            max_duration_seconds=args.max_duration_seconds,
            max_upload_bytes=args.max_upload_bytes,
            max_pixels=args.max_pixels,
        )
        result = service.predict_path(args.video, forced_side=args.hand).to_dict()
        result["checkpoint_epoch"] = predictor.checkpoint.epoch
        result["checkpoint_load_ms"] = round(checkpoint_load_ms, 3)
        result["total_including_checkpoint_load_ms"] = round(
            checkpoint_load_ms + result["processing_time_ms"], 3
        )
        result["stage_timings_ms"]["checkpoint_load"] = round(checkpoint_load_ms, 3)
    except Exception as exc:
        print(f"predict_video: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
