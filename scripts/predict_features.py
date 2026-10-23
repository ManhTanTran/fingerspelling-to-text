"""Run checkpoint inference from a local NumPy pose or feature array."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fingerspelling.inference import Predictor


def predict_array(checkpoint: Path, array_path: Path, device: str = "auto") -> dict[str, object]:
    """Accept [T, 21, 3] poses or [T, checkpoint_input_dim] features."""
    try:
        values = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load NumPy input safely: {exc}") from exc
    if not isinstance(values, np.ndarray):
        close = getattr(values, "close", None)
        if close is not None:
            close()
        raise ValueError("input must be a .npy array, not an NPZ archive")
    predictor = Predictor.from_checkpoint(checkpoint, device=device)
    if values.ndim == 3 and values.shape[1:] == (21, 3):
        prediction = predictor.predict_pose(values)
        input_kind = "pose"
    elif values.ndim == 2 and values.shape[1] == predictor.checkpoint.input_dim:
        prediction = predictor.predict_features(values)
        input_kind = "features"
    else:
        raise ValueError(
            f"input must have shape [T, 21, 3] (pose) or [T, "
            f"{predictor.checkpoint.input_dim}] (features), got {values.shape}"
        )
    return {
        "status": "success",
        "prediction": prediction.text,
        "model_version": f"bigru_ctc_{predictor.checkpoint.input_dim}d",
        "input_kind": input_kind,
        "input_frames": int(values.shape[0]),
        "latency_ms": prediction.latency_ms,
        "coordinate_compatibility_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="trusted local .npy pose or feature array")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        result = predict_array(args.checkpoint, args.input, args.device)
    except Exception as exc:
        print(f"predict_features: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
