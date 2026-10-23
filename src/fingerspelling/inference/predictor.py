"""Reusable feature/pose predictor that keeps a checkpoint loaded once."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fingerspelling.features.motion import add_motion_features
from fingerspelling.features.normalization import normalize_hand
from fingerspelling.inference.checkpoint import LoadedCheckpoint, load_checkpoint


@dataclass(frozen=True)
class Prediction:
    text: str
    latency_ms: float
    model_forward_ms: float = 0.0
    ctc_decode_ms: float = 0.0


class Predictor:
    def __init__(self, checkpoint: LoadedCheckpoint) -> None:
        self.checkpoint = checkpoint
        self.model = checkpoint.model
        self.vocabulary = checkpoint.vocabulary
        self.device = next(self.model.parameters()).device

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "auto"
    ) -> "Predictor":
        return cls(load_checkpoint(path, device=device))

    def predict_features(self, features: np.ndarray | torch.Tensor) -> Prediction:
        tensor = torch.as_tensor(features, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[1] != self.checkpoint.input_dim:
            raise ValueError(
                f"features must have shape [time, {self.checkpoint.input_dim}]"
            )
        if tensor.shape[0] == 0:
            raise ValueError("features must contain at least one frame")
        if not torch.isfinite(tensor).all():
            raise ValueError("features contain NaN or infinite values")

        length = tensor.shape[0]
        forward_start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor.unsqueeze(0).to(self.device), torch.tensor([length]))
        model_forward_ms = (time.perf_counter() - forward_start) * 1000
        decode_start = time.perf_counter()
        with torch.inference_mode():
            text = self.vocabulary.decode_logits(logits[:length, 0], input_length=length)
        ctc_decode_ms = (time.perf_counter() - decode_start) * 1000
        return Prediction(
            text=text,
            latency_ms=model_forward_ms + ctc_decode_ms,
            model_forward_ms=model_forward_ms,
            ctc_decode_ms=ctc_decode_ms,
        )

    def predict_pose(self, pose: np.ndarray) -> Prediction:
        """Predict from the dataset's selected hand pose sequence ``[T,21,3]``."""
        features = normalize_hand(pose) if self.checkpoint.input_dim == 63 else add_motion_features(pose)
        return self.predict_features(features)

