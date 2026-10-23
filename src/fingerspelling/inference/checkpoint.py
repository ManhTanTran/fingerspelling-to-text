"""Strict, weights-only loader for trusted local Task 3 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fingerspelling.data.vocabulary import Vocabulary
from fingerspelling.models import BiGRUCTC


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: BiGRUCTC
    vocabulary: Vocabulary
    input_dim: int
    hidden_size: int
    num_layers: int
    num_classes: int
    epoch: int | None
    path: Path


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"checkpoint field {key!r} must be a positive integer")
    return value


def _select_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return selected


def load_checkpoint(
    path: str | Path, device: str | torch.device = "auto"
) -> LoadedCheckpoint:
    """Load a local checkpoint with ``weights_only=True`` and strict keys.

    This intentionally refuses legacy arbitrary-pickle loading. Only load
    checkpoints obtained from a trusted source.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("this PyTorch version lacks weights_only checkpoint loading") from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a dictionary")

    input_dim = _positive_int(payload, "input_dim")
    hidden_size = _positive_int(payload, "hidden_size")
    num_layers = _positive_int(payload, "num_layers")
    num_classes = _positive_int(payload, "num_classes")
    blank_index = payload.get("blank_idx")
    vocabulary = Vocabulary.from_checkpoint(
        payload.get("char_to_idx"),
        payload.get("idx_to_char"),
        blank_index,
        num_classes,
    )

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing model_state_dict")

    model = BiGRUCTC(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        bidirectional=True,
    )
    model.load_state_dict(state_dict, strict=True)
    target_device = _select_device(device)
    model.to(target_device)
    model.eval()
    epoch = payload.get("epoch")
    if epoch is not None and (not isinstance(epoch, int) or isinstance(epoch, bool)):
        raise ValueError("checkpoint epoch must be an integer when present")

    return LoadedCheckpoint(
        model=model,
        vocabulary=vocabulary,
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        epoch=epoch,
        path=checkpoint_path.resolve(),
    )

