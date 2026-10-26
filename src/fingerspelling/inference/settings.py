"""Resolve shared video-inference settings from YAML, environment, and callers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml

from fingerspelling.features.coordinate_adapter import COORDINATE_ADAPTERS


def _find_project_root() -> Path:
    """Find the checkout root in source and installed-package executions."""
    source = Path(__file__).resolve()
    # Preserve an ASCII junction supplied as the working directory. MediaPipe's
    # Windows native task loader can reject canonical Unicode paths even though
    # Python and the OS can open them.
    candidates = (Path.cwd(), source.parent)
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if (parent / "pyproject.toml").is_file() and (
                parent / "configs" / "inference.yaml"
            ).is_file():
                return parent
    return source.parents[3]


PROJECT_ROOT = _find_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "inference.yaml"

_DEFAULTS: dict[str, Any] = {
    "checkpoint": "checkpoints/bigru_ctc_motion_best.pt",
    "hand_model": "models/hand_landmarker.task",
    "device": "auto",
    "max_duration_seconds": 120.0,
    "max_upload_bytes": 100_000_000,
    "max_pixels": 16_777_216,
    "coordinate_adapter": "pixel_xy_preserve_z",
}
_ENVIRONMENT_KEYS = {
    "checkpoint": "FSBOARD_CHECKPOINT",
    "hand_model": "FSBOARD_HAND_MODEL",
    "device": "FSBOARD_DEVICE",
    "max_duration_seconds": "FSBOARD_MAX_DURATION_SECONDS",
    "max_upload_bytes": "FSBOARD_MAX_UPLOAD_BYTES",
    "max_pixels": "FSBOARD_MAX_PIXELS",
    "coordinate_adapter": "FSBOARD_COORDINATE_ADAPTER",
}


@dataclass(frozen=True)
class InferenceSettings:
    checkpoint: Path
    hand_model: Path
    device: str
    max_duration_seconds: float
    max_upload_bytes: int
    max_pixels: int
    coordinate_adapter: str


def _project_path(value: object, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"inference setting {field!r} must be a nonempty path")
    path = Path(value).expanduser()
    # Do not canonicalize through a junction: that would turn an ASCII alias
    # back into the Unicode checkout path before passing it to MediaPipe.
    return path if path.is_absolute() else PROJECT_ROOT / path


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"inference setting {field!r} must be a positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"inference setting {field!r} must be a positive number") from exc
    if not isfinite(number) or number <= 0:
        raise ValueError(f"inference setting {field!r} must be a positive number")
    return number


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"inference setting {field!r} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"inference setting {field!r} must be a positive integer") from exc
    if number <= 0 or str(value).strip() not in (str(number), f"+{number}"):
        raise ValueError(f"inference setting {field!r} must be a positive integer")
    return number


def load_inference_settings(
    *,
    config_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> InferenceSettings:
    """Load repo-relative YAML defaults, then environment and explicit overrides.

    Relative paths in the YAML, environment, or overrides are resolved from the
    repository root. ``FSBOARD_INFERENCE_CONFIG`` selects an alternate YAML file.
    Explicit overrides are intended for CLI flags and ``create_app`` arguments.
    """
    env = os.environ if environ is None else environ
    selected_config = config_path or env.get("FSBOARD_INFERENCE_CONFIG") or DEFAULT_CONFIG_PATH
    selected_config = Path(selected_config).expanduser()
    if not selected_config.is_absolute():
        selected_config = PROJECT_ROOT / selected_config
    selected_config = selected_config.resolve()
    if not selected_config.is_file():
        raise FileNotFoundError(f"inference config not found: {selected_config}")

    with selected_config.open("r", encoding="utf-8") as source:
        payload = yaml.safe_load(source) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("inference config must contain a YAML mapping")
    unknown = set(payload) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown inference config keys: {sorted(unknown)}")

    values = {**_DEFAULTS, **payload}
    for key, env_name in _ENVIRONMENT_KEYS.items():
        if env.get(env_name) not in (None, ""):
            values[key] = env[env_name]
    for key, value in (overrides or {}).items():
        if key not in _DEFAULTS:
            raise ValueError(f"unknown inference override: {key}")
        if value is not None:
            values[key] = value

    device = values["device"]
    if not isinstance(device, str) or not device.strip():
        raise ValueError("inference setting 'device' must be a nonempty string")
    adapter = values["coordinate_adapter"]
    if not isinstance(adapter, str) or adapter not in COORDINATE_ADAPTERS:
        raise ValueError(
            f"inference setting 'coordinate_adapter' must be one of {COORDINATE_ADAPTERS}"
        )

    return InferenceSettings(
        checkpoint=_project_path(values["checkpoint"], "checkpoint"),
        hand_model=_project_path(values["hand_model"], "hand_model"),
        device=device.strip(),
        max_duration_seconds=_positive_float(
            values["max_duration_seconds"], "max_duration_seconds"
        ),
        max_upload_bytes=_positive_int(values["max_upload_bytes"], "max_upload_bytes"),
        max_pixels=_positive_int(values["max_pixels"], "max_pixels"),
        coordinate_adapter=adapter,
    )
