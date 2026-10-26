from pathlib import Path

import pytest

from fingerspelling.inference.settings import PROJECT_ROOT, load_inference_settings


def test_settings_resolve_config_env_and_explicit_overrides_in_order(tmp_path):
    config = tmp_path / "inference.yaml"
    config.write_text(
        "checkpoint: checkpoints/from-config.pt\n"
        "hand_model: models/from-config.task\n"
        "device: cpu\n"
        "max_duration_seconds: 60\n"
        "max_upload_bytes: 2000\n"
        "max_pixels: 3000\n"
        "coordinate_adapter: normalized_xyz\n",
        encoding="utf-8",
    )
    settings = load_inference_settings(
        config_path=config,
        environ={
            "FSBOARD_CHECKPOINT": "checkpoints/from-env.pt",
            "FSBOARD_DEVICE": "auto",
            "FSBOARD_MAX_PIXELS": "4000",
        },
        overrides={
            "checkpoint": "checkpoints/from-call.pt",
            "max_duration_seconds": 30,
        },
    )

    assert settings.checkpoint == PROJECT_ROOT / "checkpoints/from-call.pt"
    assert settings.hand_model == PROJECT_ROOT / "models/from-config.task"
    assert settings.device == "auto"
    assert settings.max_duration_seconds == 30
    assert settings.max_upload_bytes == 2000
    assert settings.max_pixels == 4000
    assert settings.coordinate_adapter == "normalized_xyz"


def test_settings_reject_unknown_adapter_and_invalid_limits(tmp_path):
    config = tmp_path / "inference.yaml"
    config.write_text("coordinate_adapter: unknown_adapter\n", encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate_adapter"):
        load_inference_settings(config_path=config, environ={})

    config.write_text("max_pixels: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_pixels"):
        load_inference_settings(config_path=config, environ={})


def test_settings_reject_unknown_config_keys(tmp_path):
    config = tmp_path / "inference.yaml"
    config.write_text("unrelated: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown inference config keys"):
        load_inference_settings(config_path=config, environ={})


def test_repository_inference_config_uses_a_supported_adapter():
    settings = load_inference_settings(environ={})

    assert settings.coordinate_adapter in {"pixel_xy_preserve_z", "normalized_xyz"}
