"""Resolve storage paths from YAML with environment and CLI precedence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "storage.yaml"


def _configured_path(value: str | Path | None, root: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class StorageConfig:
    project_root: Path
    config_path: Path
    drive_root: Path | None
    drive_mount_point: Path
    drive_dataset_dir: Path | None
    drive_checkpoints_dir: Path | None
    drive_experiments_dir: Path | None
    drive_evaluation_dir: Path | None
    source_archive_path: Path | None
    source_archive_file_id: str | None
    source_archive_name: str
    test_npz_member: str
    test_npz_override: Path | None
    checkpoint_override: Path | None
    dataset_dir_override: Path | None
    test_npz_name: str
    motion_checkpoint_name: str
    local_motion_checkpoint: Path
    local_test_npz: Path | None
    windows_subset_dir: Path
    local_evaluation_dir: Path
    local_experiments_dir: Path
    colab_cache_root: Path

    def resolve_test_npz(self, cli_path: str | Path | None = None) -> Path:
        """Resolve a supplied test NPZ without requiring a Windows full copy."""
        if cli_path is not None:
            candidate = Path(cli_path).expanduser().resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"test NPZ not found: {candidate}")
            return candidate

        if self.test_npz_override is not None:
            if not self.test_npz_override.is_file():
                raise FileNotFoundError(
                    f"FSBOARD_TEST_NPZ does not exist: {self.test_npz_override}"
                )
            return self.test_npz_override

        if self.dataset_dir_override is not None:
            candidate = self.dataset_dir_override / self.test_npz_name
            if not candidate.is_file():
                raise FileNotFoundError(f"FSBOARD_DATASET_DIR test NPZ not found: {candidate}")
            return candidate.resolve()

        if self.drive_dataset_dir is not None:
            candidate = self.drive_dataset_dir / self.test_npz_name
            if candidate.is_file():
                return candidate.resolve()

        if self.local_test_npz is not None:
            if not self.local_test_npz.is_file():
                raise FileNotFoundError(f"configured local test NPZ not found: {self.local_test_npz}")
            return self.local_test_npz

        subset_candidate = self.windows_subset_dir / "fsboard_test_subset.npz"
        if subset_candidate.is_file():
            return subset_candidate.resolve()

        expected = (
            str(self.drive_dataset_dir / self.test_npz_name)
            if self.drive_dataset_dir is not None
            else f"<Drive dataset>/datasets/FSboard/{self.test_npz_name}"
        )
        raise FileNotFoundError(
            "No processed test NPZ was resolved. Pass --test-npz, set "
            "FSBOARD_TEST_NPZ, configure local.test_npz, or mount Drive with "
            f"the expected file at {expected}."
        )

    def resolve_checkpoint(self, cli_path: str | Path | None = None) -> Path:
        """Prefer an explicit path, then Drive backup, then the local API copy."""
        if cli_path is not None:
            candidate = Path(cli_path).expanduser().resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"checkpoint not found: {candidate}")
            return candidate

        if self.checkpoint_override is not None:
            candidate = self.checkpoint_override
            if not candidate.is_file():
                raise FileNotFoundError(f"FSBOARD_CHECKPOINT does not exist: {candidate}")
            return candidate

        if self.drive_checkpoints_dir is not None:
            drive_checkpoint = self.drive_checkpoints_dir / self.motion_checkpoint_name
            if drive_checkpoint.is_file():
                return drive_checkpoint.resolve()

        candidate = self.local_motion_checkpoint
        if not candidate.is_file():
            raise FileNotFoundError(
                f"126D checkpoint not found at {candidate}; set --checkpoint, "
                "FSBOARD_CHECKPOINT, or provide the Drive backup."
            )
        return candidate

    def resolve_output_dir(self, cli_path: str | Path | None = None, *, kind: str = "evaluation") -> Path:
        if cli_path is not None:
            return _configured_path(cli_path, self.project_root)  # type: ignore[return-value]
        if kind == "evaluation":
            return self.local_evaluation_dir
        if kind == "experiments":
            return self.local_experiments_dir
        raise ValueError("kind must be 'evaluation' or 'experiments'")


def load_storage_config(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> StorageConfig:
    """Load `configs/storage.yaml`; environment variables override YAML paths."""
    env = os.environ if environ is None else environ
    source = Path(config_path or env.get("FSBOARD_STORAGE_CONFIG", DEFAULT_CONFIG)).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"storage config not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    storage = payload.get("storage", {})
    drive = storage.get("drive", {})
    local = storage.get("local", {})
    root = Path(project_root or source.parents[1]).expanduser().resolve()

    drive_root_value = env.get(drive.get("root_env", "FSBOARD_DRIVE_ROOT")) or drive.get("root")
    drive_root = _configured_path(drive_root_value, root)
    mount = _configured_path(env.get("FSBOARD_DRIVE_MOUNT", drive.get("mount_point")), root)
    if mount is None:
        mount = Path("/content/drive")
    if drive_root is None:
        my_drive_root = mount / "MyDrive"
        if my_drive_root.is_dir():
            drive_root = my_drive_root / str(drive.get("folder_name", "TTNM_Fingerspelling"))

    def drive_path(key: str) -> Path | None:
        if drive_root is None:
            return None
        return (drive_root / str(drive[key])).resolve()

    local_checkpoint = env.get("FSBOARD_LOCAL_CHECKPOINT") or local.get("motion_checkpoint")
    local_test_npz = env.get("FSBOARD_TEST_NPZ") or local.get("test_npz")
    local_evaluation = env.get("FSBOARD_EVALUATION_DIR") or local.get("evaluation_dir")
    local_experiments = env.get("FSBOARD_EXPERIMENTS_DIR") or local.get("experiments_dir")
    subset_dir = env.get("FSBOARD_WINDOWS_SUBSET_DIR") or local.get("windows_subset_dir")
    cache_root = env.get("FSBOARD_LOCAL_CACHE_ROOT") or local.get("colab_cache_root")
    source_archive_path = env.get("FSBOARD_SOURCE_ARCHIVE") or drive.get("source_archive_path")
    source_archive_file_id = env.get("FSBOARD_SOURCE_ARCHIVE_FILE_ID") or drive.get(
        "source_archive_file_id"
    )

    return StorageConfig(
        project_root=root,
        config_path=source,
        drive_root=drive_root,
        drive_mount_point=mount,
        drive_dataset_dir=drive_path("datasets_dir"),
        drive_checkpoints_dir=drive_path("checkpoints_dir"),
        drive_experiments_dir=drive_path("experiments_dir"),
        drive_evaluation_dir=drive_path("evaluation_dir"),
        source_archive_path=_configured_path(source_archive_path, root),
        source_archive_file_id=(str(source_archive_file_id) if source_archive_file_id else None),
        source_archive_name=str(drive.get("source_archive_name", "fsboard_proc_clean.zip")),
        test_npz_member=str(drive.get("test_npz_member", "fsboard_test_proc_clean.npz")),
        test_npz_override=_configured_path(env.get("FSBOARD_TEST_NPZ"), root),
        checkpoint_override=_configured_path(env.get("FSBOARD_CHECKPOINT"), root),
        dataset_dir_override=_configured_path(env.get("FSBOARD_DATASET_DIR"), root),
        test_npz_name=str(drive.get("test_npz_name", "fsboard_test_proc_clean.npz")),
        motion_checkpoint_name=str(drive.get("motion_checkpoint_name", "bigru_ctc_motion_best.pt")),
        local_motion_checkpoint=_configured_path(local_checkpoint, root),  # type: ignore[arg-type]
        local_test_npz=_configured_path(local_test_npz, root),
        windows_subset_dir=_configured_path(subset_dir, root),  # type: ignore[arg-type]
        local_evaluation_dir=_configured_path(local_evaluation, root),  # type: ignore[arg-type]
        local_experiments_dir=_configured_path(local_experiments, root),  # type: ignore[arg-type]
        colab_cache_root=_configured_path(cache_root, root),  # type: ignore[arg-type]
    )
