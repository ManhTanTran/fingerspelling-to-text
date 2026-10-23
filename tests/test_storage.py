from pathlib import Path

import pytest

from fingerspelling.storage import load_storage_config


def _write_config(path: Path, *, drive_root: str | None = None) -> None:
    root = "null" if drive_root is None else f"'{Path(drive_root).as_posix()}'"
    path.write_text(
        "\n".join(
            [
                "storage:",
                "  drive:",
                f"    root: {root}",
                "    root_env: FSBOARD_DRIVE_ROOT",
                "    mount_point: /content/drive",
                "    folder_name: TTNM_Fingerspelling",
                "    datasets_dir: datasets/FSboard",
                "    checkpoints_dir: checkpoints",
                "    experiments_dir: experiments",
                "    evaluation_dir: evaluation",
                "    test_npz_name: fsboard_test_proc_clean.npz",
                "    motion_checkpoint_name: bigru_ctc_motion_best.pt",
                "  local:",
                "    motion_checkpoint: checkpoints/bigru_ctc_motion_best.pt",
                "    windows_subset_dir: data/windows_subset",
                "    test_npz: null",
                "    evaluation_dir: results/recomputed",
                "    experiments_dir: results/experiments",
                "    colab_cache_root: /content/fsboard_cache",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_explicit_test_npz_has_cli_precedence_over_environment_and_drive(tmp_path):
    config_path = tmp_path / "storage.yaml"
    drive_root = tmp_path / "drive" / "TTNM_Fingerspelling"
    _write_config(config_path, drive_root=str(drive_root))
    explicit = tmp_path / "cli-test.npz"
    env_test = tmp_path / "env-test.npz"
    explicit.write_bytes(b"cli")
    env_test.write_bytes(b"env")
    drive_test = drive_root / "datasets" / "FSboard" / "fsboard_test_proc_clean.npz"
    drive_test.parent.mkdir(parents=True)
    drive_test.write_bytes(b"drive")

    storage = load_storage_config(
        config_path,
        project_root=tmp_path,
        environ={"FSBOARD_TEST_NPZ": str(env_test)},
    )

    assert storage.resolve_test_npz(explicit) == explicit.resolve()
    assert storage.resolve_test_npz() == env_test.resolve()


def test_drive_root_is_discovered_from_colab_mount_and_local_checkpoint_fallback(tmp_path):
    config_path = tmp_path / "storage.yaml"
    _write_config(config_path)
    mount = tmp_path / "drive-mount"
    drive_root = mount / "MyDrive" / "TTNM_Fingerspelling"
    dataset = drive_root / "datasets" / "FSboard"
    dataset.mkdir(parents=True)
    drive_npz = dataset / "fsboard_test_proc_clean.npz"
    drive_npz.write_bytes(b"npz")
    local_checkpoint = tmp_path / "checkpoints" / "bigru_ctc_motion_best.pt"
    local_checkpoint.parent.mkdir()
    local_checkpoint.write_bytes(b"checkpoint")

    storage = load_storage_config(
        config_path,
        project_root=tmp_path,
        environ={"FSBOARD_DRIVE_MOUNT": str(mount)},
    )

    assert storage.drive_root == drive_root.resolve()
    assert storage.resolve_test_npz() == drive_npz.resolve()
    assert storage.resolve_checkpoint() == local_checkpoint.resolve()


def test_cli_test_npz_must_exist(tmp_path):
    config_path = tmp_path / "storage.yaml"
    _write_config(config_path)
    storage = load_storage_config(config_path, project_root=tmp_path, environ={})

    with pytest.raises(FileNotFoundError, match="test NPZ not found"):
        storage.resolve_test_npz(tmp_path / "missing.npz")
