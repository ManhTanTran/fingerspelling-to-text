"""Check local reproduction requirements and optionally inventory source inputs.

This script only reads file metadata, hashes the notebook bytes, and lists ZIP
member names. It does not extract archives, load checkpoints, or execute code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


LOCAL_REQUIREMENTS = (
    "notebooks/original_task3.ipynb",
    "checkpoints/bigru_ctc_full_best.pt",
    "checkpoints/bigru_ctc_motion_best.pt",
)
EXPECTED_DATA = "fsboard_test_proc_clean.npz"
EXPECTED_HAND_MODEL = "hand_landmarker.task"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    root: Path,
    *,
    source_notebook: Path | None = None,
    source_archive: Path | None = None,
    test_npz: Path | None = None,
    hand_model: Path | None = None,
) -> dict[str, object]:
    local = {
        name: (root / Path(name)).is_file()
        for name in LOCAL_REQUIREMENTS
    }
    notebook_copy = root / "notebooks" / "original_task3.ipynb"
    notebook_result: dict[str, object] | None = None
    if source_notebook is not None:
        notebook_result = {
            "path": str(source_notebook.resolve()),
            "exists": source_notebook.is_file(),
        }
        if source_notebook.is_file():
            notebook_result["sha256"] = _sha256(source_notebook)
            notebook_result["copy_matches"] = (
                notebook_copy.is_file()
                and _sha256(source_notebook) == _sha256(notebook_copy)
            )

    archive_result: dict[str, object] | None = None
    if source_archive is not None:
        archive_result = {
            "path": str(source_archive.resolve()),
            "exists": source_archive.is_file(),
        }
        if source_archive.is_file():
            try:
                with zipfile.ZipFile(source_archive) as archive:
                    members = archive.namelist()
                archive_result["sha256"] = _sha256(source_archive)
                archive_result["member_count"] = len(members)
                archive_result["members"] = members
            except (OSError, zipfile.BadZipFile) as exc:
                archive_result["error"] = str(exc)

    data_path = test_npz or (root / "data" / EXPECTED_DATA)
    model_path = hand_model or (root / "models" / EXPECTED_HAND_MODEL)
    return {
        "root": str(root.resolve()),
        "local_requirements": local,
        "missing_local_requirements": [name for name, present in local.items() if not present],
        "source_notebook": notebook_result,
        "source_archive": archive_result,
        "processed_test_npz": {"path": str(data_path.resolve()), "present": data_path.is_file()},
        "mediapipe_hand_model": {"path": str(model_path.resolve()), "present": model_path.is_file()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to this repository)",
    )
    parser.add_argument("--notebook", type=Path, help="optional original notebook to compare with the preserved copy")
    parser.add_argument("--archive", type=Path, help="optional source ZIP to inventory without extracting")
    parser.add_argument("--test-npz", type=Path, help="processed test NPZ path, if supplied")
    parser.add_argument("--hand-model", type=Path, help="MediaPipe hand_landmarker.task path, if supplied")
    args = parser.parse_args()
    print(json.dumps(
        audit(
            args.root,
            source_notebook=args.notebook,
            source_archive=args.archive,
            test_npz=args.test_npz,
            hand_model=args.hand_model,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
