"""Safely import the known Task 3 checkpoint and reference artifacts.

Only an explicit allowlist of regular files is copied. The script does not
extract arbitrary ZIP members, execute archived scripts, or deserialize models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


ALLOWLIST = {
    "TASK3/Day2/dataset_audit_day2.md": "docs/reference/day2_dataset_audit.md",
    "TASK3/Day3/day3_results/bigru_ctc_full_best.pt": "checkpoints/bigru_ctc_full_best.pt",
    "TASK3/Day3/day3_results/fsboard_final_metrics.json": "results/reference/baseline_63d_metrics.json",
    "TASK3/Day3/day3_results/fsboard_test_predictions.csv": "results/reference/baseline_63d_predictions.csv",
    "TASK3/Day5-7/task3_days5_7_results/bigru_ctc_motion_best.pt": "checkpoints/bigru_ctc_motion_best.pt",
    "TASK3/Day5-7/task3_days5_7_results/task3_day5_ablation_metrics.json": "results/reference/baseline_126d_ablation_metrics.json",
    "TASK3/Day5-7/task3_days5_7_results/task3_day5_motion_predictions.csv": "results/reference/baseline_126d_predictions.csv",
    "TASK3/Day5-7/task3_days5_7_results/task3_day6_summary.json": "results/reference/day6_summary.json",
    "TASK3/Day5-7/task3_days5_7_results/task3_day6_error_by_group.csv": "results/reference/day6_error_by_group.csv",
    "TASK3/Day5-7/task3_days5_7_results/task3_day6_sample_errors.csv": "results/reference/day6_sample_errors.csv",
    "TASK3/Day5-7/task3_days5_7_results/task3_day6_top_error_cases.csv": "results/reference/day6_top_error_cases.csv",
    "TASK3/Day5-7/task3_days5_7_results/task3_day7_model_comparison.csv": "results/reference/day7_model_comparison.csv",
    "TASK3/Day5-7/task3_days5_7_results/task3_day7_research_decision.json": "results/reference/day7_research_decision.json",
    "TASK3/Day5-7/task3_days5_7_results/task3_day7_research_decision.md": "docs/reference/day7_research_decision.md",
}
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024


def safe_import(archive: Path, repo_root: Path, overwrite: bool = False) -> dict[str, object]:
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    repo_root = repo_root.resolve()
    imported: list[dict[str, str]] = []

    with zipfile.ZipFile(archive) as zf:
        if sum(info.file_size for info in zf.infolist()) > MAX_ARCHIVE_BYTES:
            raise ValueError(f"archive exceeds {MAX_ARCHIVE_BYTES} byte safety limit")
        corrupt_member = zf.testzip()
        if corrupt_member:
            raise ValueError(f"archive CRC check failed for {corrupt_member}")
        members = {info.filename: info for info in zf.infolist()}

        for archive_name, destination in ALLOWLIST.items():
            info = members.get(archive_name)
            if info is None:
                raise FileNotFoundError(f"required archive member missing: {archive_name}")
            member_path = PurePosixPath(archive_name)
            if member_path.is_absolute() or ".." in member_path.parts or info.is_dir():
                raise ValueError(f"unsafe archive member: {archive_name}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"symbolic link is not allowed: {archive_name}")

            output = (repo_root / Path(destination)).resolve()
            if repo_root not in output.parents:
                raise ValueError(f"destination escapes repository: {destination}")
            content = zf.read(info)
            digest = hashlib.sha256(content).hexdigest()
            if output.exists():
                existing_digest = hashlib.sha256(output.read_bytes()).hexdigest()
                if existing_digest != digest and not overwrite:
                    raise FileExistsError(
                        f"{output} already exists with different content; rerun with --overwrite"
                    )
                if existing_digest == digest:
                    imported.append({"archive_member": archive_name, "path": destination, "sha256": digest})
                    continue
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
            imported.append({"archive_member": archive_name, "path": destination, "sha256": digest})

    manifest = {
        "archive": str(archive.resolve()),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "imported_files": imported,
    }
    manifest_path = repo_root / "results" / "reference" / "import_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(safe_import(args.archive, args.repo, args.overwrite), indent=2))


if __name__ == "__main__":
    main()

