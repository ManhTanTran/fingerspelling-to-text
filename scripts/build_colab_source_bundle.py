"""Build the small, data-free source bundle used by the P3 Colab launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_NAME = "fingerspelling-to-text"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_files() -> list[Path]:
    required = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "configs" / "storage.yaml",
        REPO_ROOT / "scripts" / "colab_benchmark.py",
        REPO_ROOT / "scripts" / "evaluate_checkpoint.py",
        REPO_ROOT / "scripts" / "export_npz_subset.py",
        REPO_ROOT / "results" / "reference" / "baseline_126d_predictions.csv",
        REPO_ROOT / "checkpoints" / "bigru_ctc_motion_best.pt",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required Colab bundle files are missing: {missing}")

    source_modules = sorted((REPO_ROOT / "src" / "fingerspelling").rglob("*.py"))
    return sorted(set(required + source_modules))


def build_bundle(output_path: str | Path | None = None) -> dict[str, object]:
    destination = Path(output_path or REPO_ROOT / "colab" / "p3_source_bundle.zip")
    if not destination.is_absolute():
        destination = REPO_ROOT / destination
    destination = destination.resolve()
    if destination == REPO_ROOT or REPO_ROOT not in destination.parents:
        raise ValueError("bundle output must be inside the repository")

    files = _source_files()
    manifest_files: list[dict[str, object]] = []
    contents: list[tuple[str, bytes]] = []
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        payload = path.read_bytes()
        manifest_files.append({
            "path": relative,
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        })
        contents.append((relative, payload))

    manifest = {
        "schema_version": 1,
        "purpose": "P3 Colab smoke and full benchmark; contains no FSboard NPZ dataset.",
        "drive_source_file_id": "1dhcrbZeb3-5IshIkzatVmNO3MOTVFjJW",
        "files": manifest_files,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, payload in contents:
            info = zipfile.ZipInfo(f"{ROOT_NAME}/{relative}", date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        manifest_info = zipfile.ZipInfo(f"{ROOT_NAME}/COLAB_BUNDLE_MANIFEST.json", date_time=(2020, 1, 1, 0, 0, 0))
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.external_attr = 0o644 << 16
        archive.writestr(manifest_info, manifest_bytes, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)

    bundle_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": bundle_sha,
        "included_files": [entry["path"] for entry in manifest_files],
        "contains_dataset_npz": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="defaults to colab/p3_source_bundle.zip")
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
