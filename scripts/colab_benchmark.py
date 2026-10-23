"""Run the P3 FSboard smoke gate, full 126D evaluation, and direct CSV comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fingerspelling.evaluation.comparison import compare_prediction_csvs
from fingerspelling.storage import extract_npz_member, load_storage_config
from fingerspelling.storage.integrity import copy_verified, sha256_file


def mount_google_drive(mount_point: Path) -> None:
    if (mount_point / "MyDrive").is_dir():
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive is not mounted. Run this script in Colab or set "
            "FSBOARD_DRIVE_ROOT and pass --no-mount-drive."
        ) from exc
    drive.mount(str(mount_point))
    if not (mount_point / "MyDrive").is_dir():
        raise RuntimeError(f"Drive mount did not expose MyDrive at {mount_point / 'MyDrive'}")


def _hash_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def download_drive_archive(file_id: str, destination: Path) -> dict[str, Any]:
    """Probe actual Drive media access, then download and validate it in Colab scratch."""
    started = time.perf_counter()
    try:
        from google.colab import auth
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise RuntimeError(
            "Drive file-ID retrieval needs Colab plus google-api-python-client and google-auth; "
            "install the project with `pip install -e '.[colab]'`."
        ) from exc

    auth.authenticate_user()
    credentials, _ = google.auth.default()
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    # Verify content access with an authenticated, four-byte Range request.
    # Use a streaming authorized session so an ignored Range header cannot pull
    # the entire multi-gigabyte archive into memory during the access probe.
    probe_request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with AuthorizedSession(credentials).get(
        probe_request.uri,
        headers={"Range": "bytes=0-3", "Accept-Encoding": "identity"},
        stream=True,
        timeout=60,
    ) as response:
        if response.status_code != 206:
            raise IOError(
                "Authenticated Drive range probe did not return HTTP 206; "
                f"received HTTP {response.status_code}."
            )
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith("bytes 0-3/"):
            raise IOError(f"Unexpected Drive Content-Range header: {content_range!r}")
        probe_bytes = response.raw.read(4, decode_content=False)
    zip_signatures = {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}
    if len(probe_bytes) != 4 or probe_bytes not in zip_signatures:
        raise IOError(
            "Authenticated Drive content probe did not return a 4-byte ZIP signature; "
            f"received {len(probe_bytes)} bytes ({probe_bytes[:16].hex()})."
        )
    range_probe = {
        "range": "bytes=0-3",
        "bytes_read": len(probe_bytes),
        "signature_hex": probe_bytes.hex(),
        "content_range": content_range,
        "data_access_verified": True,
    }
    print(
        "Authenticated Drive media read verified: "
        f"file_id={file_id}, range={range_probe['range']}, "
        f"ZIP signature={range_probe['signature_hex']}.",
        flush=True,
    )
    metadata = service.files().get(
        fileId=file_id,
        fields="id,name,mimeType,size,md5Checksum,capabilities(canDownload)",
        supportsAllDrives=True,
    ).execute()
    if metadata.get("mimeType") not in {
        "application/zip", "application/x-zip-compressed", "application/octet-stream"
    }:
        raise ValueError(
            f"Drive source is not a ZIP/binary artifact: {metadata.get('mimeType')}"
        )
    if metadata.get("capabilities", {}).get("canDownload") is False:
        raise PermissionError(f"Drive user cannot download source file {file_id}")
    expected_size = int(metadata["size"])
    expected_md5 = metadata.get("md5Checksum")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size == expected_size:
        cached_sha256, cached_md5 = _hash_file(destination)
        if expected_md5 is None or cached_md5 == expected_md5:
            return {
                "id": metadata["id"], "name": metadata["name"],
                "mime_type": metadata["mimeType"], "size_bytes": expected_size,
                "drive_md5": expected_md5, "downloaded_sha256": cached_sha256,
                "cache_path": str(destination), "reused_cache": True,
                "range_probe": range_probe,
                "download_seconds": time.perf_counter() - started,
            }

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

        with os.fdopen(fd, "wb") as stream:
            downloader = MediaIoBaseDownload(stream, request, chunksize=32 * 1024 * 1024)
            done = False
            last_reported = -1
            while not done:
                status, done = downloader.next_chunk()
                if status is not None:
                    percent = int(status.progress() * 100)
                    if percent >= last_reported + 10 or done:
                        print(f"Drive archive download: {percent}%", flush=True)
                        last_reported = percent
            stream.flush()
            os.fsync(stream.fileno())
        actual_size = temp_path.stat().st_size
        if actual_size != expected_size:
            raise IOError(f"Drive archive size mismatch: expected {expected_size}, got {actual_size}")
        archive_sha256, actual_md5 = _hash_file(temp_path)
        if expected_md5 and actual_md5 != expected_md5:
            raise IOError("Drive archive MD5 checksum mismatch")
        os.replace(temp_path, destination)
        return {
            "id": metadata["id"], "name": metadata["name"],
            "mime_type": metadata["mimeType"], "size_bytes": actual_size,
            "drive_md5": expected_md5, "downloaded_md5": actual_md5,
            "downloaded_sha256": archive_sha256,
            "cache_path": str(destination), "reused_cache": False,
            "range_probe": range_probe,
            "download_seconds": time.perf_counter() - started,
        }
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _run_evaluation(
    *,
    repo_root: Path,
    storage_config: Path,
    checkpoint: Path,
    test_npz: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
    test_npz_sha256: str,
    checkpoint_sha256: str,
    max_samples: int | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "evaluate_checkpoint.py"),
        "--storage-config", str(storage_config),
        "--checkpoint", str(checkpoint),
        "--test-npz", str(test_npz),
        "--output-dir", str(output_dir),
        "--batch-size", str(batch_size),
        "--device", device,
        "--test-npz-sha256", test_npz_sha256,
        "--checkpoint-sha256", checkpoint_sha256,
        "--trusted-local-npz",
    ]
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    started = time.perf_counter()
    subprocess.run(command, check=True, cwd=repo_root)
    elapsed = time.perf_counter() - started
    metrics_path = output_dir / "126d_test_metrics.json"
    predictions_path = output_dir / "126d_test_predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise RuntimeError("evaluation did not produce expected 126D metrics and prediction files")
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if report.get("model_version") != "bigru_ctc_126d":
        raise ValueError("configured checkpoint is not the 126D BiGRU-CTC baseline")
    report["runtime"]["subprocess_wall_seconds"] = elapsed
    return {
        "report": report,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _persist_files(paths: list[Path], destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    return [copy_verified(path, destination / path.name) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-config", type=Path)
    parser.add_argument("--drive-mount", type=Path, help="override YAML/environment Drive mount path")
    parser.add_argument("--drive-root", type=Path, help="mounted TTNM_Fingerspelling folder override")
    parser.add_argument("--no-mount-drive", action="store_true", help="use an already-mounted/local Drive path")
    parser.add_argument("--test-npz", type=Path, help="processed test NPZ if already extracted on Drive")
    parser.add_argument("--source-archive", type=Path, help="ZIP path on mounted Drive; defaults to configured Drive archive ID")
    parser.add_argument("--source-archive-file-id", help="Drive ZIP file ID; defaults to storage.yaml")
    parser.add_argument("--npz-member", help="archive member; defaults to storage.yaml")
    parser.add_argument("--checkpoint", type=Path, help="126D checkpoint; defaults to Drive then local copy")
    parser.add_argument("--cache-root", type=Path, help="Colab scratch root; defaults to storage.yaml")
    parser.add_argument("--no-copy-to-cache", action="store_true", help="read direct NPZ/checkpoint from Drive")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-count", type=int, default=24, help="smoke sample count, from 16 to 32")
    parser.add_argument("--smoke-only", action="store_true", help="run smoke and reference comparison, then stop")
    parser.add_argument("--expected-samples", type=int, default=12515)
    parser.add_argument("--reference-csv", type=Path, help="defaults to results/reference/baseline_126d_predictions.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if args.batch_size < 1:
            raise ValueError("batch size must be positive")
        if not 16 <= args.smoke_count <= 32:
            raise ValueError("--smoke-count must be between 16 and 32")
        if args.expected_samples < 1:
            raise ValueError("--expected-samples must be positive")

        repo_root = Path(__file__).resolve().parents[1]
        if args.drive_mount is not None:
            os.environ["FSBOARD_DRIVE_MOUNT"] = str(args.drive_mount.expanduser())
        if args.drive_root is not None:
            os.environ["FSBOARD_DRIVE_ROOT"] = str(args.drive_root.expanduser())
        if args.source_archive_file_id:
            os.environ["FSBOARD_SOURCE_ARCHIVE_FILE_ID"] = args.source_archive_file_id

        storage = load_storage_config(args.storage_config, project_root=repo_root)
        if not args.no_mount_drive and not os.environ.get("FSBOARD_DRIVE_ROOT"):
            mount_google_drive(storage.drive_mount_point)
        storage = load_storage_config(args.storage_config, project_root=repo_root)
        if storage.drive_root is None:
            raise RuntimeError("Drive root unresolved; set --drive-root or FSBOARD_DRIVE_ROOT")
        for path in (
            storage.drive_dataset_dir,
            storage.drive_checkpoints_dir,
            storage.drive_experiments_dir,
            storage.drive_evaluation_dir,
        ):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

        cache_root = (args.cache_root or storage.colab_cache_root).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        reference_csv = (args.reference_csv or repo_root / "results" / "reference" / "baseline_126d_predictions.csv").expanduser().resolve()
        if not reference_csv.is_file():
            raise FileNotFoundError(f"reference prediction CSV not found: {reference_csv}")

        source_npz: Path | None = None
        archive_record: dict[str, Any] | None = None
        extraction_record: dict[str, Any] | None = None
        force_archive = args.source_archive is not None or args.source_archive_file_id is not None
        if not force_archive:
            try:
                source_npz = storage.resolve_test_npz(args.test_npz)
            except FileNotFoundError:
                if (
                    args.test_npz is not None
                    or storage.test_npz_override is not None
                    or storage.dataset_dir_override is not None
                    or storage.local_test_npz is not None
                ):
                    raise

        if source_npz is None:
            archive_path_arg = args.source_archive
            member_name = args.npz_member or storage.test_npz_member
            archive_file_id = args.source_archive_file_id
            if archive_path_arg is None and archive_file_id is None:
                archive_path_arg = storage.source_archive_path
                archive_file_id = storage.source_archive_file_id
            archive_tmp_root = cache_root / "source_archives"
            archive_tmp_root.mkdir(parents=True, exist_ok=True)
            if archive_path_arg is not None:
                archive_path = archive_path_arg.expanduser().resolve()
                if not archive_path.is_file():
                    raise FileNotFoundError(f"source archive not found: {archive_path}")
                archive_started = time.perf_counter()
                archive_sha256 = sha256_file(archive_path)
                archive_record = {
                    "origin": "mounted_drive_path",
                    "name": archive_path.name,
                    "size_bytes": archive_path.stat().st_size,
                    "sha256": archive_sha256,
                    "path": str(archive_path),
                    "source_hash_seconds": time.perf_counter() - archive_started,
                }
            elif archive_file_id:
                archive_name = storage.source_archive_name
                archive_path = archive_tmp_root / archive_name
                free_bytes = shutil.disk_usage(cache_root).free
                if free_bytes < 7 * 1024**3:
                    raise OSError(
                        f"Colab scratch has {free_bytes} free bytes; at least 7 GiB is needed "
                        "to download the 6.36 GB Drive ZIP plus extract the test NPZ."
                    )
                archive_record = download_drive_archive(archive_file_id, archive_path)
                archive_record["origin"] = "drive_file_id"
            else:
                raise FileNotFoundError(
                    "No processed test NPZ was found and no source archive was configured. "
                    "Set --source-archive-file-id or FSBOARD_SOURCE_ARCHIVE_FILE_ID."
                )

            extracted_npz = cache_root / "datasets" / Path(member_name).name
            extraction_started = time.perf_counter()
            extraction_record = extract_npz_member(archive_path, member_name, extracted_npz)
            extraction_record["extract_seconds"] = time.perf_counter() - extraction_started
            assert storage.drive_dataset_dir is not None
            persist_started = time.perf_counter()
            drive_npz_copy = copy_verified(
                extracted_npz,
                storage.drive_dataset_dir / extracted_npz.name,
            )
            drive_npz_copy["persist_seconds"] = time.perf_counter() - persist_started
            extraction_record["drive_npz_copy"] = drive_npz_copy
            source_npz = extracted_npz

        source_checkpoint = storage.resolve_checkpoint(args.checkpoint)
        if args.no_copy_to_cache:
            if extraction_record is not None:
                raise ValueError("--no-copy-to-cache cannot be used with archive extraction")
            runtime_npz, runtime_checkpoint = source_npz, source_checkpoint
            data_record = {
                "source": str(source_npz), "destination": str(source_npz),
                "size_bytes": source_npz.stat().st_size,
                "sha256": sha256_file(source_npz), "copied": False, "verified": True,
            }
            checkpoint_record = {
                "source": str(source_checkpoint), "destination": str(source_checkpoint),
                "size_bytes": source_checkpoint.stat().st_size,
                "sha256": sha256_file(source_checkpoint), "copied": False, "verified": True,
            }
        else:
            runtime_npz = cache_root / "datasets" / source_npz.name
            runtime_checkpoint = cache_root / "checkpoints" / source_checkpoint.name
            cache_copy_started = time.perf_counter()
            data_record = copy_verified(source_npz, runtime_npz, overwrite=True)
            data_record["cache_copy_seconds"] = time.perf_counter() - cache_copy_started
            checkpoint_copy_started = time.perf_counter()
            checkpoint_record = copy_verified(source_checkpoint, runtime_checkpoint, overwrite=True)
            checkpoint_record["cache_copy_seconds"] = time.perf_counter() - checkpoint_copy_started

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        assert storage.drive_evaluation_dir is not None
        assert storage.drive_experiments_dir is not None
        evaluation_target = storage.drive_evaluation_dir / run_id
        experiment_target = storage.drive_experiments_dir / run_id
        evaluation_target.mkdir(parents=True, exist_ok=False)
        experiment_target.mkdir(parents=True, exist_ok=False)
        local_output_root = cache_root / "evaluation" / run_id
        smoke_dir = local_output_root / "smoke"
        checkpoint_hash = checkpoint_record["sha256"]
        dataset_hash = data_record["sha256"]

        smoke_started = time.perf_counter()
        smoke_output = _run_evaluation(
            repo_root=repo_root,
            storage_config=storage.config_path,
            checkpoint=runtime_checkpoint,
            test_npz=runtime_npz,
            output_dir=smoke_dir,
            batch_size=args.batch_size,
            device=args.device,
            test_npz_sha256=dataset_hash,
            checkpoint_sha256=checkpoint_hash,
            max_samples=args.smoke_count,
        )
        smoke_comparison = compare_prediction_csvs(
            smoke_output["predictions_path"],
            reference_csv,
            max_rows=args.smoke_count,
        )
        smoke_runtime_seconds = time.perf_counter() - smoke_started
        smoke_report = smoke_output["report"]
        smoke_report_path = smoke_dir / "reference_comparison.json"
        _write_json(smoke_report_path, smoke_comparison)
        smoke_passed = (
            smoke_report.get("n_test") == args.smoke_count
            and smoke_comparison["row_counts_match"]
            and smoke_comparison["all_reference_text_matches"]
            and smoke_comparison["all_predictions_match"]
        )

        print(json.dumps({
            "gate": "smoke",
            "status": "passed" if smoke_passed else "mismatch",
            "sample_count": args.smoke_count,
            "metrics": smoke_report["metrics"],
            "direct_reference_comparison": smoke_comparison,
            "runtime": smoke_report["runtime"],
        }, indent=2, ensure_ascii=False), flush=True)

        common_report = {
            "schema_version": 1,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "smoke_passed" if smoke_passed else "smoke_mismatch",
            "dataset_source": {
                "archive": archive_record,
                "extraction": extraction_record,
                "test_npz_copy": data_record,
            },
            "checkpoint": {
                "path": str(runtime_checkpoint),
                "sha256": checkpoint_hash,
                "copy": checkpoint_record,
            },
            "reference_artifact": {
                "path": str(reference_csv),
                "sha256": sha256_file(reference_csv),
            },
            "smoke_test": {
                "sample_count": args.smoke_count,
                "metrics": smoke_report["metrics"],
                "runtime": smoke_report["runtime"],
                "wall_seconds": smoke_runtime_seconds,
                "direct_reference_comparison": smoke_comparison,
            },
            "runtime_environment": smoke_report["runtime"],
            "coordinate_note": (
                "Processed NPZ inference uses the existing 126D preprocessing. "
                "No RGB coordinate adapter or checkpoint was changed."
            ),
        }

        if not smoke_passed:
            common_report["status"] = "smoke_mismatch"
            reproducibility_path = local_output_root / "p3_reproducibility_report.json"
            _write_json(reproducibility_path, common_report)
            persisted = _persist_files(
                [smoke_output["metrics_path"], smoke_output["predictions_path"], smoke_report_path, reproducibility_path],
                evaluation_target,
            )
            experiment_copy = copy_verified(reproducibility_path, experiment_target / reproducibility_path.name)
            print(json.dumps({
                "status": "smoke_mismatch",
                "smoke_metrics": smoke_report["metrics"],
                "comparison": smoke_comparison,
                "evaluation_files": persisted,
                "experiment_report": experiment_copy,
            }, indent=2, ensure_ascii=False))
            raise RuntimeError("smoke test did not match the saved reference; full evaluation was not started")

        if args.smoke_only:
            common_report["status"] = "smoke_passed_full_not_run"
            reproducibility_path = local_output_root / "p3_reproducibility_report.json"
            _write_json(reproducibility_path, common_report)
            persisted = _persist_files(
                [smoke_output["metrics_path"], smoke_output["predictions_path"], smoke_report_path, reproducibility_path],
                evaluation_target,
            )
            experiment_copy = copy_verified(reproducibility_path, experiment_target / reproducibility_path.name)
            print(json.dumps({
                "status": common_report["status"],
                "smoke_metrics": smoke_report["metrics"],
                "comparison": smoke_comparison,
                "evaluation_files": persisted,
                "experiment_report": experiment_copy,
            }, indent=2, ensure_ascii=False))
            return

        full_dir = local_output_root / "full"
        full_started = time.perf_counter()
        full_output = _run_evaluation(
            repo_root=repo_root,
            storage_config=storage.config_path,
            checkpoint=runtime_checkpoint,
            test_npz=runtime_npz,
            output_dir=full_dir,
            batch_size=args.batch_size,
            device=args.device,
            test_npz_sha256=dataset_hash,
            checkpoint_sha256=checkpoint_hash,
        )
        full_runtime_seconds = time.perf_counter() - full_started
        full_comparison = compare_prediction_csvs(
            full_output["predictions_path"],
            reference_csv,
        )
        full_comparison_path = full_dir / "reference_comparison.json"
        _write_json(full_comparison_path, full_comparison)
        full_report = full_output["report"]
        success = (
            full_report.get("n_test") == args.expected_samples
            and full_report.get("n_valid_test_examples") == args.expected_samples
            and full_comparison["row_counts_match"]
            and full_comparison["all_reference_text_matches"]
            and full_comparison["all_predictions_match"]
        )
        common_report.update({
            "status": "success" if success else "review_required",
            "expected_test_samples": args.expected_samples,
            "full_evaluation": {
                "n_test": full_report.get("n_test"),
                "n_valid_test_examples": full_report.get("n_valid_test_examples"),
                "metrics": full_report["metrics"],
                "test_data_stats": full_report["test_data_stats"],
                "runtime": full_report["runtime"],
                "wall_seconds": full_runtime_seconds,
                "direct_reference_comparison": full_comparison,
            },
            "wrapper_total_seconds": time.perf_counter() - started,
        })
        reproducibility_path = local_output_root / "p3_reproducibility_report.json"
        _write_json(reproducibility_path, common_report)
        evaluation_files = _persist_files(
            [
                smoke_output["metrics_path"], smoke_output["predictions_path"], smoke_report_path,
                full_output["metrics_path"], full_output["predictions_path"], full_comparison_path,
                reproducibility_path,
            ],
            evaluation_target,
        )
        experiment_copy = copy_verified(reproducibility_path, experiment_target / reproducibility_path.name)
        print(json.dumps({
            "status": common_report["status"],
            "run_id": run_id,
            "smoke_metrics": smoke_report["metrics"],
            "smoke_comparison": smoke_comparison,
            "full_metrics": full_report["metrics"],
            "full_sample_count": full_report["n_test"],
            "full_valid_sample_count": full_report["n_valid_test_examples"],
            "full_comparison": full_comparison,
            "runtime": full_report["runtime"],
            "evaluation_files": evaluation_files,
            "experiment_report": experiment_copy,
        }, indent=2, ensure_ascii=False))
        if not success:
            raise RuntimeError(
                "full run completed but sample-count or reference-comparison checks need review; see p3_reproducibility_report.json"
            )
    except Exception as exc:
        parser.exit(2, f"colab_benchmark: {exc}\n")


if __name__ == "__main__":
    main()
