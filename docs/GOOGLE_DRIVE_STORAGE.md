# Google Drive storage and Colab benchmark

Google Drive is the primary store for the FSboard dataset, large NPZ artifacts, checkpoint backups, experiment reports, and evaluation outputs. Windows keeps the 126D checkpoint needed by FastAPI and can receive a 16–32 sample NPZ subset for local checks. FastAPI loads its checkpoint at startup and never contacts Drive during a request.

## Drive layout

Create this structure under `MyDrive`:

```text
MyDrive/TTNM_Fingerspelling/
├── datasets/FSboard/
│   ├── fingerspelling_fsboard_proc_clean.zip   # optional source archive
│   └── fsboard_test_proc_clean.npz             # extracted test split
├── checkpoints/
│   └── bigru_ctc_motion_best.pt
├── experiments/
└── evaluation/
```

The original notebook's source archive ID is `1dhcrbZeb3-5IshIkzatVmNO3MOTVFjJW`. Drive metadata identifies it as `fingerspelling_fsboard_proc_clean.zip`, MIME type `application/zip`, size 6,361,248,095 bytes. `configs/storage.yaml` contains this ID, so users do not need to copy the full archive to Windows or manually enter a file URL. Desktop metadata access did not verify media access. After you sign in through Colab, the runner makes an authenticated streaming `Range: bytes=0-3` request and requires HTTP 206, the expected content range, and a ZIP signature before starting the large download. It fails fast if the signed-in account cannot read file contents.

The processed NPZ format uses pickle-backed object arrays. Only load this artifact with `--trusted-local-npz` after obtaining it from the trusted FSboard source. Never load an API upload as an NPZ.

## Path configuration

`configs/storage.yaml` supplies the defaults. CLI arguments override environment variables, which override YAML. A mounted Colab Drive is detected at `/content/drive/MyDrive/TTNM_Fingerspelling`; on Windows, Drive need not be mounted to run the API or evaluate a local subset.

| Environment variable | Purpose |
| --- | --- |
| `FSBOARD_STORAGE_CONFIG` | Alternate storage YAML; `--storage-config` takes precedence. |
| `FSBOARD_DRIVE_ROOT` | Absolute path to the `TTNM_Fingerspelling` folder itself. |
| `FSBOARD_DRIVE_MOUNT` | Drive mount point; defaults to `/content/drive`. |
| `FSBOARD_TEST_NPZ` | Exact processed test NPZ path. |
| `FSBOARD_DATASET_DIR` | Directory containing `fsboard_test_proc_clean.npz`. |
| `FSBOARD_SOURCE_ARCHIVE` | Local or mounted-Drive ZIP path. |
| `FSBOARD_SOURCE_ARCHIVE_FILE_ID` | Drive file ID for the source ZIP. |
| `FSBOARD_CHECKPOINT` | Checkpoint path for evaluation/inference scripts. |
| `FSBOARD_LOCAL_CHECKPOINT` | Windows 126D checkpoint used by FastAPI and the Drive backup script. |
| `FSBOARD_EVALUATION_DIR` | Local evaluation output directory. |
| `FSBOARD_EXPERIMENTS_DIR` | Local experiment output directory. |
| `FSBOARD_WINDOWS_SUBSET_DIR` | Local small-subset output directory. |
| `FSBOARD_LOCAL_CACHE_ROOT` | Colab scratch directory for verified input copies. |

Do not put credentials or access tokens in YAML. Colab uses its interactive Google authentication flow.

## Quick start with the P3 Colab launcher

From Windows, regenerate the small source bundle when repository code changes:

```powershell
python scripts/build_colab_source_bundle.py
```

Then upload `notebooks/p3_colab_launcher.ipynb` to Colab, sign in, and run the cells in order. When prompted, select `colab/p3_source_bundle.zip`. The bundle is about 2 MiB and includes only the benchmark source, 126D checkpoint, and reference predictions; it contains no dataset NPZ. The notebook installs the Colab dependencies and starts `scripts/colab_benchmark.py` with a 24-sample gate. Select a GPU runtime if available; `--device auto` records the selected hardware.

The runner first verifies a four-byte content range through the authenticated Drive API, then downloads the source ZIP into Colab scratch and validates it. It extracts only the test NPZ. The downloaded archive and extracted NPZ stay in Drive/Colab storage; the workflow does not require a full dataset download to Windows.

## Run the ordered P3 benchmark in Colab

Make the repository available in Colab, then install the project and Drive API dependencies:

```python
%cd /content/fingerspelling-to-text
!python -m pip install -e ".[colab]"
```

Run the benchmark from the repository root:

```python
!python scripts/colab_benchmark.py --smoke-count 24 --batch-size 16 --device auto
```

The script mounts Drive when needed and authenticates the Drive API. If the processed NPZ already exists at `MyDrive/TTNM_Fingerspelling/datasets/FSboard/fsboard_test_proc_clean.npz`, it uses that file as the source. Otherwise it downloads the configured ZIP by Drive file ID into `/content/fsboard_cache/source_archives/`, validates the Drive-reported size and MD5, computes SHA-256, and extracts only `fsboard_test_proc_clean.npz` to Colab scratch. Extraction streams one member, checks its uncompressed size and ZIP CRC, and records SHA-256. The derived NPZ is then copied to the Drive dataset folder with a SHA-256 verified copy. The source ZIP is not changed.

The dataset and checkpoint are copied to `/content/fsboard_cache/` and evaluated from Colab-local storage for speed. The archive route needs at least 7 GiB free in Colab scratch. Keep the Drive backup `checkpoints/bigru_ctc_motion_best.pt` available; if it is not there, create it from the Windows checkpoint using the backup command below.

The first stage evaluates 24 examples (choose any value from 16 through 32) and compares fresh predictions directly, row by row, with `results/reference/baseline_126d_predictions.csv`. The comparison checks both reference-label alignment and exact predictions. If it does not match, the script saves the smoke metrics, prediction CSV, comparison, and reproducibility report to Drive, then stops before full evaluation. To run only this gate:

```python
!python scripts/colab_benchmark.py --smoke-count 24 --smoke-only --batch-size 16 --device auto
```

When the smoke gate passes, the default run evaluates all 12,515 expected valid examples. It calculates CER, WER, Exact Match, and character- and word-level substitutions, deletions, and insertions. It compares all generated references and predictions directly against the saved prediction artifact; metrics recalculated from a CSV alone do not count as this comparison. A full run is marked `success` only if it has 12,515 valid rows and exact row-wise matches. Otherwise it saves a `review_required` report with row mismatch details.

Each run creates a unique folder under both `evaluation/<run_id>/` and `experiments/<run_id>/`. The outputs include smoke/full metrics JSON, fresh prediction CSVs, direct comparison JSON, and `p3_reproducibility_report.json` with artifact SHA-256, archive metadata, model/checkpoint information, data filtering statistics, device/software versions, and measured data-load, checkpoint-load, inference, and wall-clock runtimes. Existing P0/P1/P2 reference and recomputed outputs are left in place.

To specify a source path or override the Drive ID:

```python
!python scripts/colab_benchmark.py \
  --source-archive-file-id 1dhcrbZeb3-5IshIkzatVmNO3MOTVFjJW \
  --npz-member fsboard_test_proc_clean.npz \
  --smoke-count 24 --batch-size 16 --device auto
```

If the NPZ is already extracted on Drive, it can be selected explicitly:

```python
!python scripts/colab_benchmark.py \
  --test-npz /content/drive/MyDrive/TTNM_Fingerspelling/datasets/FSboard/fsboard_test_proc_clean.npz \
  --checkpoint /content/drive/MyDrive/TTNM_Fingerspelling/checkpoints/bigru_ctc_motion_best.pt
```

`--no-copy-to-cache` reads existing Drive files directly and can be slower. It is not available when the script needs to extract from the ZIP. Use `--cache-root` to choose another Colab scratch directory.

This test starts from the processed NPZ and reuses the existing 126D preprocessing and checkpoint. It does not run MediaPipe or modify coordinates. A separate RGB video test is still required before claiming coordinate compatibility.

## Back up the local 126D API checkpoint

Keep `checkpoints/bigru_ctc_motion_best.pt` on Windows for FastAPI and copy its backup to Drive when the Drive folder is mounted or available as a synced folder:

```powershell
$env:FSBOARD_DRIVE_ROOT = "G:\My Drive\TTNM_Fingerspelling"
python scripts/sync_checkpoint_to_drive.py
```

The script computes SHA-256, refuses to replace a different existing backup unless `--overwrite` is passed, and writes a `.sha256.json` sidecar. This is a manual backup. FastAPI does not use Drive.

## Export a 16–32 sample Windows subset

After the NPZ is on Drive, create a new 24-sample artifact and manifest without changing the source:

```python
!python scripts/export_npz_subset.py \
  --input-npz /content/drive/MyDrive/TTNM_Fingerspelling/datasets/FSboard/fsboard_test_proc_clean.npz \
  --output /content/drive/MyDrive/TTNM_Fingerspelling/datasets/FSboard/fsboard_test_subset.npz \
  --count 24 --seed 42 --trusted-local-npz
```

Download only `fsboard_test_subset.npz` to Windows, for example into `data/windows_subset/`. The sample is useful for checking local loading and preprocessing, not for a statistically meaningful benchmark.

Evaluate the local subset using the retained 126D checkpoint:

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint checkpoints/bigru_ctc_motion_best.pt `
  --test-npz data/windows_subset/fsboard_test_subset.npz `
  --trusted-local-npz
```

`--test-npz` can name any trusted NPZ path. It is optional when `FSBOARD_TEST_NPZ` points at a file or the configured local subset exists. Use `--output-dir` to select another local results directory.

## Prepare and test the Windows video pipeline

The official MediaPipe Hand Landmarker model is prepared locally at `models/hand_landmarker.task` (SHA-256 `FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1`) and is ignored by Git. The repository-local `.venv-p3` has the video/API extras installed. To recreate it with Python 3.10:

```powershell
py -3.10 -m venv .venv-p3
.\.venv-p3\Scripts\Activate.ps1
python -m pip install -e ".[video,api]"
```

When a real signed video is available, test the CLI:

```powershell
python scripts/predict_video.py `
  --video "D:\path\to\sample.mp4" `
  --checkpoint checkpoints/bigru_ctc_motion_best.pt `
  --hand-model models/hand_landmarker.task
```

Then run FastAPI locally and submit the same video to `POST /predict`:

```powershell
$env:FSBOARD_CHECKPOINT = "checkpoints/bigru_ctc_motion_best.pt"
$env:FSBOARD_HAND_MODEL = "models/hand_landmarker.task"
python -m uvicorn --app-dir . backend.app.main:app --host 127.0.0.1 --port 8000
```

The environment has been checked: MediaPipe imports, the `.task` model constructs a HandLandmarker, FastAPI `/health` reports ready, and `/model-info` reports input dimension 126. The video must still be a user-provided test sample; no real video has been supplied, so video decoding, landmark detections, 126D feature parity on real footage, and end-to-end predictions remain unverified. No preprocessing or checkpoint change is made as part of this storage and benchmark workflow.
