# FSboard Fingerspelling-to-Text

Research repository for **Confidence-Aware One-Hand Fingerspelling-to-Text System with Interactive Error Correction**. It contains the audited Task 3 notebook, strict checkpoint loading, notebook-compatible preprocessing, reference metric recomputation, experimental RGB video inference, a shared CLI/API inference service, FastAPI, and a React MVP.

## What is verified

- The original 45-cell notebook is preserved byte-for-byte at `notebooks/original_task3.ipynb`; no cell was run.
- Both supplied Task 3 checkpoints load strictly with `weights_only=True`. The recovered BiGRU-CTC architecture, 59 character tokens, 60 output classes, and blank index 0 match their state dictionaries.
- Notebook-compatible 63D normalization and 126D hand-plus-motion features have unit tests. Greedy CTC decoding and corpus metrics have unit tests.
- Metrics recalculated from the two supplied prediction CSVs match the updated ZIP artifacts. This is a CSV metric reproduction, not a fresh checkpoint evaluation.
- Video decoding, hand-side locking, feature conversion, API upload handling, and browser draft/confirm interactions have automated tests. The real-video acceptance flow and RGB coordinate compatibility have not been measured.
- Large FSboard inputs and archived run outputs are configured for Google Drive. The Colab benchmark uses verified temporary copies and keeps Drive source artifacts unchanged.

The processed FSboard NPZ is stored in Google Drive, not in this checkout. Its source archive is configured by Drive file ID and the Colab benchmark downloads it to temporary Colab storage, extracts only the test NPZ, and leaves the source archive unchanged. Checkpoint inference against the test set has not yet been run. The Windows MediaPipe `hand_landmarker.task` asset is prepared locally and excluded from Git. The current MediaPipe-to-FSboard coordinate mapping is explicitly a candidate adapter. P4 implementation tests do not establish real-video inference or coordinate parity.

## Baselines

| Baseline | Input | GRU | Classes | Blank | Test CSV CER | Test CSV WER | Exact match |
|---|---:|---|---:|---:|---:|---:|---:|
| 63D hand-only | 63 | 2-layer, bidirectional, hidden 128 | 60 | 0 | 10.4083% | 57.1322% | 33.0963% |
| 126D hand + motion | 126 | 2-layer, bidirectional, hidden 128 | 60 | 0 | 10.1006% | 55.7822% | 34.0392% |

Each row covers 12,515 predictions. These values were recomputed from the prediction CSVs in the supplied `TASK3.zip`; they are not recomputed from checkpoint inference in this repository.

## Setup

Use Python 3.10 or newer. Install the core package and development tools:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install optional video and API dependencies as needed:

```powershell
python -m pip install -e ".[video,api]"
```

OpenCV codec support depends on the installed OpenCV build. The video pipeline uses the MediaPipe Tasks HandLandmarker API and needs a local `.task` model file at `models/hand_landmarker.task`. Get a compatible bundle from the [official Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python); the code does not download it automatically.

## Artifacts and data

- The two `.pt` checkpoints imported from `TASK3.zip` are in the ignored local `checkpoints/` directory. Keep `checkpoints/bigru_ctc_motion_best.pt` on Windows for FastAPI; never load checkpoint files supplied by an API user.
- The source `TASK3.zip` is not copied into the repository. `scripts/import_task3_archive.py` documents the safe whitelist import path.
- The processed source is available in Drive as file ID `1dhcrbZeb3-5IshIkzatVmNO3MOTVFjJW`; `configs/storage.yaml` configures the Colab workflow to retrieve this ZIP and extract only `fsboard_test_proc_clean.npz`. Windows can use an exported 16–32 sample subset instead of downloading the full dataset. These notebook-format NPZs use pickle-backed object arrays; only load trusted artifacts.
- Do not commit datasets, large checkpoints, model assets, credentials, or user videos. See [data/README.md](data/README.md), [checkpoints/README.md](checkpoints/README.md), and [docs/GOOGLE_DRIVE_STORAGE.md](docs/GOOGLE_DRIVE_STORAGE.md).

## Commands

Recompute metrics from an imported prediction CSV:

```powershell
python scripts/evaluate_predictions.py --csv results/reference/baseline_126d_predictions.csv
```

Audit the local checkpoint/notebook copies and inventory the source archive without extracting it:

```powershell
python scripts/audit_artifacts.py `
  --notebook D:\Downloads\task-3-day-3-baseline-setup.ipynb `
  --archive D:\Downloads\TASK3.zip
```

Inspect a trusted local checkpoint:

```powershell
python scripts/inspect_checkpoint.py --checkpoint checkpoints/bigru_ctc_motion_best.pt
```

Run checkpoint inference on the trusted local subset, or pass any trusted NPZ path with `--test-npz`:

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint checkpoints/bigru_ctc_motion_best.pt `
  --test-npz data/windows_subset/fsboard_test_subset.npz `
  --trusted-local-npz
```

Build a small Colab source bundle, then upload `notebooks/p3_colab_launcher.ipynb` to Colab, sign in, upload the bundle when prompted, and run the cells. The notebook starts a 24-sample smoke test against the saved prediction artifact. If labels and predictions match, it evaluates all 12,515 expected examples, computes CER/WER/Exact Match and S/D/I, compares every generated row directly against the reference CSV, and saves a SHA-256/runtime reproducibility report to Drive:

```powershell
python scripts/build_colab_source_bundle.py
```

The bundle contains the source, local 126D checkpoint, and reference CSV, but no FSboard NPZ. Full data remains on Drive or Colab scratch. See [the Colab launcher](notebooks/p3_colab_launcher.ipynb) and [Drive storage guide](docs/GOOGLE_DRIVE_STORAGE.md).

Use `scripts/export_npz_subset.py` to create a deterministic 16–32 sample artifact for Windows. Full setup and storage configuration are in [docs/GOOGLE_DRIVE_STORAGE.md](docs/GOOGLE_DRIVE_STORAGE.md).

Run one video through the candidate RGB pipeline. Put the MediaPipe model at `models/hand_landmarker.task`, or set `FSBOARD_HAND_MODEL`. The local 126D checkpoint is the CLI default; override it with `--checkpoint` or `FSBOARD_CHECKPOINT`:

```powershell
python scripts/predict_video.py --video D:\path\to\sample.mp4
```

For a prepared local NumPy input, `scripts/predict_features.py` accepts a `.npy` pose array `[T,21,3]` or a `[T,63]` / `[T,126]` feature array matching the checkpoint. It uses `allow_pickle=False`.

The CLI accepts MP4 and WebM, defaults to 120 seconds and 100 MB, picks the first highest-score hand and locks to its handedness, then prints the text and measured stage timings. It reports a clear error when the local checkpoint, video dependency, codec, or HandLandmarker model is missing. A successful prediction is experimental until coordinate parity is checked on a shared FSboard sample. No confidence or review decision is inferred.

Run the API from the repository root:

```powershell
$env:FSBOARD_CHECKPOINT = "checkpoints/bigru_ctc_motion_best.pt"
$env:FSBOARD_HAND_MODEL = "models/hand_landmarker.task"
uvicorn --app-dir . backend.app.main:app --host 127.0.0.1 --port 8000
```

The API exposes `GET /health`, `GET /model-info`, and `POST /predict` with multipart field `file` and optional `hand`. Uploads are limited to 100 MB, 120 seconds, and about 16.8 megapixels per frame by default, copied to a temporary directory, and removed after inference. It does not retain videos. The API and CLI use the same `VideoInferenceService`.

FastAPI loads the local checkpoint once at startup and does not contact Google Drive per request. Keep the local 126D checkpoint available even when the dataset and evaluation artifacts live in Drive.

### Start the frontend and API together

Use a second PowerShell terminal for the browser app:

```powershell
Set-Location frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open the Vite URL (usually `http://127.0.0.1:5173`). For full setup, response details, timing definitions, privacy behavior, and the real-video checklist, see [docs/E2E_INTEGRATION.md](docs/E2E_INTEGRATION.md) and [docs/E2E_TEST_REPORT.md](docs/E2E_TEST_REPORT.md).

## Tests

```powershell
python -m pytest
```

Current tests cover normalization, motion features, dataset selection, CTC decoding, checkpoint loading, metrics, prediction shapes, bounded video decoding, hand selection, coordinate adaptation, the API upload lifecycle, storage-path resolution, checksum-verified copies, archive extraction, direct prediction-artifact comparison, and subset export using mocks/synthetic fixtures. They do not replace the pending Colab run against the Drive NPZ or a real RGB sample.

Frontend checks run from `frontend/` with `npm test` and `npm run build`. UI tests use a mocked API response; backend integration includes inference with the local checkpoint on synthetic features. These checks do not count as real-video end-to-end acceptance.

## Notebook audit

See [docs/NOTEBOOK_CODE_AUDIT.md](docs/NOTEBOOK_CODE_AUDIT.md) for cell-level component mapping, exact filtering and preprocessing rules, and identified missing artifacts. The notebook is reference material; inference does not require executing notebook cells.

## Known gaps and limitations

- No same-sample comparison between checkpoint predictions and the notebook is possible until `fsboard_test_proc_clean.npz` is supplied.
- The Colab benchmark still needs an authenticated Google account and available Colab runtime; no full dataset is stored in this checkout.
- A real FastAPI/browser video check still needs an actual signing video. The ignored local model asset is available at `models/hand_landmarker.task`.
- Video landmarks are currently adapted by scaling MediaPipe x/y to image pixels and preserving z. The saved notebook sample supports this candidate mixed-scale convention, but matching preprocessing and RGB distribution are unverified.
- Missing hand frames are omitted, with no interpolation. Motion uses adjacent detected poses, as the notebook uses adjacent stored pose rows.
- Webcam/realtime performance, confidence calibration, human correction outcomes, and user-study metrics are unknown.
- The MVP interaction event log is optional, in-memory, and local to the browser; there is no event database or API endpoint. No training has been started.

## Initial model card

- **Intended use:** research baseline for one-hand fingerspelling-to-text and later HCI studies.
- **Inputs:** a sequence of 63D hand features or 126D hand-plus-motion features, prepared using the notebook convention.
- **Output:** a CTC greedy-decoded character sequence.
- **Training/evaluation data:** FSboard; the trained checkpoints and reference outputs came from the supplied Task 3 archive. The processed dataset is not in this repository.
- **Known evaluation:** recomputed reference CSV metrics shown above; no fresh NPZ checkpoint evaluation or RGB-video benchmark.
- **Limitations:** unknown webcam generalization, no calibrated confidence, no streaming, and no verified coordinate parity.
