# P4 Application Integration

## Architecture

The browser records or uploads an MP4/WebM clip, posts it to FastAPI, and shows the returned model text as an editable draft. The user must confirm a draft before it is appended to the separate final-text buffer.

```text
React / TypeScript
  └─ POST /predict (multipart video, optional hand)
       └─ FastAPI temporary upload
            └─ VideoInferenceService (shared by API and CLI)
                 ├─ OpenCV decode in frame order
                 ├─ MediaPipe Tasks HandLandmarker
                 ├─ configured MediaPipe coordinate adapter
                 ├─ existing FSboard hand normalization + motion features
                 ├─ local BiGRU + CTC checkpoint
                 └─ greedy CTC text decode
```

FastAPI loads the trusted checkpoint once at startup and uses it for all requests. Uploaded files are bounded and placed in a temporary directory removed after processing. The request path has no Google Drive dependency. The current response marks coordinate compatibility as unverified and leaves `needs_review` null.

## Windows development setup

Use two PowerShell terminals from the repository root. This Windows checkout has a Vietnamese path that MediaPipe's native loader cannot open reliably. Use an ASCII junction once, then launch Python from that alias so the settings loader preserves the ASCII asset path:

```powershell
$alias = "D:\Work\fsboard-p4"
if (-not (Test-Path -LiteralPath $alias)) {
    New-Item -ItemType Junction -Path $alias -Target "D:\Work\Tương tác người máy\fingerspelling-to-text"
}
Set-Location $alias
```

### Terminal 1: API

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install ".[video,api,dev]"
uvicorn --app-dir . backend.app.main:app --host 127.0.0.1 --port 8000
```

Keep the trusted 126D checkpoint at `checkpoints/bigru_ctc_motion_best.pt` and the MediaPipe asset at `models/hand_landmarker.task`, or point `FSBOARD_CHECKPOINT` and `FSBOARD_HAND_MODEL` at local copies. A regular install is used here because setuptools' editable install can fail to encode a Windows checkout path containing Vietnamese combining marks. Run the API from the ASCII junction so its YAML-relative paths and MediaPipe's native file access stay on the junction path. The model asset can be obtained from the official [MediaPipe Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python). Visit `http://127.0.0.1:8000/health`; `video_inference_ready` should be `true` before submitting a clip.

### Terminal 2: browser app

```powershell
Set-Location frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open the local URL printed by Vite (normally `http://127.0.0.1:5173`). `.env.local` can set `VITE_API_BASE_URL` if FastAPI is listening elsewhere. Development CORS allows localhost and 127.0.0.1 on port 5173 by default; `FSBOARD_CORS_ORIGINS` overrides the list.

The browser accepts MP4/WebM uploads. Webcam capture uses the first MP4/WebM format the browser supports and reports an actionable message when it cannot create one. The default request caps are 100 MB, 120 seconds, and 16,777,216 pixels per frame.

## Inference configuration precedence

Both the CLI and FastAPI read `configs/inference.yaml`. Relative checkpoint and model paths in the YAML and `FSBOARD_*` variables resolve from the repository root. Select another YAML file with `FSBOARD_INFERENCE_CONFIG`; per-setting environment overrides are `FSBOARD_CHECKPOINT`, `FSBOARD_HAND_MODEL`, `FSBOARD_DEVICE`, `FSBOARD_MAX_DURATION_SECONDS`, `FSBOARD_MAX_UPLOAD_BYTES`, `FSBOARD_MAX_PIXELS`, and `FSBOARD_COORDINATE_ADAPTER`.

Settings resolve in this order, from lowest to highest priority:

1. Built-in defaults, overridden by `configs/inference.yaml` (or the file selected by `FSBOARD_INFERENCE_CONFIG`).
2. The corresponding `FSBOARD_*` environment variable.
3. An explicit CLI option such as `--checkpoint`, or an explicit `create_app(...)` argument for embedded FastAPI use.

The configured adapter must be one of `pixel_xy_preserve_z` or `normalized_xyz`; the old YAML alias `mediapipe_xy_to_pixels_preserve_z_candidate` was not accepted by the implementation and has been corrected. The default remains a candidate pending feature-parity evidence.

## CLI and HTTP interfaces

From the repository root, the trusted local checkpoint is the default for the CLI:

```powershell
python scripts/predict_video.py --video path\to\sample.mp4
```

Optional settings include `--checkpoint`, `--hand-model`, `--hand left|right`, `--device cpu|auto`, `--coordinate-adapter pixel_xy_preserve_z|normalized_xyz`, `--max-duration-seconds`, `--max-upload-bytes`, and `--max-pixels`. The CLI prints JSON with the prediction, video metadata, coverage, model/device and stage timings. CLI and API call the same `VideoInferenceService`.

The API endpoints are `GET /health`, `GET /model-info`, and `POST /predict`. `/predict` expects multipart `file` and an optional `hand` value of `left` or `right`. Successful responses include `prediction`, `needs_review: null`, `processing_time_ms`, `model_forward`, `ctc_decode`, decode/tracking/feature stage timings, input metadata and a coordinate-compatibility warning. Error responses use `{ "status": "error", "message": "..." }`.

`processing_time_ms` is backend processing from the start of file-based video processing through decoded output; it excludes upload transfer and checkpoint startup. The UI separately shows browser elapsed time around the upload request and response. Stage timing fields distinguish model forward and CTC decoding from the wider backend total. These figures are per-run observations, not a real-time performance claim.

## Interaction and privacy

- Model prediction and user-approved final text are separate states. Confirmation is explicit.
- The optional interaction log is off by default, held in browser memory, and downloadable only after consent. It contains event names, timestamps and character counts, never video or text content.
- FastAPI deletes the temporary uploaded file after inference. There is no video archive or event database in this MVP.
- The model output is experimental. Review it before use; coordinate parity with the FSboard training representation has not been established.

## Verification commands

```powershell
python -m pytest
Set-Location frontend
npm test
npm run build
```

The automated UI tests use a mocked HTTP response to verify display, edit, explicit confirmation, undo, clear, API error and retry behavior. Backend tests also run the real local checkpoint on a synthetic valid 126D feature sequence and compare API output with the shared service. Neither test is evidence of MediaPipe performance or recognition accuracy on real video. See [E2E_TEST_REPORT.md](E2E_TEST_REPORT.md) for observed results and the remaining acceptance step.
