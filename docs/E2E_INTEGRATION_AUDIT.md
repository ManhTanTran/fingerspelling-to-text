# P4 End-to-End Integration Audit

Audit date: 2026-10-23  
Repository: `D:\Work\TTNM\fingerspelling-to-text`

This audit inspects the current checkout before P4 changes. It separates
components that exist from paths that have actually been exercised. In
particular, mocked video tests and an API startup check do not establish real
video-to-text compatibility.

This is the requested pre-implementation audit snapshot. For the implemented
P4 state, verification evidence, and remaining acceptance gate, see
[`E2E_INTEGRATION.md`](E2E_INTEGRATION.md) and
[`E2E_TEST_REPORT.md`](E2E_TEST_REPORT.md).

## Component status

| Component | Current implementation | Integration status | Required action |
|---|---|---|---|
| Video Input | No browser frontend. The CLI accepts one local MP4/WebM; `video_to_features` accepts a path. | CLI input exists; webcam/upload UX is absent. | Add React upload and webcam recording, with browser-supported WebM handling. |
| Video Decoder | `VideoDecoder` streams OpenCV frames in file order and enforces duration, upload-size, and pixel limits. | Unit tests use fake captures. No user video has been decoded. | Preserve limits/order; surface actionable codec/format errors to the UI. |
| Hand Tracking | MediaPipe Tasks `SigningHandTracker` selects a hand, locks side, and exposes detection rate. A local `.task` model exists. | Mock tracking tests pass; the model can be constructed locally, but there is no real clip detection evidence. | Feed the same processor from CLI and API, expose hand coverage, and retain no-hand errors. |
| Feature Extraction | `video_to_features` adapts coordinates, applies notebook normalization, and emits 63D/126D rows. Missing detections are omitted. | Synthetic tests verify shape and sequence behavior. FSboard/RGB coordinate parity is unverified. | Keep the adapter named/configurable and mark compatibility as unverified in results. Do not alter preprocessing without matched evidence. |
| AI Model | `Predictor` loads strict weights-only checkpoints and predicts from feature tensors. Local checkpoint is 126D, hidden size 128, two bidirectional GRU layers, 60 classes, blank 0. | Checkpoint inspection and feature-level tests pass. No real video prediction has been observed. | Keep one loaded predictor per API process and share it with CLI/API video inference. |
| CTC Decoder | Vocabulary and greedy decode are part of `Predictor.predict_features`. | Unit-tested on logits/feature inputs. | Report model-forward and decode timings separately while preserving decoded text behavior. |
| API | FastAPI exposes `/health`, `/model-info`, `/predict`; request files use a temporary directory and are deleted. | Mock upload and validation tests pass; startup was checked with the local checkpoint. `/predict` has not processed real video. | Use shared video service, add optional hand-side selection, CORS, stable response/error data, and timing/coverage fields. |
| Frontend | No `frontend/` directory or React/Vite files exist. | Not implemented. | Build the MVP for webcam recording/upload, status, editable draft, confirmed text, retry/undo/clear/copy. |
| Text Interaction | No browser-side draft/final text workflow or event log exists. | Not implemented. | Keep predicted/edited draft separate from confirmed text; add in-memory, consent-based event export without video or text-content capture. |

## Existing contracts to preserve

- The local FastAPI checkpoint is `checkpoints/bigru_ctc_motion_best.pt` and
  the local MediaPipe asset is `models/hand_landmarker.task`.
- `Predictor.predict_features` accepts a finite `[T, input_dim]` array/tensor;
  the video route must produce `[T, 126]` for this checkpoint.
- `video_to_features` processes frames in decode order, chooses/locks one
  handedness, drops missing-hand frames, and computes existing notebook-style
  normalization and motion features. No interpolation or new scaling should
  be introduced without matched data evidence.
- The current coordinate adapter (`x * width`, `y * height`, unchanged `z`)
  is explicitly a candidate. A browser result must not imply validated accuracy.
- FastAPI loads the checkpoint at application startup, uses CPU when selected
  or when CUDA is unavailable, and must not access Google Drive per request.

## Verified environment and test evidence before P4 changes

- `python scripts/inspect_checkpoint.py --checkpoint checkpoints/bigru_ctc_motion_best.pt`
  reports input dimension 126, hidden size 128, two layers, 60 classes, blank
  index 0, strict state-dict loading, and CPU execution.
- `.venv-p3` has OpenCV 5.0.0 and MediaPipe 1.1.1 installed; the real
  HandLandmarker asset constructs and closes successfully. Uvicorn startup
  loads the checkpoint; `/health` reports ready and `/model-info` reports
  input dimension 126.
- The existing suite previously passed 52 tests. Those tests cover synthetic
  and mocked inputs, not the real-video acceptance path.
- Node.js 22.17.1 and npm 10.9.2 are available. No frontend package currently
  exists.

## P4 integration sequence

1. Create one reusable video inference service used by CLI and FastAPI.
2. Return real prediction, hand coverage, video metadata, coordinate warning,
   and stage timings from the API.
3. Add Vite/React user flow with separate draft and confirmed text states.
4. Add frontend and real-checkpoint API integration tests, plus UI state tests.
5. Run the available suite/build and document what still requires a real video.

The final real-video acceptance test remains pending until a sample video is
provided. P3 FSboard NPZ reproducibility remains an independent research check.
