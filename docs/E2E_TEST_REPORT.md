# P4 End-to-End Verification Report

Date: 2026-10-25

Repository: `D:\Work\Tương tác người máy\fingerspelling-to-text`

Branch / base commit: `ui/capture-review-flow` / `26ef4e652c5146a9e562866f017aa674167fd27d`

## Current acceptance gates

| Gate | Status | Evidence |
|---|---|---|
| **1 — Runtime Ready** | **PASS** | The 126D checkpoint strict-loads, MediaPipe initializes, and FastAPI `/health` and `/model-info` return 200. Earlier automated checks: Python 59 passed, frontend 2 passed, TypeScript and production build passed. |
| **2A — Real Video Pipeline Smoke Test** | **PASS** | The reconstructed 34-frame MP4 ran through the real CLI and FastAPI pipeline. MediaPipe detected a hand in all frames, real 126D features were finite, the checkpoint ran and CTC decoded `jasl`; CLI and API results match. |
| **2B — Browser E2E** | **PASS** | React uploaded the same MP4 to local FastAPI and displayed `jasl`. Retry, edit, Confirm, Copy, Undo and Clear were exercised in the browser. |
| **3 — Feature Compatibility / Recognition Accuracy** | **FEATURE_PARITY_NOT_VERIFIED** | No matched FSboard training/inference sample or ground-truth transcript is available. This smoke-test output is not an accuracy result. |

## P4.2 — ben_jarashow_1759 Real-Frame Smoke Test

This is an engineering smoke test. The 34 images are consecutive frames, but the source FPS and transcript are unknown, and the clip has not been confirmed as continuous ASL fingerspelling. No CER/WER or recognition-quality claim is made.

### Results

| Item | Result |
|---|---|
| Input ZIP | `D:\Downloads\ben_jarashow_1759-20261024T181819Z-1-001.zip`; SHA-256 `CD08740CBB9D694AC68E5348D242000FC1AF891DF135271D2FD2545AC218566A` |
| Extracted frames | 34 readable JPGs, `0001.jpg` through `0034.jpg`, 1280 × 720, 3 channels |
| Reconstructed video | `test_assets/ben_jarashow_1759/ben_jarashow_1759_smoke.mp4`; H.264 stream, OpenCV reports I420 (`yuv420p`) |
| Assumed FPS | 15 FPS for reconstruction only; original FPS is unknown |
| Decoded frame count | 34 frames; OpenCV reports 1280 × 720 at 15 FPS |
| Assumed duration | 2.267 s (= 34 / 15); this is not the original clip duration |
| Hand-detected frames | 34 / 34; selected hand: right; frames discarded: 0 |
| Hand detection coverage | 100% |
| Landmark tensor | `(34, 21, 3)` |
| Valid feature timesteps | 34 |
| Feature shape | `(34, 126)`, `float32`; all finite, 0 NaN and 0 Inf |
| Feature statistics | min `-1.8212`, max `1.8480`, mean `-0.0352`, std `0.4063` |
| CLI prediction | `jasl` |
| API prediction | `jasl` (HTTP 200) |
| CLI/API consistent? | Yes, exact string match |
| Model output | `(34, 1, 60)`; blank argmax on 26/34 steps (76.47%); greedy CTC result `jasl` |
| Model inference latency | CLI 30.573 ms; API 6.527 ms in the saved API run |
| Backend total latency | CLI 2061.140 ms; API 1913.826 ms; browser run 1015 ms backend / 1046 ms round trip |
| Frontend upload tested? | Yes; the real MP4 was uploaded through React and POSTed to the local FastAPI service, HTTP 200 |
| Edit/Confirm tested? | Yes; the displayed draft was edited for the UI flow check to `JASL (manual review demo)` and Confirm updated Final Text |
| Retry tested? | Yes; a second real request returned `jasl` again |
| Undo / Clear / Copy | Yes; Undo reset the confirmed text, Clear removed the clip and reset text, Copy showed `Copied` |
| API validation check | Unsupported `.jpg` upload returned HTTP 415 with `video format must be MP4 or WebM` |
| Temporary upload cleanup | No `fsboard-inference-*` temporary directories remained after requests |
| Gate 2A status | **PASS — real video pipeline smoke test** |
| Gate 2B status | **PASS — browser E2E** |
| Gate 3 status | **FEATURE_PARITY_NOT_VERIFIED** |

### Runtime and model evidence

The selected checkpoint is `checkpoints/bigru_ctc_motion_best.pt`, SHA-256 `25C000B5E460444EAC6A09FBC545267D395DC7CE83EA25E87DE0A72E56FAF3FD`. Strict loading reports input dimension 126, hidden size 128, two bidirectional GRU layers, 60 classes, blank index 0, epoch 3, and CPU execution.

The MediaPipe asset is `models/hand_landmarker.task`, SHA-256 `FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1`. Both artifacts were loaded locally. Inference used the configured `pixel_xy_preserve_z` adapter and existing normalization, motion features, checkpoint and greedy CTC decoder; none were changed for this test.

The API returned `/health` 200 with `model_loaded=true`, `video_inference_ready=true`, adapter `pixel_xy_preserve_z`, and no blockers. `/model-info` returned 200 with `bigru_ctc_126d`, input dimension 126, 60 classes, blank 0 and CPU. The real multipart request decoded all frames, detected the hand, produced 126D features and executed the loaded checkpoint. An unsupported `.jpg` request returned the expected 415.

MediaPipe logged its non-fatal feedback-manager and `NORM_RECT` image-dimensions warnings. It still detected the hand in all 34 frames. The browser console had no errors or warnings during the exercised flow.

### Local evidence and data handling

The extracted frames, reconstructed MP4, JSON evidence, API log and helper scripts are in `test_assets/ben_jarashow_1759/` (with helper scripts directly in `test_assets/`). The folder is excluded from Git using this checkout's `.git/info/exclude`; no binary test data was added to the repository. The API response and CLI/API comparison are in `api-result-and-comparison.json`; tensor/model details are in `pipeline-diagnostics.json`; browser observations are in `browser-e2e-summary.json`; server output is in `api-server.log`.

The dataset was not downloaded. Only the user-provided ZIP was read and its 34 frames were extracted locally.

## Gate 1 — Runtime Ready

### Previously recorded automated checks

| Command | Result | Scope |
|---|---|---|
| `.venv-p4\Scripts\python.exe -m pytest -q` | **59 passed, 0 skipped** | Python suite; prior API parity test used synthetic feature rows and was not considered real-video E2E |
| `npm test -- --reporter=dot` | **2 passed** | Frontend component tests with mocked HTTP responses |
| `npx tsc --noEmit` | **passed** | Type checking |
| `npm run build` | **passed** | Production frontend build |

These checks were recorded during the earlier P4 work. The P4.2 results above are separate direct CLI, HTTP and browser runs with the actual reconstructed video.

### Existing preprocessing compatibility limit

Training preprocessing records 21-landmark poses with pixel-scale X/Y and small Z values, wrist-relative normalization by wrist-to-landmark-9 distance (minimum 1), clipping to `[-5,5]`, landmark-major XYZ flattening, and 63D pose plus clipped adjacent-row motion for 126D input. Inference uses MediaPipe image coordinates, the candidate `pixel_xy_preserve_z` adapter, and those same normalization and motion functions.

There is still no matched FSboard pose/RGB sample to verify landmark order, coordinate and Z conventions, handedness, orientation, missing-frame handling, temporal cadence or numeric feature parity. A valid model execution and returned text do not establish recognition correctness.

## Status and repository changes

The P4.2 change to tracked files is this report. The pre-existing P4 implementation changes remain in the working tree. No checkpoint, preprocessing or architecture was changed, and no commit or push was made.
