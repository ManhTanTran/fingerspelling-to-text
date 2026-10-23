# P4 End-to-End Test Report

Date: 2026-10-23  
Repository: `D:\Work\TTNM\fingerspelling-to-text`

## Status

The application integration is implemented, but **P4 is not fully accepted**. No user-provided signing video is present in the repository, so the real-video path through MediaPipe, the browser, and FastAPI has not been exercised. Coordinate compatibility with FSboard training inputs and recognition accuracy remain unverified.

## Evidence

| Check | Result | What it establishes |
|---|---|---|
| Python unit suite | `.\.venv\Scripts\python.exe -m pytest -q` — **55 passed** | Existing and new backend behavior under unit/mock/synthetic fixtures. |
| Real-checkpoint API integration | Included in the 55 passing tests | Loads the local 126D checkpoint, infers a synthetic valid `[T,126]` feature sequence, and checks API prediction equality with `VideoInferenceService`. The video processor is injected; OpenCV and MediaPipe are not run. |
| Local API startup | **200** for `/health` and `/model-info`; `video_inference_ready: true`, `input_dim: 126`, CPU | The local checkpoint, HandLandmarker asset, OpenCV and MediaPipe dependencies were available in `.venv` at verification time. This checks readiness, not video inference. |
| Frontend tests | `npm test -- --reporter=dot` — **2 passed** | A mocked API response reaches the draft UI; edit, explicit confirm, undo, clear, API error and retry controls behave as expected. It does not prove real HTTP/backend inference. |
| Frontend production build | `npm run build` — **passed** | TypeScript and Vite production compilation succeed. |
| Real browser/video acceptance | Pending sample video | No claim of a successful actual video prediction or CLI/API same-video match. |
| FSboard Colab NPZ benchmark | Independent P3 task, pending Colab authentication/run | No fresh FSboard inference metrics are claimed here. Historical CSV-derived metrics remain CSV-derived only. |

## Real-video acceptance checklist

Once a representative MP4 or WebM signing clip is available:

1. Start FastAPI and Vite using [E2E_INTEGRATION.md](E2E_INTEGRATION.md).
2. Verify `GET http://127.0.0.1:8000/health` returns `video_inference_ready: true`.
3. Run `python scripts/predict_video.py --video <sample.mp4>` and save the complete JSON output.
4. Upload that same file in the browser with the same hand selection and coordinate adapter.
5. Compare the CLI and API prediction, selected hand, `(T,126)` feature dimension, hand coverage, model version and settings.
6. Verify the browser displays the exact API prediction, edit the draft, confirm it, then exercise retry, undo and clear.
7. Record browser elapsed time and the backend stage timings. Keep the original clip and any reference transcript separate from the prediction report.
8. Do not change preprocessing if output looks wrong until coordinate and landmark conventions are compared with a matched FSboard source example.

The expected command is runnable from the repository root; replace the placeholder with the supplied local clip path:

```powershell
python scripts/predict_video.py --video D:\path\to\signing-sample.mp4
```

## Remaining limits

- No real video has been submitted; MediaPipe detection, codec support, coordinate adapter behavior and actual decoded text are unobserved.
- The default `pixel_xy_preserve_z` adapter is a candidate. `normalized_xyz` is available for controlled comparison, but neither is established as compatible with the training distribution.
- The real-checkpoint integration fixture verifies API/service parity on the same synthetic feature rows, not CLI/API parity on identical decoded video.
- No accuracy value, real-time claim, or fresh full-FSboard metric is reported by this P4 work.
- Browser interaction events stay local and are excluded unless the user opts in; their export is an event log, not a user-study result.
