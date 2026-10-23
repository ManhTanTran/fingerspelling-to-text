# Task 3 notebook and artifact audit

## Inputs inspected

The notebook was read statically from all 45 cells and copied byte-for-byte to `notebooks/original_task3.ipynb`; no notebook cell was executed. `D:\Downloads\TASK3.zip` was inventoried, then a whitelist of checkpoints and reference outputs was imported into the repository. The ZIP does not contain the processed NPZ files.

The notebook's cells 17 and 19 check/download the separate `fsboard_proc_clean.zip` using Drive file ID `1dhcrbZeb3-5IshIkzatVmNO3MOTVFjJW`. Its saved archive inventory is 6.79 GiB of uncompressed members, including `fsboard_test_proc_clean.npz` (560.5 MiB), `fsboard_validation_proc_clean.npz` (489.2 MiB), and `fsboard_train_proc_clean.npz` (5905.4 MiB). Drive metadata reports the ZIP itself as 6,361,248,095 bytes. Actual media access remains unverified until the authenticated Colab range probe succeeds. The outputs in those cells are historical notebook outputs, not files currently in the workspace.

## Component audit

| Component | Cell source | Input | Output | Dependency | Status |
|---|---|---|---|---|---|
| Dataset loading | 33, 37 | NPZ arrays `left_hand_poses`, `right_hand_poses`, `signing_hand`, `words` | Valid `(pose, label)` examples | Trusted processed NPZ | Implemented in `data/fsboard_dataset.py`; not run on real NPZ yet |
| Metadata / labels | 33 | `words` and checkpoint mapping | Character targets | NPZ, train-derived vocabulary | Mapping recovered from checkpoints |
| Signing-hand selection | 33, 37 | `signing_hand` | `0` selects right; `1` selects left; pose is `entry[0]` | Processed NPZ | Implemented exactly |
| Coordinate normalization | 33, 37 | `(T, 21, 3)` pose | Wrist-relative normalized pose | Stored FSboard pose convention | Implemented: 3D wrist-to-landmark-9 norm, minimum scale 1.0, clip `[-5, 5]` |
| Hand-only features | 33, 37 | Normalized pose | `(T, 63)` | Normalization | Implemented and unit-tested |
| Motion features | 40, 42 | Normalized `(T, 63)` pose | `(T, 126)` | Normalization | Implemented: zero first velocity; next rows are adjacent differences, clip `[-5, 5]`, concatenate pose then velocity |
| Vocabulary / blank | 33; checkpoint metadata | Training character set | 59 tokens, 60 classes | Checkpoint | Recovered exactly; blank index `0`; chars indexed `1–59` |
| PyTorch Dataset / collate | 33, 40 | Variable-length poses and targets | Zero-padded `[B,T,D]`, concatenated targets and CPU lengths | PyTorch | Implemented |
| BiGRU architecture | 33, 37, 40 | 63D or 126D features | CTC logits `[T,B,60]` | PyTorch | Implemented with 2-layer bidirectional GRU, hidden 128, GRU dropout 0.2, output dropout 0.2; both real state dicts load strictly |
| CTC loss | 33, 36, 40 | `log_softmax(logits)`, targets, lengths | CTC loss | PyTorch | `blank=0`, `zero_infinity=True`; no training run performed here |
| Greedy decoding | 33, 37, 40 | Frame argmax path | Text | Vocabulary, blank index | Generic decoder implemented; CSV metrics independently match references |
| Training | 34–36, 40 | Official train/validation split | Checkpoints | NPZ, GPU in original run | Audited, not rerun; 63D baseline fine-tuned from quick checkpoint for 3 full-data epochs; 126D ablation uses 6400-sample warm-up then 3 full-data epochs |
| Evaluation | 37, 40, 42 | Test NPZ and predictions | CER/WER/exact match, error breakdown | Checkpoint, test NPZ | Metrics recomputed from supplied CSVs; inference from features pending NPZ |
| Checkpoint load/save | 34–37, 40 | State dict plus vocabulary and metadata | Restored model | `.pt` files | Both supplied checkpoints load with `weights_only=True` and strict state-dict checks |
| Error analysis | 42–43 | Aligned 63D/126D prediction CSVs | S/D/I, group summaries, sample errors | Reference predictions | Imported; corpus metrics match recomputation |

## Exact preprocessing behavior

- The notebook uses the saved pose arrays as sequences. It does not recreate the original video frame grid, fill missing detections, or interpolate.
- For train and validation, it skips missing/invalid signing-hand metadata, unreadable or empty pose, wrong shape, non-finite values, empty labels, and examples shorter than `len(label) + adjacent_repeat_count` CTC frames. Validation examples containing out-of-vocabulary characters are then dropped.
- Test inference selects the same signing hand and skips missing/invalid pose and empty labels. It records unknown characters but does not apply the CTC minimum-frame filter.
- `right_frame_indices` and `left_frame_indices` are present in the NPZ but are not read by the model Dataset. Motion differences are computed over consecutive stored pose rows; they are not divided by elapsed time or actual frame-index gaps.
- Normalization is `(pose - wrist) / max(norm_3d(landmark_9 - wrist), 1.0)`, followed by clipping to `[-5, 5]`. Features are flattened in landmark-major XYZ order.
- Motion is `velocity[0] = 0`; for `t > 0`, `velocity[t] = normalized[t] - normalized[t-1]`, clipped to `[-5, 5]`; 126D is `[pose63, velocity63]`.
- Notebook cell 25's saved pose sample has x/y in pixel-scale ranges (x 714.8–1902.6, y 1264.9–2693.4) while z is small (-0.216 to 0.032). MediaPipe's official Python guide describes x/y as normalized and z as relative depth with roughly x's scale. The implemented candidate adapter multiplies x/y by image width/height and keeps z unscaled, which matches the observed FSboard mixed-scale sample. It is not a verified coordinate transform; exact parity requires the processed test NPZ or a shared RGB/landmark sample.

## Metric reproduction

Metrics were computed from the imported prediction CSVs, not copied from the notebook's saved outputs:

| Model | Samples | CER | WER | Exact match |
|---|---:|---:|---:|---:|
| 63D hand-only | 12,515 | 10.4083% | 57.1322% | 33.0963% |
| 126D hand + motion | 12,515 | 10.1006% | 55.7822% | 34.0392% |

The recomputed values match the updated metrics in `TASK3.zip`. Saved outputs embedded in the notebook's cell 37/40/42 are from an earlier run and differ; they are not treated as current metrics.

## RGB inference and API added

- `video/decoder.py` streams MP4/WebM frames in file order, validates FPS and dimensions, and enforces duration, upload-byte, and pixel-count limits.
- `video/hand_tracker.py` uses MediaPipe Tasks VIDEO mode, reports a detection rate, supports a forced left/right side, and otherwise locks to the first highest-score hand rather than switching sides.
- `video/video_to_features.py` omits frames without usable landmarks, stores their decoded frame indices for diagnostics, and computes 63D or 126D features with the audited functions. No interpolation is applied.
- `scripts/predict_video.py` reports decode, tracking, coordinate adaptation, feature, model, and load latency. Its RGB prediction output marks coordinate compatibility unverified.
- `backend/app/main.py` provides `/health`, `/model-info`, and `/predict`. It bounds uploads, uses a temporary file, and removes it after inference. The API does not report a confidence or review decision.
- Mock-based tests pass for decoder order and error cases, side locking, feature shapes, and upload deletion. The official `models/hand_landmarker.task` asset is prepared locally. A clean Windows `.venv-p3` with OpenCV and MediaPipe now starts the real 126D checkpoint behind FastAPI; `/health` reports video inference ready and `/model-info` reports input_dim 126. Real video inference and coordinate parity remain unverified because no sample video has been supplied.

## Still required for checkpoint inference reproduction

The test member `fsboard_test_proc_clean.npz` is expected in the Drive archive referenced by the original notebook ID `1dhcrbZeb3-5IshIkzatVmNO3MOTVFjJW`. Drive metadata reports the ZIP as 6,361,248,095 bytes (about 5.93 GiB); the Colab runner checks authenticated content access before downloading it to Colab scratch and extracting only the test member. Training reproduction additionally needs the train and validation NPZ members. These NPZ files use pickle-backed object arrays and must remain trusted artifacts; never pass an API upload to `np.load(..., allow_pickle=True)`.

For actual RGB-to-text execution also install the `video` extras. A compatible MediaPipe model is prepared locally at `models/hand_landmarker.task`; however, no real signing video has been supplied, so a CLI/API result and RGB coordinate parity remain unverified. Do not treat a mock test as evidence of RGB benchmark parity.

