# P5 — Recognition Evaluation

Date: 2026-10-25  
Checkpoint: `bigru_ctc_motion_best.pt`, epoch 3, SHA-256 `25c000b5e460444eac6a09fbc545267d395dc7ce83ea25e87de0a72e56faf3fd`  
Checkpoint mode: 126D BiGRU + CTC; hidden size 128; two bidirectional layers; 59 character classes plus CTC blank at index 0.

## Evaluation result

### Held-out checkpoint reproduction

The genuine held-out FSboard NPZ is unavailable in the local checkout and the Drive search did not find an extracted NPZ or small subset. Drive metadata lists a 6,361,248,095-byte source archive, but the archive contents were not downloaded or read. The existing Colab workflow downloads the full source archive when no extracted test NPZ exists, so it was not started.

As a result, **P5 did not produce a new checkpoint inference evaluation**. Strict checkpoint loading, vocabulary/blank metadata, the 126D input, and greedy CTC decoding were verified. Reference alignment against fresh model outputs, and fresh held-out CER/WER/Exact Match, remain blocked on the NPZ.

### Historical FSboard test metrics — CSV consistency only

The supplied historical prediction CSV contains 12,515 `(reference,prediction)` rows. Recomputing metrics from that CSV produced the same values as the saved historical metrics JSON. This validates the CSV metric calculation and is not an independent checkpoint run.

| Metric | Historical value recomputed from CSV | Interpretation |
|---|---:|---|
| CER | 10.1006% | Historical test output; not rerun in P5 |
| WER | 55.7822% | Historical test output; not rerun in P5 |
| Exact match | 34.0392% | Historical test output; not rerun in P5 |
| Character edits | 22,322 / 220,996 reference characters | Historical CSV-derived count |
| Character substitutions / deletions / insertions | 6,196 / 11,575 / 4,551 | Deletions are the largest historical edit category |
| Word edits | 13,429 / 24,074 reference words | Historical CSV-derived count |
| Rows | 12,515 | CSV row count agrees with expected valid-test count |

Sources: `results/reference/baseline_126d_predictions.csv`, `results/reference/baseline_126d_ablation_metrics.json`, and `results/recomputed/baseline_126d_metrics.json`. These files contain no per-row source IDs that could independently prove the row-to-NPZ mapping without the NPZ. P5 did not treat their agreement as an inference reproduction.

### Real-video smoke result — not an accuracy result

The P4.2 sample returned `jasl` from both CLI and API on the same reconstructed 34-frame MP4. The model output was greedily CTC-decoded; blank was the framewise argmax on 26/34 time steps. The video source FPS and transcript are unknown, and the clip is not confirmed to contain continuous fingerspelling. Therefore it has no reference, CER, WER, or Exact Match score and is excluded from the benchmark manifest.

| Observation | Result |
|---|---:|
| Frames / valid timesteps | 34 / 34 |
| Hand detections | 34/34 (100%), right hand selected |
| Feature tensor | finite `float32`, `(34,126)` |
| CLI / API prediction | `jasl` / `jasl` (identical) |
| Model forward latency in saved P4.2 run | CLI 30.573 ms; API 6.527 ms |
| Full request processing | CLI 2,061.140 ms; API 1,913.826 ms |

The feature matrix used for the distribution chart was re-extracted during P5 through the real MediaPipe pipeline. Its detailed statistics are in [`P5_FEATURE_COMPATIBILITY_REPORT.md`](P5_FEATURE_COMPATIBILITY_REPORT.md) and the local `test_assets/p5/` artifacts.

## Small labeled real-video benchmark

No supplied video has a trusted transcript, so there are currently **zero scored real-video examples**. The blank manifest is [`P5_REAL_VIDEO_BENCHMARK_MANIFEST.csv`](P5_REAL_VIDEO_BENCHMARK_MANIFEST.csv); it contains headers only and no fabricated samples.

### Collection instructions

1. Collect 20 clips if possible: 10 development clips and 10 held-out clips. A 10-clip pilot can use 7 development and 3 held-out clips, but report the small sample size prominently. Keep held-out clips untouched while investigating adapters or decoding.
2. Use the same sign system and fingerspelling task as the FSboard model. Record one continuous target phrase per clip, with the exact transcript including spaces and punctuation. Do not derive references from model output.
3. Split by signer or recording session where possible. Avoid nearly identical takes of the same phrase across development and held-out sets. Do not train or fine-tune on these videos.
4. Record the original FPS (from source metadata or camera settings), frame count, resolution, signer pseudonym, session, and whether the camera preview or saved video is mirrored. Do not reconstruct the source FPS from frame count.
5. Keep the hand and fingers visible throughout, use steady lighting and a fixed camera, and save MP4 or WebM. Record consent and remove identifying metadata from filenames where needed.
6. Enter one row per clip in the manifest. Use `split=development` or `split=held_out`; keep the held-out file list separate when tuning anything.

### Reproducible scoring procedure

From the repository root, run the real CLI for every manifest video and save its JSON output separately. Example:

```powershell
.\.venv-p4\Scripts\python.exe scripts\predict_video.py `
  --video "D:\data\clip_001.mp4" `
  --checkpoint checkpoints\bigru_ctc_motion_best.pt `
  --hand-model models\hand_landmarker.task
```

For every result, record the exact `reference`, raw `prediction`, `hand_detection_rate`, `landmark_frames` (valid timesteps), `input_fps`, and `stage_timings_ms.model_inference` or the selected end-to-end latency measure. Do not substitute the UI-edited/final text for the raw prediction.

Create one `reference,prediction` CSV for development and another for held-out clips. Calculate corpus metrics separately for each split:

```powershell
.\.venv-p4\Scripts\python.exe scripts\evaluate_predictions.py `
  --csv results\p5\development_predictions.csv `
  --output results\p5\development_metrics.json

.\.venv-p4\Scripts\python.exe scripts\evaluate_predictions.py `
  --csv results\p5\held_out_predictions.csv `
  --output results\p5\held_out_metrics.json
```

Retain all per-video outputs and manifest rows so low coverage, temporal outliers, and character errors can be inspected. Do not pool development and held-out references into one score.

## Failure analysis

No labeled video is available for cross-example failure attribution. The one smoke clip has perfect hand coverage but an unknown reference, so neither detection success nor the `jasl` prediction establishes recognition quality. Blank-dominant framewise logits are expected in CTC and do not by themselves indicate a decoder fault.

Historical FSboard test outputs have more deletions than substitutions or insertions (11,575 vs. 6,196 and 4,551). That is a pattern in the prior evaluation, not proof that the runtime adapter, cadence, model, or decoder caused those errors. No causal ranking can be made from the one unlabeled video.

| Candidate source | Evidence now | Conclusion |
|---|---|---|
| Hand detection/tracking | 34/34 detections on one clip | Working on this clip; no broad robustness claim |
| Coordinate conversion | Pixel X/Y plus unscaled relative Z; no matched sample | Plausible compatibility risk; parity unverified |
| Normalization and motion | Exact formula agreement in final notebook and runtime code | No source-code discrepancy found in these operations |
| Temporal mismatch | Training records use saved rows; runtime uses retained video frames and adjacent-row differences. Source FPS is unknown for the sample | Material risk to validate with true FPS and matched frame indices |
| CTC decoding | Greedy CTC behavior matches the notebook: remove blank and collapse adjacent repeated tokens | No evidence it is the primary error source |
| Model generalization/domain shift | Historical held-out metrics show substantial WER; real clips lack labels | Performance on newly collected videos remains unknown |

## Prioritized P6 recommendations

1. **First: reproduce the existing checkpoint on a trusted 16–32 example held-out NPZ subset.** Confirm exact row/reference alignment and compare fresh predictions to the historical CSV. Do not use the full source archive for this gate.
2. **Then: verify input numerically.** Compare a matched RGB frame sequence and FSboard pose sequence at raw XYZ, normalized 63D, and 126D motion stages; record mirroring, chosen side, frame indices, and timestamps.
3. **Collect labeled development and held-out videos** using the manifest; report per-video detection, lengths, latency, CER, and Exact Match, and held-out corpus CER/WER/Exact Match.
4. **Only after feature compatibility is measured**, evaluate a justified coordinate/cadence correction on development data, leaving the checkpoint and held-out split frozen.
5. **Assess decoding/model changes last.** Compare decoder alternatives only on identical feature tensors and use held-out data once for final measurement.

Do not fine-tune yet. Validate and correct any demonstrated preprocessing mismatch first; consider fine-tuning only if compatible features still show a measured generalization gap on development data.

## Answers requested

1. **Is the current input numerically compatible with FSboard training features?** Not established. Core normalization/motion source code matches; raw MediaPipe-to-FSboard numeric parity does not.
2. **How well does it recognize genuine fingerspelling videos?** The historical FSboard test CSV reports CER 10.10%, WER 55.78%, and Exact Match 34.04%, but P5 could not independently rerun it. The P4.2 real-video clip has no transcript, so current real-world video accuracy is unknown.
3. **Are errors primarily preprocessing, temporal shift, model limitations, or decoding?** Evidence is insufficient to attribute primary cause. Coordinate scaling and cadence are risks; no matched data or labeled video comparison isolates them.
4. **Is fine-tuning necessary, or should preprocessing be corrected first?** Correct only a demonstrated preprocessing mismatch first. Fine-tuning is not justified until parity and labeled development-video performance are measured.
