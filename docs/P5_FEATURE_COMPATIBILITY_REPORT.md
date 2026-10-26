# P5 — Feature Compatibility Report

Date: 2026-10-25  
Repository: `fingerspelling-to-text`  
Branch: `ui/capture-review-flow`  
Checkpoint: `checkpoints/bigru_ctc_motion_best.pt` (epoch 3; SHA-256 `25c000b5e460444eac6a09fbc545267d395dc7ce83ea25e87de0a72e56faf3fd`)

## Result

**Feature parity remains `FEATURE_PARITY_NOT_VERIFIED`.** The repository's final 126D training transform and runtime transform agree on wrist-relative normalization, scale floor, clipping, flatten order, and adjacent-row motion differences. That is source-code parity for these operations. It is not numerical parity between the actual FSboard training landmarks and MediaPipe landmarks: there is no local processed FSboard NPZ or matched RGB/landmark pair to compare.

The current `pixel_xy_preserve_z` adapter makes X and Y pixel-scaled but leaves MediaPipe Z in its normalized relative-depth scale. This is not a uniform 3D coordinate transform. The saved FSboard notebook sample also has pixel-scale X/Y and small Z, so the adapter is a plausible range-matching candidate; that observation does not verify values, orientation, image scaling, or landmark equivalence.

## Evidence and data availability

| Item | Finding | Evidence level |
|---|---|---|
| Current input contract | 126 float32 values per valid hand frame: 63 pose values followed by 63 motion values | Verified in notebook cell 40 and `src/fingerspelling/features/motion.py` |
| Checkpoint load | Strict-load succeeds; input dim 126, 59 character classes plus blank, blank index 0, epoch 3 | Verified locally with `.venv-p4` |
| Local held-out NPZ | No FSboard NPZ in the checkout or supplied P4 test artifacts. The NPZs found under the checkout are Python package sample/test data | Search result; FSboard NPZ unavailable locally |
| Drive/Colab held-out data | Drive search found no extracted test NPZ or small subset. It found metadata for a 6,361,248,095-byte source ZIP; its media contents were not read. The existing Colab runner downloads that archive when the extracted NPZ is absent | No dataset bytes downloaded; full archive route not run |
| Matched RGB and training landmarks | Not available | Numerical parity test blocked |
| P4.2 user-provided video | 34 consecutive images reconstructed as a 15 FPS MP4; original FPS, signing system, and transcript are unknown | Real video pipeline smoke only; not a labeled recognition sample |

The held-out NPZ is required for independent checkpoint reproduction and a training-feature distribution. Do not run the source-archive Colab route until a small NPZ subset is available or full-archive use is explicitly authorized.

## Training and runtime preprocessing comparison

The comparison below follows the final Day 5/Day 7 126D path in `notebooks/original_task3.ipynb` cell 40, the NPZ loader in `src/fingerspelling/data/fsboard_dataset.py`, and the live video path in `src/fingerspelling/video/video_to_features.py`.

| Property | Final FSboard training path | Current runtime path | Assessment |
|---|---|---|---|
| Landmark shape/order | Selects `entry[0]` with shape `(T,21,3)`; cell 40 flattens each landmark's XYZ consecutively | MediaPipe returns 21 XYZ landmarks; adapter keeps their order; normalization flattens landmark-major XYZ | Shape and flatten operation agree. Actual point-by-point ordering against the FSboard extractor has not been matched |
| X/Y coordinates | Saved notebook sample has pixel-scale X/Y | `pixel_xy_preserve_z` multiplies normalized X by image width and Y by image height | Similar magnitude is plausible, but original source image dimensions and same-frame values are unavailable |
| Z convention | Saved sample has small relative-depth values while X/Y are in the hundreds/thousands | Preserves MediaPipe Z without scaling it by width/height | Mixed scale appears in both artifacts, but this is not a geometrically uniform transform. Compatibility is unverified |
| Wrist translation | Subtract landmark 0 independently in each frame | Same | Code-level match |
| Scale | Euclidean 3D distance from landmark 0 to landmark 9, floored at 1.0 | Same | Code-level match |
| Clip and flatten | Clip normalized pose to `[-5,5]`, then contiguous float32 landmark-major XYZ | Same | Code-level match |
| Motion | First row zeros; later rows are adjacent stored-row differences, clipped to `[-5,5]`; no time division | Same | Code-level match; cadence changes the physical time represented by a row difference |
| Frame cadence | Uses saved pose rows as-is; no resampling or interpolation in the final loader | Decodes every video frame; timestamps are passed to MediaPipe, but motion is still computed per retained row | No cadence normalization. Gaps and FPS differences can change motion magnitude |
| Missing hand | NPZ loader selects the signing-hand pose sequence as saved and does not reconstruct missing rows | Frames with no selected hand are omitted; motion then spans adjacent retained rows | Similar row-compaction behavior, but the exact missing-frame/index provenance cannot be compared |
| Hand selection/mirroring | `signing_hand` chooses right (`0`) or left (`1`); final loader does not mirror | Locks onto one MediaPipe handedness side; adapter does not mirror | Side-selection intent is similar; camera mirroring and label orientation are unverified |
| Sequence lengths | Dataset metadata reports test duration median 7.02 s / mean 7.49 s. One inspected training shard averaged 185.47 frames, range 38–496; a sampled record was about 23.917 FPS | P4.2 reconstruction contains 34 frames at assumed 15 FPS (2.267 s); source FPS is unknown | Not a representative cadence comparison; source video is not confirmed continuous fingerspelling |

### Coordinate adapter interpretation

MediaPipe documents image-space X/Y as normalized by image width/height and Z as a relative depth value whose magnitude is roughly on the scale of normalized X ([Python Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python), [Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)). The current adapter scales X/Y to pixels while leaving Z unchanged. Thus an X displacement of 0.01 becomes 12.8 pixels in a 1280-pixel-wide frame, while Z remains near 0.01. That conversion changes relative axis scale and is not an isotropic 3D transform.

Notebook outputs from one saved FSboard training example report raw X `714.77–1902.65`, Y `1264.86–2693.42`, and Z `-0.2163–0.0323`; its wrist-to-landmark-9 scale ranges from `350.53` to `612.98`. Those values support the observation that the stored representation itself uses large X/Y and small Z. They do not establish that the same physical hand pose would produce matching coordinates through the current adapter. The inspected example and shard are limited notebook evidence, not population statistics.

### Source-code equivalence

The 126D feature functions are preserved. They match the final notebook implementation for normalization, clipping, feature ordering, and motion differences. The remaining unresolved part is the conversion of RGB frames into the training pose representation: coordinate provenance, Z transform, handedness/camera orientation, and sampling cadence. No preprocessing or checkpoint was changed for P5.

## Inference-only feature distribution

The real-video pipeline was run again on the P4.2 reconstructed MP4 to capture its actual feature matrix. This is one unlabeled clip, not a sample of general inference traffic. No training distribution is available for overlay or statistical distance calculations.

Reproduction command from the repository root:

```powershell
.\.venv-p4\Scripts\python.exe -X utf8 test_assets\p5_extract_inference_features.py
```

The script uses the current local MP4, MediaPipe model, and `pixel_xy_preserve_z` adapter; it writes the feature matrix, per-dimension CSV/JSON, and plot beneath `test_assets/p5/`. These are local ignored evidence artifacts.

| Statistic | Pose 63D | Motion 63D | Full 126D |
|---|---:|---:|---:|
| Shape | `(34,63)` | `(34,63)` | `(34,126)` |
| Mean | -0.066334 | -0.004107 | -0.035221 |
| Population standard deviation | 0.559085 | 0.125308 | 0.406334 |
| Minimum | -1.821163 | -0.814108 | -1.821163 |
| 1st percentile | -1.493175 | -0.429922 | -1.265881 |
| Median | -0.000216 | 0.000000 | -0.000050 |
| 99th percentile | 1.333429 | 0.442842 | 1.143410 |
| Maximum | 1.847984 | 1.133498 | 1.847984 |
| Fraction at `±5` clip bound | 0% | 0% | 0% |

The full-vector percentile values are recorded per dimension in the CSV rather than treated as a single semantically uniform coordinate distribution. The motion first row is zero by definition. Hand detection was 34/34 (100%); selected side was right; all 34 decoded frame indices were retained. There were no NaN or infinite values.

![Inference-only pose and motion feature distribution for the 34-frame clip](../test_assets/p5/ben_jarashow_feature_distribution.png)

Per-feature mean, standard deviation, quantiles, and clip-bound fraction: [`ben_jarashow_feature_summary.csv`](../test_assets/p5/ben_jarashow_feature_summary.csv). Machine-readable run metadata: [`ben_jarashow_feature_summary.json`](../test_assets/p5/ben_jarashow_feature_summary.json). The `.npy`, CSV, JSON, and plot are local ignored test artifacts under `test_assets/p5/`; they were not added as tracked binary data.

## Gate status

| Gate | Status | Reason |
|---|---|---|
| Runtime and 126D shape | **PASS** | Existing checkpoint strict-loads; P4.2 real video yields finite `(34,126)` features and executes the model |
| Training/runtime source transform | **PARTIAL — source-code match** | Normalization, motion, order, and clipping agree in source; RGB-to-training landmark parity is unknown |
| Feature compatibility | **FEATURE_PARITY_NOT_VERIFIED** | No matched frame/landmark sequence and no local held-out NPZ |
| Training-vs-inference distributions | **BLOCKED for training side** | Only one inference feature matrix can be measured; no training poses/NPZ available locally |

## P6 priorities

1. Obtain a trusted 16–32-example held-out NPZ subset and rerun the existing checkpoint evaluator, preserving references and row order. Do not download the 6.36 GB archive for this step.
2. Create one golden matched example containing the original RGB frame(s), FSboard pose values, frame indices, selected hand, and corresponding MediaPipe landmarks. Compare raw XYZ, normalized 63D, and all 126D values frame-by-frame before changing the adapter.
3. Record true source FPS and missing-frame indices. Compare time intervals between retained rows; do not infer cadence from the reconstructed 15 FPS test MP4.
4. Collect the labeled video benchmark described in `P5_RECOGNITION_EVALUATION.md`, splitting by signer or recording session before any adapter/decoder experiments.
5. Keep the checkpoint frozen until the matched input representation is understood. Revisit decoding only after feature compatibility is measured.

## Answer to the compatibility question

**Is current input numerically compatible with FSboard training features?** Not established. The normalization and 126D construction match the training code; `pixel_xy_preserve_z` remains a candidate conversion with mixed axis scales and no matched numerical comparison. Keep `FEATURE_PARITY_NOT_VERIFIED`.
