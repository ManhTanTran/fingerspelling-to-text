# Day 2 — Dataset Audit and Frozen Split

**Project:** One-Hand Alphabet and Continuous Fingerspelling Recognition for Text Entry  
**Task:** Task 3 — Fingerspelling video to text  
**Dataset:** FSboard (`dmk_v3` and `daun_v3`)  
**Status:** Metadata audit completed; landmark audit sampled on one shard; official split retained.

## 1. Purpose

Before training a recognizer, inspect the dataset metadata, label distribution, signer/phrase overlap, clip duration, and landmark format. The goal is to document data issues and settle a reproducible train/validation/test split.

## 2. Dataset overview

| Split/source | Samples | Signers | Unique phrases |
|---|---:|---:|---:|
| `daun_v3` train | 70,699 | 94 | Not reported separately in the final combined audit |
| `dmk_v3` train | 55,504 | 72 | 508 |
| **Combined train** | **126,203** | **117** | **48,711** |
| `daun_v3` validation | 12,022 | 15 | 9,640 |
| `daun_v3` test | 12,529 | 15 | 9,786 |
| **Total** | **150,754** | — | — |

The split-level signer counts are not additive: some signers occur in both training sources, while the combined train contains 117 unique signers.

### Metadata checks on `dmk_v3-train.json`

The file contains 55,504 records, 72 unique signers, 508 unique phrases, and 55,504 unique clip filenames. The following fields had no missing or empty values in this file: `clipFilename`, `phrase`, `signerId`, `clipStartTimeS`, and `clipEndTimeS`. These missing-field results apply only to this metadata file, not to every split.

All 55,504 records had `signingType = fs`. Phrase length ranged from 16 to 43 characters, with a mean of 28.65 characters. This character count includes spaces.

## 3. Label and character distribution

The combined training metadata contains **59 distinct characters**:

- lowercase letters `a`–`z`;
- digits `0`–`9`;
- a space;
- punctuation/symbols: `! # $ % & ' ( ) * + , - . / : ; = ? @ [ _ ~`.

No uppercase letters were observed in the training phrases. Validation and test contain **no character types absent from training** according to the character-set comparison performed.

The character set differs from the 74-class figure mentioned in earlier planning material. For this experiment, the observed metadata character set should be used to define labels unless a separate, verified model label map specifies otherwise. A CTC model will also require its special `blank` token in addition to the target characters.

The count of 59 is the number of distinct character types, not the total number of character occurrences. Per-character frequency counts should be exported from the combined training rows when generating `character_distribution.csv`.

## 4. Split integrity and overlap

The existing published split is retained rather than creating a new random split.

| Pair | Signer overlap | Phrase overlap |
|---|---:|---:|
| Train – validation | 0 | 0 |
| Train – test | 0 | 0 |
| Validation – test | 0 | 877 |

Phrase comparison was performed on the phrase strings (with lowercase and whitespace normalization in the initial comparison). Thus, no phrase overlap was found between train and either evaluation split, but **877 unique phrases occur in both validation and test**. This is recorded as a limitation; the split should not be described as completely phrase-disjoint across all three subsets.

### Frozen-split decision

- **Train:** combine `dmk_v3-train.json` and `daun_v3-train.json` (126,203 rows).
- **Validation:** keep `daun_v3-validation.json` (12,022 rows).
- **Test:** keep `daun_v3-test.json` (12,529 rows).
- Do not move validation or test samples into training, and do not reshuffle the official split for the primary experiment.
- Record the validation–test phrase overlap when reporting test results. Use validation for checkpoint selection and reserve test for final evaluation.

This decision prioritizes reproducibility and preserves the supplied split. The observed validation–test overlap is a known limitation, not something silently corrected by changing the split.

## 5. Clip duration

Durations were calculated as `clipEndTimeS - clipStartTimeS` from the metadata.

| Split | Minimum (s) | Median (s) | Mean (s) | Maximum (s) | Duration ≤ 0 |
|---|---:|---:|---:|---:|---:|
| Train | 0.99 | 7.00 | 7.47 | 38.97 | 0 |
| Validation | 1.00 | 6.02 | 6.79 | 30.11 | 0 |
| Test | 1.00 | 7.02 | 7.49 | 24.98 | 0 |

No non-positive clip durations were found in the three metadata splits. These are durations from metadata timestamps, not frame counts.

## 6. Landmark schema and missing-value sample audit

A single training landmark shard was inspected: `daun_v3/landmarks/daun_v3-train.arrow-00000-of-00100`.

| Inspection | Result |
|---|---:|
| Videos/records in shard | 739 |
| Total frames, from `num_frames` | 137,063 |
| Mean frames per video | 185.47 |
| Minimum frames per video | 38 |
| Maximum frames per video | 496 |
| Right-hand frames with empty `x_right_hand` | 42,777 (31.21%) |
| Videos with at least one missing right-hand X frame | 675 / 739 |
| Left-hand frames with empty `x_left_hand` | 109,594 (79.96%) |
| Videos with at least one missing left-hand X frame | 722 / 739 |

The Arrow file has one `serialized` binary field; records decode as TensorFlow `SequenceExample`. A sample record had `num_frames = 159`, frame rate approximately 23.917 FPS, and 159 steps in its feature lists. In the first inspected record, each nonempty hand coordinate list contained 21 values per frame. Pose features contained 25 values per frame in that inspected sample.

Empty hand-coordinate lists do **not automatically mean corrupt data**: a hand may be outside the frame or not detected. The above rates describe only this one shard and specifically count empty X-coordinate lists; they must not be reported as full-dataset missing-landmark rates. The observed 25-value pose representation also needs to be reconciled with the preprocessing assumptions before model input is finalized.

## 7. Decisions before training

1. Preserve the supplied train/validation/test split as described above.
2. Use the observed 59-character vocabulary as the starting point for the label map; verify the model's label-map file before training and add the CTC blank class separately.
3. Explicitly handle frames with absent hand landmarks. Do not silently treat empty lists as valid all-zero hand coordinates without defining and testing that policy.
4. Validate feature dimensions before training. Twenty-one 3D hand landmarks give 63 coordinate values per frame; a planned 64-dimensional feature vector must define the extra feature (for example, an explicit detection flag) rather than assume it exists.
5. Keep validation for model/checkpoint selection and test for final reporting. Mention the 877 validation–test phrase overlap as a limitation.
6. Expand landmark quality checks beyond the one inspected shard before making claims about the entire dataset, if time and compute permit.

## 8. Limitations of this audit

- Missing metadata fields were checked on `dmk_v3-train.json`; the same field-by-field check was not separately reported for every metadata file.
- Landmark missing-value statistics cover one shard (739 videos), not all FSboard landmark files.
- The audit did not train or evaluate a model. No CER, WER, exact-match score, or model checkpoint is produced by this stage.
- The exact character-frequency table has not yet been exported as a file.

## 9. Day 2 outcome

The metadata split counts, signer/phrase overlap, observed character vocabulary, clip durations, and sample landmark schema have been inspected. The supplied split is retained for a reproducible baseline, with validation–test phrase overlap explicitly documented. The next step is baseline setup and a small end-to-end test on real data, after deciding how to decode and handle missing landmark values.
