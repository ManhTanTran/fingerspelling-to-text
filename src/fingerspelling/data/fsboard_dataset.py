"""FSboard processed-NPZ loading and variable-length CTC batching.

The notebook's processed NPZ files contain object arrays and therefore use
NumPy pickle payloads. Loading is refused unless the caller explicitly marks a
local file as trusted. Never pass a browser/API upload to this loader.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from fingerspelling.data.vocabulary import Vocabulary
from fingerspelling.features.motion import add_motion_features
from fingerspelling.features.normalization import normalize_hand


@dataclass(frozen=True)
class FSboardExample:
    source_index: int
    pose: NDArray[np.float32]
    label: str
    signing_hand: int


def text_value(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def minimum_ctc_frames(text: str) -> int:
    repeated = sum(text[index] == text[index - 1] for index in range(1, len(text)))
    return len(text) + repeated


def load_processed_npz(
    path: str | Path,
    *,
    trusted_local: bool = False,
    filter_ctc_impossible: bool = True,
) -> tuple[list[FSboardExample], dict[str, int]]:
    """Load one notebook-format split from a trusted local processed NPZ.

    Selects right-hand poses when ``signing_hand == 0`` and left-hand poses
    when it is 1. It reads each saved ``entry[0]`` pose sequence as-is; it does
    not interpolate, synthesize absent frames, or consult frame-index arrays.
    Training/validation should use the default CTC-length filter. The notebook's
    test path did not apply this filter, so set it false for test inference.
    """
    source = Path(path)
    if not trusted_local:
        raise ValueError(
            "processed NPZ uses pickle-backed object arrays; pass trusted_local=True "
            "only for a trusted local dataset file"
        )
    if not source.is_file():
        raise FileNotFoundError(f"processed FSboard NPZ not found: {source}")

    stats = Counter()
    examples: list[FSboardExample] = []
    with np.load(source, allow_pickle=True) as data:
        required = {"left_hand_poses", "right_hand_poses", "signing_hand", "words"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}")

        left = data["left_hand_poses"]
        right = data["right_hand_poses"]
        hands = data["signing_hand"]
        labels = data["words"]
        n = len(labels)
        if any(len(array) != n for array in (left, right, hands)):
            raise ValueError("pose, signing_hand, and words arrays have different lengths")
        stats["total"] = n

        for index in range(n):
            try:
                hand = int(hands[index])
            except (TypeError, ValueError, OverflowError):
                stats["missing_hand"] += 1
                continue
            if hand not in (0, 1):
                stats["missing_hand"] += 1
                continue

            entry = right[index] if hand == 0 else left[index]
            try:
                pose = np.asarray(entry[0], dtype=np.float32)
            except (KeyError, IndexError, TypeError, ValueError):
                stats["empty_pose"] += 1
                continue
            if pose.size == 0:
                stats["empty_pose"] += 1
                continue
            if pose.ndim != 3 or pose.shape[1:] != (21, 3) or not np.isfinite(pose).all():
                stats["invalid_pose"] += 1
                continue

            label = text_value(labels[index])
            if not label.strip():
                stats["empty_label"] += 1
                continue
            if filter_ctc_impossible and len(pose) < minimum_ctc_frames(label):
                stats["too_short_for_ctc"] += 1
                continue

            examples.append(FSboardExample(index, pose, label, hand))
            stats["kept"] += 1

    for key in (
        "missing_hand", "empty_pose", "invalid_pose", "empty_label", "too_short_for_ctc"
    ):
        stats.setdefault(key, 0)
    return examples, dict(stats)


class FSboardDataset(Dataset):
    def __init__(
        self,
        examples: list[FSboardExample],
        vocabulary: Vocabulary,
        feature_mode: Literal["hand", "motion"] = "hand",
    ) -> None:
        if feature_mode not in ("hand", "motion"):
            raise ValueError("feature_mode must be 'hand' or 'motion'")
        self.examples = examples
        self.vocabulary = vocabulary
        self.feature_mode = feature_mode

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        example = self.examples[index]
        features = (
            normalize_hand(example.pose)
            if self.feature_mode == "hand"
            else add_motion_features(example.pose)
        )
        try:
            target = [self.vocabulary.char_to_idx[char] for char in example.label]
        except KeyError as exc:
            raise ValueError(
                f"label contains character outside vocabulary: {exc.args[0]!r}"
            ) from exc
        return (
            torch.from_numpy(features),
            torch.tensor(target, dtype=torch.long),
            example.label,
        )


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor, str]]):
    if not batch:
        raise ValueError("cannot collate an empty batch")
    features, targets, texts = zip(*batch)
    lengths = torch.tensor([feature.shape[0] for feature in features], dtype=torch.long)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    padded = torch.zeros(
        len(features), int(lengths.max()), features[0].shape[1], dtype=torch.float32
    )
    for index, feature in enumerate(features):
        if feature.ndim != 2 or feature.shape[1] != padded.shape[2]:
            raise ValueError("all features must have the same [time, feature_dim] shape")
        padded[index, :len(feature)] = feature
    return padded, torch.cat(targets), lengths, target_lengths, list(texts)

