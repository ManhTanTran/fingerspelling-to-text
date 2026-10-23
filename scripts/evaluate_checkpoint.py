"""Run notebook-compatible checkpoint inference on a trusted local test NPZ."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

from fingerspelling.data.fsboard_dataset import FSboardExample, load_processed_npz
from fingerspelling.evaluation.metrics import corpus_error_metrics, edit_counts
from fingerspelling.features.motion import add_motion_features
from fingerspelling.features.normalization import normalize_hand
from fingerspelling.inference import load_checkpoint
from fingerspelling.storage import load_storage_config
from fingerspelling.storage.integrity import sha256_file


class _InferenceDataset(Dataset):
    def __init__(self, examples: list[FSboardExample], feature_mode: str) -> None:
        self.examples = examples
        self.feature_mode = feature_mode

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        features = (
            normalize_hand(example.pose)
            if self.feature_mode == "hand"
            else add_motion_features(example.pose)
        )
        return torch.from_numpy(features), example


def _collate_inference(batch):
    features, examples = zip(*batch)
    lengths = torch.tensor([len(features_i) for features_i in features], dtype=torch.long)
    padded = torch.zeros(len(features), int(lengths.max()), features[0].shape[1])
    for index, feature in enumerate(features):
        padded[index, :len(feature)] = feature
    return padded, lengths, list(examples)


def evaluate_checkpoint(
    checkpoint_path: str | Path | None = None,
    test_npz: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    batch_size: int = 16,
    device: str = "auto",
    trusted_local_npz: bool = False,
    storage_config_path: str | Path | None = None,
    max_samples: int | None = None,
    test_npz_sha256: str | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    if not trusted_local_npz:
        raise ValueError("set trusted_local_npz=True only for a trusted local artifact")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when provided")
    run_started = time.perf_counter()
    storage = load_storage_config(storage_config_path)
    resolved_checkpoint = storage.resolve_checkpoint(checkpoint_path)
    resolved_test_npz = storage.resolve_test_npz(test_npz)
    resolved_output_dir = storage.resolve_output_dir(output_dir, kind="evaluation")

    checkpoint_started = time.perf_counter()
    loaded = load_checkpoint(resolved_checkpoint, device=device)
    checkpoint_load_seconds = time.perf_counter() - checkpoint_started
    if loaded.input_dim == 63:
        feature_mode = "hand"
    elif loaded.input_dim == 126:
        feature_mode = "motion"
    else:
        raise ValueError(f"unsupported checkpoint input dimension: {loaded.input_dim}")

    data_started = time.perf_counter()
    examples, test_stats = load_processed_npz(
        resolved_test_npz,
        trusted_local=trusted_local_npz,
        filter_ctc_impossible=False,
    )
    valid_test_count = len(examples)
    if max_samples is not None:
        examples = examples[:max_samples]
        test_stats = {**test_stats, "evaluated_subset": len(examples)}
    data_load_seconds = time.perf_counter() - data_started
    if not examples:
        raise ValueError("test NPZ has no usable examples after notebook-compatible filtering")
    loader = DataLoader(
        _InferenceDataset(examples, feature_mode),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_inference,
    )

    device_obj = next(loaded.model.parameters()).device
    predictions: list[str] = []
    references: list[str] = []
    output_rows: list[dict[str, object]] = []
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for features, lengths, batch_examples in loader:
            logits = loaded.model(features.to(device_obj), lengths)
            for batch_index, example in enumerate(batch_examples):
                length = int(lengths[batch_index])
                prediction = loaded.vocabulary.decode_logits(
                    logits[:length, batch_index], input_length=length
                )
                reference = example.label
                char = edit_counts(reference, prediction)
                word = edit_counts(reference.split(), prediction.split())
                unknown = "".join(sorted(set(reference) - set(loaded.vocabulary.char_to_idx)))
                predictions.append(prediction)
                references.append(reference)
                output_rows.append({
                    "source_index": example.source_index,
                    "signing_hand": example.signing_hand,
                    "reference": reference,
                    "prediction": prediction,
                    "char_edits": char["distance"],
                    "char_substitutions": char["substitutions"],
                    "char_deletions": char["deletions"],
                    "char_insertions": char["insertions"],
                    "reference_characters": len(reference),
                    "cer": char["distance"] / max(len(reference), 1),
                    "word_edits": word["distance"],
                    "word_substitutions": word["substitutions"],
                    "word_deletions": word["deletions"],
                    "word_insertions": word["insertions"],
                    "reference_words": len(reference.split()),
                    "exact_match": reference == prediction,
                    "unknown_reference_characters": unknown,
                })

    inference_seconds = time.perf_counter() - inference_started
    metrics = corpus_error_metrics(references, predictions)
    device_obj = next(loaded.model.parameters()).device
    cuda_device_name = (
        torch.cuda.get_device_name(device_obj)
        if device_obj.type == "cuda" and torch.cuda.is_available()
        else None
    )
    report: dict[str, object] = {
        "checkpoint": str(loaded.path),
        "checkpoint_epoch": loaded.epoch,
        "model_version": f"bigru_ctc_{loaded.input_dim}d",
        "n_test": len(output_rows),
        "n_valid_test_examples": valid_test_count,
        "sample_limit": max_samples,
        "metrics": metrics,
        "test_data_stats": test_stats,
        "feature_mode": feature_mode,
        "integrity": {
            "test_npz_sha256": test_npz_sha256 or sha256_file(resolved_test_npz),
            "checkpoint_sha256": checkpoint_sha256 or sha256_file(resolved_checkpoint),
        },
        "runtime": {
            "device": str(device_obj),
            "cuda_device_name": cuda_device_name,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "batch_size": batch_size,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "dataset_load_seconds": data_load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - run_started,
        },
        "coordinate_note": (
            "Input is the preprocessed NPZ pose sequence; no RGB video or MediaPipe "
            "coordinate adapter is used in this evaluation."
        ),
    }
    output_path = Path(resolved_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{loaded.input_dim}d_test_predictions.csv"
    json_path = output_path / f"{loaded.input_dim}d_test_metrics.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["predictions_csv"] = str(csv_path)
    report["metrics_json"] = str(json_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-config", type=Path, help="defaults to configs/storage.yaml")
    parser.add_argument("--checkpoint", type=Path, help="defaults to Drive backup, then local 126D checkpoint")
    parser.add_argument("--test-npz", type=Path, help="processed test NPZ; defaults to configured Drive/local subset")
    parser.add_argument("--output-dir", type=Path, help="defaults to the configured local evaluation directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, help="evaluate only the first N valid examples (smoke tests)")
    parser.add_argument("--test-npz-sha256", help="known SHA-256 from a verified copy/extraction")
    parser.add_argument("--checkpoint-sha256", help="known SHA-256 from a verified copy")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--trusted-local-npz",
        action="store_true",
        help="Required because the notebook-format NPZ contains pickle-backed object arrays.",
    )
    args = parser.parse_args()
    if not args.trusted_local_npz:
        parser.error("pass --trusted-local-npz only for a trusted local FSboard artifact")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    result = evaluate_checkpoint(
        args.checkpoint,
        args.test_npz,
        args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        trusted_local_npz=True,
        storage_config_path=args.storage_config,
        max_samples=args.max_samples,
        test_npz_sha256=args.test_npz_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

