"""Export a deterministic 16-32 sample processed FSboard NPZ subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fingerspelling.data.fsboard_dataset import load_processed_npz
from fingerspelling.storage import load_storage_config
from fingerspelling.storage.integrity import sha256_file


def export_subset(
    source_npz: str | Path,
    output_npz: str | Path,
    *,
    count: int = 24,
    seed: int = 42,
    trusted_local: bool = False,
) -> dict[str, object]:
    if not 16 <= count <= 32:
        raise ValueError("subset count must be between 16 and 32")
    if not trusted_local:
        raise ValueError("only export from a trusted local processed NPZ")
    source = Path(source_npz).expanduser().resolve()
    output = Path(output_npz).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("output path must end with .npz")
    if source == output:
        raise ValueError("output NPZ must not overwrite the source artifact")
    examples, stats = load_processed_npz(
        source, trusted_local=True, filter_ctc_impossible=False
    )
    if len(examples) < count:
        raise ValueError(f"source has only {len(examples)} usable samples; requested {count}")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(examples), size=count, replace=False))
    selected = [examples[int(index)] for index in chosen]

    left = np.empty(count, dtype=object)
    right = np.empty(count, dtype=object)
    left[:] = [None] * count
    right[:] = [None] * count
    signing_hands = np.empty(count, dtype=np.int64)
    words = np.empty(count, dtype=object)
    source_indices = np.empty(count, dtype=np.int64)
    for out_index, example in enumerate(selected):
        (right if example.signing_hand == 0 else left)[out_index] = {0: example.pose}
        signing_hands[out_index] = example.signing_hand
        words[out_index] = example.label
        source_indices[out_index] = example.source_index

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        left_hand_poses=left,
        right_hand_poses=right,
        signing_hand=signing_hands,
        words=words,
        source_indices=source_indices,
    )
    # Read the newly written trusted artifact with the same loader to ensure its format works.
    exported, exported_stats = load_processed_npz(
        output, trusted_local=True, filter_ctc_impossible=False
    )
    if len(exported) != count:
        raise RuntimeError(f"subset verification retained {len(exported)} of {count} samples")

    manifest = {
        "source_npz": str(source),
        "source_sha256": sha256_file(source),
        "output_npz": str(output),
        "output_sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "sample_count": count,
        "seed": seed,
        "source_indices": source_indices.tolist(),
        "source_stats": stats,
        "exported_stats": exported_stats,
        "format_note": "processed NPZ object arrays; trusted local artifact only",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-config", type=Path)
    parser.add_argument("--input-npz", type=Path, help="defaults to storage config / Drive resolution")
    parser.add_argument("--output", type=Path, help="defaults to the configured Windows subset directory")
    parser.add_argument("--count", type=int, default=24, help="must be from 16 to 32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trusted-local-npz",
        action="store_true",
        help="acknowledge that this local NPZ is trusted; object arrays use pickle",
    )
    args = parser.parse_args()
    if not args.trusted_local_npz:
        parser.error("pass --trusted-local-npz only for a trusted FSboard artifact")
    try:
        storage = load_storage_config(args.storage_config)
        source = storage.resolve_test_npz(args.input_npz)
        output = args.output or storage.windows_subset_dir / "fsboard_test_subset.npz"
        print(json.dumps(
            export_subset(
                source, output, count=args.count, seed=args.seed, trusted_local=True
            ),
            indent=2,
            ensure_ascii=False,
        ))
    except Exception as exc:
        parser.exit(2, f"export_npz_subset: {exc}\n")


if __name__ == "__main__":
    main()
