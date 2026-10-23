"""Strictly load a trusted local checkpoint and print recovered metadata."""

from __future__ import annotations

import argparse
import json

from fingerspelling.inference import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    loaded = load_checkpoint(args.checkpoint, args.device)
    print(json.dumps({
        "checkpoint": str(loaded.path),
        "input_dim": loaded.input_dim,
        "hidden_size": loaded.hidden_size,
        "num_layers": loaded.num_layers,
        "bidirectional": loaded.model.bidirectional,
        "num_classes": loaded.num_classes,
        "blank_index": loaded.vocabulary.blank_index,
        "vocabulary_size": len(loaded.vocabulary.char_to_idx),
        "epoch": loaded.epoch,
        "device": str(next(loaded.model.parameters()).device),
        "strict_state_dict_load": True,
    }, indent=2))


if __name__ == "__main__":
    main()

