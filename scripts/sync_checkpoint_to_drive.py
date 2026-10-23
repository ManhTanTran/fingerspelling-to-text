"""Back up the local 126D FastAPI checkpoint into the configured Drive folder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fingerspelling.storage import load_storage_config
from fingerspelling.storage.integrity import copy_verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        storage = load_storage_config(args.storage_config)
        local_path = args.checkpoint or Path(
            os.environ.get("FSBOARD_LOCAL_CHECKPOINT", storage.local_motion_checkpoint)
        )
        local_path = local_path.expanduser().resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"local checkpoint not found: {local_path}")
        if storage.drive_checkpoints_dir is None:
            raise RuntimeError(
                "Google Drive is not configured. Mount it or set FSBOARD_DRIVE_ROOT "
                "to the TTNM_Fingerspelling folder."
            )
        destination = storage.drive_checkpoints_dir / local_path.name
        result = copy_verified(local_path, destination, overwrite=args.overwrite)
        manifest_path = destination.with_suffix(destination.suffix + ".sha256.json")
        manifest_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({**result, "manifest": str(manifest_path)}, indent=2, ensure_ascii=False))
    except Exception as exc:
        parser.exit(2, f"sync_checkpoint_to_drive: {exc}\n")


if __name__ == "__main__":
    main()
