# Data

Keep the full FSboard dataset and derived NPZ files out of Git. Google Drive is the primary store at `MyDrive/TTNM_Fingerspelling/datasets/FSboard/`. The original notebook uses source ID `1dhcrbZeb3-5IshIkzatVmNO3MOTVFjJW`; Drive metadata resolves it to `fingerspelling_fsboard_proc_clean.zip` (6,361,248,095 bytes). Content access has not yet been verified from the desktop session. The Colab launcher performs an authenticated four-byte media read before downloading; only after that succeeds does it download the ZIP to `/content/fsboard_cache/source_archives/`, verify the Drive size/MD5 and SHA-256, stream only `fsboard_test_proc_clean.npz` into Colab scratch with ZIP CRC and SHA-256 checks, and copy that derived NPZ to Drive. It does not copy the full dataset to Windows or modify the ZIP.

For Windows pipeline checks, export a deterministic 16–32 sample file with `scripts/export_npz_subset.py` and place only `fsboard_test_subset.npz` under `data/windows_subset/`. The subset remains ignored by Git. See [the Drive storage guide](../docs/GOOGLE_DRIVE_STORAGE.md).

Only use `--trusted-local-npz` with a dataset artifact obtained from a trusted source. The notebook-format NPZ uses object arrays and requires pickle-backed loading; never accept an API upload as an NPZ input.

