# Checkpoints

The Task 3 ZIP's two checkpoints are available in this local directory and are excluded from Git. Keep `bigru_ctc_motion_best.pt` here on Windows for FastAPI; back up a copy to `MyDrive/TTNM_Fingerspelling/checkpoints/` with `scripts/sync_checkpoint_to_drive.py`. FastAPI loads the local file once at startup and never accesses Drive per request. Checkpoints are loaded with PyTorch `weights_only=True` and strict state-dict checks. Never accept checkpoint files uploaded by an API user.

