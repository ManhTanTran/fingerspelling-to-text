# MediaPipe model asset

Place a compatible MediaPipe Tasks Hand Landmarker bundle here as:

```text
models/hand_landmarker.task
```

The `.task` file is required by `scripts/predict_video.py` and the API. The official MediaPipe Hand Landmarker model bundle is prepared locally at this path and is excluded from Git. Its current SHA-256 is `FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1`. Install `.[video]` to run it; the source is the official [Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python). A real video has not yet been tested.
