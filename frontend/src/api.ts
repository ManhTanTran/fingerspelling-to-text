export type PredictionResponse = {
  status: "success";
  prediction: string;
  needs_review: null;
  processing_time_ms: number;
  latency_ms: number;
  model_version: string;
  device: string;
  input_duration_seconds: number;
  input_fps: number;
  input_width: number;
  input_height: number;
  input_frames: number;
  landmark_frames: number;
  hand_detection_rate: number;
  selected_hand: string;
  feature_dim: number;
  coordinate_adapter: string;
  coordinate_compatibility_verified: false;
  stage_timings_ms: Record<string, number>;
  upload_size_bytes: number;
  note: string;
};

export type ApiError = { status: "error"; message: string; details?: unknown };

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export async function recognizeVideo(
  file: File,
  hand: "left" | "right" | "auto" = "auto",
): Promise<PredictionResponse> {
  const body = new FormData();
  body.append("file", file, file.name);
  if (hand !== "auto") body.append("hand", hand);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, { method: "POST", body });
  } catch {
    throw new Error(`Cannot reach the recognition service at ${API_BASE_URL}. Start FastAPI and try again.`);
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "message" in payload && typeof payload.message === "string"
        ? payload.message
        : `Recognition failed with HTTP ${response.status}.`;
    throw new Error(message);
  }
  if (
    !payload ||
    typeof payload !== "object" ||
    !("status" in payload) || payload.status !== "success" ||
    !("prediction" in payload) || typeof payload.prediction !== "string" ||
    !("processing_time_ms" in payload) || typeof payload.processing_time_ms !== "number"
  ) {
    throw new Error("Recognition service returned an invalid response. Check that the backend is up to date.");
  }
  return payload as PredictionResponse;
}
