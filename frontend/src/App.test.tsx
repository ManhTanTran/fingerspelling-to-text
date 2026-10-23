import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const predictionPayload = {
  status: "success",
  prediction: "HELO WORLD",
  needs_review: null,
  processing_time_ms: 830,
  latency_ms: 830,
  model_version: "bigru_ctc_126d",
  device: "cpu",
  input_duration_seconds: 2,
  input_fps: 30,
  input_width: 640,
  input_height: 480,
  input_frames: 60,
  landmark_frames: 56,
  hand_detection_rate: 0.94,
  selected_hand: "right",
  feature_dim: 126,
  coordinate_adapter: "pixel_xy_preserve_z",
  coordinate_compatibility_verified: false,
  stage_timings_ms: { model_forward: 5, ctc_decode: 1 },
  upload_size_bytes: 64000,
  note: "Experimental output.",
};

describe("FSboard review and confirmation flow", () => {
  beforeEach(() => {
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:sample") });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps model draft separate until the user confirms, then supports undo and clear", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => predictionPayload });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    const file = new File(["video"], "signing-sample.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByLabelText("Upload MP4 or WebM video"), { target: { files: [file] } });
    await user.click(screen.getByRole("button", { name: "Recognize video" }));

    expect(await screen.findByText("HELO WORLD", { selector: "q" })).toBeInTheDocument();
    const draft = screen.getByRole("textbox", { name: "Edit predicted text" });
    expect(draft).toHaveValue("HELO WORLD");
    expect(screen.getByRole("textbox", { name: "Edit final text" })).toHaveValue("");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("http://127.0.0.1:8000/predict");

    await user.clear(draft);
    await user.type(draft, "HELLO WORLD");
    await user.click(screen.getByRole("button", { name: "Confirm text" }));
    const finalText = screen.getByRole("textbox", { name: "Edit final text" });
    expect(finalText).toHaveValue("HELLO WORLD");
    expect(screen.queryByRole("textbox", { name: "Edit predicted text" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Undo" }));
    expect(finalText).toHaveValue("");
    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByText("No video selected")).toBeInTheDocument();
  });

  it("shows API errors and lets the user retry the selected clip", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 400, json: async () => ({ status: "error", message: "No hand landmarks were found." }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => predictionPayload });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.change(screen.getByLabelText("Upload MP4 or WebM video"), {
      target: { files: [new File(["video"], "sample.webm", { type: "video/webm" })] },
    });
    await user.click(screen.getByRole("button", { name: "Recognize video" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No hand landmarks were found.");

    await user.click(screen.getByRole("button", { name: "Retry recognition" }));
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Edit predicted text" })).toHaveValue("HELO WORLD"));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
