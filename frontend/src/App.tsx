import { useCallback, useEffect, useRef, useState } from "react";
import { recognizeVideo, type PredictionResponse } from "./api";

type Status = "ready" | "recording" | "processing" | "prediction" | "error";
type CaptureMode = "camera" | "upload";
type ServiceStatus = "checking" | "ready" | "not_ready" | "offline";
type EventName =
  | "recognition_started"
  | "recognition_completed"
  | "prediction_received"
  | "edit_started"
  | "text_modified"
  | "retry"
  | "confirm"
  | "undo"
  | "clear"
  | "task_completed";
type InteractionEvent = { event: EventName; at: string; details?: Record<string, number | string> };

const statusCopy: Record<Status, string> = {
  ready: "Ready for a clip",
  recording: "Recording from your camera",
  processing: "Finding hand shapes and decoding text",
  prediction: "Draft ready for your review",
  error: "Recognition needs attention",
};

const serviceStatusCopy: Record<ServiceStatus, string> = {
  checking: "Checking",
  ready: "Online",
  not_ready: "Needs setup",
  offline: "Offline",
};

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function LandmarkMark() {
  const points = [
    [92, 25], [68, 30], [48, 43], [35, 58], [26, 71],
    [115, 36], [107, 55], [100, 72], [94, 88],
    [135, 53], [131, 74], [127, 94], [122, 112],
    [151, 76], [153, 97], [154, 116], [154, 134],
    [163, 102], [171, 120], [177, 138], [180, 156],
  ];
  const bones = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
  return (
    <svg className="landmark-mark" viewBox="0 0 210 180" role="img" aria-label="Hand landmark guide">
      {bones.map(([a, b]) => <line key={`${a}-${b}`} x1={points[a][0]} y1={points[a][1]} x2={points[b][0]} y2={points[b][1]} />)}
      {points.map(([cx, cy], index) => <circle key={index} cx={cx} cy={cy} r={index === 0 ? 5 : 3.2} />)}
    </svg>
  );
}

function App() {
  const [status, setStatus] = useState<Status>("ready");
  const [captureMode, setCaptureMode] = useState<CaptureMode>("camera");
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus>("checking");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [draftText, setDraftText] = useState("");
  const [confirmedText, setConfirmedText] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [hand, setHand] = useState<"auto" | "left" | "right">("auto");
  const [error, setError] = useState("");
  const [cameraError, setCameraError] = useState("");
  const [cameraOn, setCameraOn] = useState(false);
  const [consent, setConsent] = useState(false);
  const [events, setEvents] = useState<InteractionEvent[]>([]);
  const [copyMessage, setCopyMessage] = useState("");
  const [browserElapsedMs, setBrowserElapsedMs] = useState<number | null>(null);
  const cameraRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const editStartedRef = useRef(false);

  const checkService = useCallback(async () => {
    setServiceStatus("checking");
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const payload: unknown = await response.json().catch(() => null);
      const ready = response.ok && payload !== null && typeof payload === "object" &&
        "video_inference_ready" in payload && payload.video_inference_ready === true;
      setServiceStatus(ready ? "ready" : "not_ready");
    } catch {
      setServiceStatus("offline");
    }
  }, []);

  const logEvent = useCallback((event: EventName, details?: InteractionEvent["details"]) => {
    if (!consent) return;
    setEvents((current) => [...current, { event, at: new Date().toISOString(), ...(details ? { details } : {}) }]);
  }, [consent]);

  useEffect(() => {
    if (cameraRef.current && streamRef.current) {
      cameraRef.current.srcObject = streamRef.current;
      void cameraRef.current.play().catch(() => undefined);
    }
  }, [cameraOn]);

  useEffect(() => {
    void checkService();
    const interval = window.setInterval(() => void checkService(), 30000);
    return () => window.clearInterval(interval);
  }, [checkService]);

  useEffect(() => {
    if (!videoFile) {
      setPreviewUrl("");
      return;
    }
    const nextUrl = URL.createObjectURL(videoFile);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [videoFile]);

  useEffect(() => () => {
    recorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  const stopCamera = () => {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
  };

  const chooseCaptureMode = (mode: CaptureMode) => {
    if (mode === captureMode || videoFile || status === "recording" || status === "processing") return;
    if (streamRef.current) stopCamera();
    setCaptureMode(mode);
    setPrediction(null);
    setDraftText("");
    setBrowserElapsedMs(null);
    setError("");
    setCameraError("");
    setStatus("ready");
  };

  const startCamera = async () => {
    setCameraError("");
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not provide webcam access.");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      streamRef.current = stream;
      setCameraOn(true);
    } catch (cause) {
      setCameraError(cause instanceof Error ? cause.message : "Camera access was denied.");
    }
  };

  const startRecording = () => {
    const stream = streamRef.current;
    if (!stream) {
      setCameraError("Start the camera before recording.");
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      setCameraError("This browser cannot record video. Upload an MP4 or WebM clip instead.");
      return;
    }
    const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"]
      .find((candidate) => MediaRecorder.isTypeSupported(candidate));
    if (!mimeType) {
      setCameraError("No supported MP4 or WebM recording format was found. Upload a clip instead.");
      return;
    }
    try {
      chunksRef.current = [];
      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => { if (event.data.size > 0) chunksRef.current.push(event.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const extension = mimeType.includes("webm") ? "webm" : "mp4";
        if (!blob.size) {
          setCameraError("The browser recorded an empty clip. Try again or upload a video.");
          setStatus("ready");
          return;
        }
        setVideoFile(new File([blob], `fingerspelling-${Date.now()}.${extension}`, { type: mimeType }));
        setPrediction(null);
        setBrowserElapsedMs(null);
        setDraftText("");
        setStatus("ready");
      };
      recorder.start(250);
      setStatus("recording");
    } catch (cause) {
      setCameraError(cause instanceof Error ? cause.message : "The browser could not start recording.");
    }
  };

  const stopRecording = () => {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  };

  const selectFile = (file?: File) => {
    if (!file) return;
    const supported = /\.(mp4|webm)$/i.test(file.name);
    if (!supported) {
      setError("Choose an MP4 or WebM video. Other formats are not accepted by the recognition service.");
      return;
    }
    setError("");
    setCameraError("");
    setCaptureMode("upload");
    setPrediction(null);
    setBrowserElapsedMs(null);
    setDraftText("");
    setVideoFile(file);
    setStatus("ready");
    if (streamRef.current) stopCamera();
  };

  const removeClip = () => {
    setVideoFile(null);
    setPrediction(null);
    setDraftText("");
    setBrowserElapsedMs(null);
    setError("");
    setStatus("ready");
  };

  const runRecognition = async () => {
    if (!videoFile || status === "processing") return;
    setStatus("processing");
    setError("");
    setCopyMessage("");
    setBrowserElapsedMs(null);
    editStartedRef.current = false;
    logEvent(prediction ? "retry" : "recognition_started", { source: "video" });
    try {
      const browserStarted = performance.now();
      const result = await recognizeVideo(videoFile, hand);
      setBrowserElapsedMs(performance.now() - browserStarted);
      setPrediction(result);
      setDraftText(result.prediction);
      setStatus("prediction");
      logEvent("recognition_completed", { elapsed_ms: Math.round(result.processing_time_ms) });
      logEvent("prediction_received", { characters: result.prediction.length });
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Recognition failed. Check the video and try again.";
      setError(message);
      setStatus("error");
      logEvent("recognition_completed", { outcome: "error" });
    }
  };

  const confirmDraft = () => {
    const text = draftText.trim();
    if (!text) return;
    setHistory((current) => [...current, confirmedText]);
    setConfirmedText((current) => current ? `${current}\n${text}` : text);
    setDraftText("");
    setPrediction(null);
    setBrowserElapsedMs(null);
    setStatus("ready");
    editStartedRef.current = false;
    logEvent("confirm", { characters: text.length });
    logEvent("task_completed");
  };

  const undo = () => {
    if (!history.length) return;
    setConfirmedText(history[history.length - 1]);
    setHistory((current) => current.slice(0, -1));
    logEvent("undo");
  };

  const clearAll = () => {
    stopCamera();
    setHistory((current) => [...current, confirmedText]);
    setConfirmedText("");
    setDraftText("");
    setPrediction(null);
    setBrowserElapsedMs(null);
    setVideoFile(null);
    setError("");
    setCameraError("");
    setStatus("ready");
    editStartedRef.current = false;
    logEvent("clear");
  };

  const copyText = async () => {
    if (!confirmedText) return;
    try {
      await navigator.clipboard.writeText(confirmedText);
      setCopyMessage("Copied");
    } catch {
      setCopyMessage("Copy is unavailable in this browser context.");
    }
  };

  const exportEvents = () => {
    const payload = new Blob([JSON.stringify(events, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(payload);
    const link = document.createElement("a");
    link.href = url;
    link.download = "fsboard-interaction-events.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const editDraft = (value: string) => {
    if (!editStartedRef.current) {
      logEvent("edit_started");
      editStartedRef.current = true;
    }
    setDraftText(value);
    logEvent("text_modified", { characters: value.length });
  };

  const editConfirmed = (value: string) => {
    setConfirmedText(value);
    logEvent("text_modified", { characters: value.length });
  };

  const removeConfirmedPhrase = (phraseIndex: number) => {
    const phrases = confirmedText.split(/\n+/).filter((phrase) => phrase.trim());
    if (phraseIndex < 0 || phraseIndex >= phrases.length) return;
    phrases.splice(phraseIndex, 1);
    setHistory((current) => [...current, confirmedText]);
    const nextText = phrases.join("\n");
    setConfirmedText(nextText);
    logEvent("text_modified", { characters: nextText.length });
  };

  const downloadName = videoFile?.name ?? "No video selected";
  const confirmedPhrases = confirmedText.split(/\n+/).filter((phrase) => phrase.trim());
  const currentStep = prediction || confirmedText ? 2 : videoFile ? 1 : 0;
  const workflowSteps = [
    { title: "Add a clip", detail: "Record or upload", complete: Boolean(videoFile || prediction || confirmedText) },
    { title: "Create a draft", detail: "Run recognition", complete: Boolean(prediction || confirmedText) },
    { title: "Review and confirm", detail: "Keep the text you approve", complete: Boolean(confirmedText) },
  ].map((step, index) => ({ ...step, current: index === currentStep }));

  return (
    <main className="app-frame">
      <div className="app-shell" id="top">
        <header className="topbar">
          <a className="brand" href="#top" aria-label="FSboard home">
            <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>
            <span>FSboard</span>
          </a>
          <div className="topbar-right">
            <span className="model-pill">BiGRU + CTC <span>126D</span></span>
            <button className={`service-pill service-${serviceStatus}`} type="button" onClick={() => void checkService()} title="Check the local recognition service" aria-live="polite">
              <i />{serviceStatusCopy[serviceStatus]}
            </button>
            <a className="quiet-link" href={`${API_BASE_URL}/health`} target="_blank" rel="noreferrer">Service details</a>
          </div>
        </header>

        <section className="page-heading" aria-labelledby="page-title">
          <div>
            <h1 id="page-title">Fingerspelling to text</h1>
            <p>Add a short clip, review the model draft, then confirm the text you want to keep.</p>
          </div>
          <span className={`page-state status-${status}`} aria-live="polite">{statusCopy[status]}</span>
        </section>

        <nav className="workflow-steps" aria-label="Recognition workflow">
          {workflowSteps.map((step) => (
            <div key={step.title} className={`workflow-step ${step.current ? "is-current" : ""} ${step.complete ? "is-complete" : ""}`} aria-current={step.current ? "step" : undefined}>
              <span className="workflow-mark" aria-hidden="true" />
              <span className="workflow-copy"><strong>{step.title}</strong><small>{step.detail}</small></span>
            </div>
          ))}
        </nav>

        <section className="workbench" aria-label="Recognition workspace">
          <div className="capture-column">
            <div className="section-heading">
              <h2>Your clip</h2>
              {status === "recording" && <span className="recording-label"><i />Recording</span>}
            </div>

            <div className="capture-mode-switch" role="group" aria-label="Choose a video input method">
              <button className={captureMode === "camera" ? "is-selected" : ""} type="button" aria-pressed={captureMode === "camera"} onClick={() => chooseCaptureMode("camera")} disabled={status === "recording" || status === "processing" || Boolean(videoFile && captureMode !== "camera")}>Record with camera</button>
              <button className={captureMode === "upload" ? "is-selected" : ""} type="button" aria-pressed={captureMode === "upload"} onClick={() => chooseCaptureMode("upload")} disabled={status === "recording" || status === "processing" || Boolean(videoFile && captureMode !== "upload")}>Upload a clip</button>
            </div>
            {videoFile && <p className="capture-mode-note">Remove the current clip to switch input methods.</p>}

            <div className={`preview-frame ${cameraOn && !videoFile ? "preview-camera" : ""}`}>
              {videoFile && previewUrl ? (
                <video key={previewUrl} className="selected-video" src={previewUrl} controls playsInline />
              ) : cameraOn ? (
                <video ref={cameraRef} className="camera-video" autoPlay muted playsInline aria-label="Camera preview" />
              ) : (
                <div className="empty-preview">
                  <div className="landmark-disc"><LandmarkMark /></div>
                  <div>
                    <p>Keep one signing hand visible</p>
                    <span>Frame the hand from wrist to fingertips in clear light.</span>
                  </div>
                </div>
              )}
              {status === "processing" && <div className="processing-shade"><span className="spinner" /><span>Reading the movement</span></div>}
              {status === "recording" && <span className="recording-indicator"><i />Recording</span>}
            </div>

            <div className="file-row">
              <span className="file-name" title={downloadName}>{downloadName}</span>
              {videoFile && <button className="text-button" type="button" onClick={removeClip}>Remove clip</button>}
            </div>

            <div className="capture-actions">
              {captureMode === "camera" ? (
                !cameraOn ? (
                  <button className="button button-primary" type="button" onClick={() => void startCamera()}>Open camera</button>
                ) : status === "recording" ? (
                  <button className="button button-primary" type="button" onClick={stopRecording}>Stop recording</button>
                ) : (
                  <>
                    <button className="button button-primary" type="button" onClick={startRecording}>Start recording</button>
                    <button className="button button-secondary" type="button" onClick={stopCamera}>Close camera</button>
                  </>
                )
              ) : (
                <label className="button button-primary upload-button">
                  {videoFile ? "Choose another clip" : "Choose video file"}
                  <input type="file" accept="video/mp4,video/webm,.mp4,.webm" onChange={(event) => { selectFile(event.currentTarget.files?.[0]); event.currentTarget.value = ""; }} aria-label="Upload MP4 or WebM video" />
                </label>
              )}
              {captureMode === "camera" && <span className="capture-hint">Camera opens only when you choose.</span>}
            </div>
            {cameraError && <p className="inline-error" role="alert">{cameraError}</p>}

            <div className="clip-settings">
              <div className="hand-setting">
                <label htmlFor="hand-side">Signing hand</label>
                <select id="hand-side" value={hand} onChange={(event) => setHand(event.currentTarget.value as typeof hand)}>
                  <option value="auto">Choose automatically</option>
                  <option value="left">Left hand</option>
                  <option value="right">Right hand</option>
                </select>
              </div>
              <span className="format-note">MP4 or WebM · up to 100 MB · 120 seconds</span>
            </div>
          </div>

          <section className="review-column" aria-label="Prediction and final text">
            <div className="review-heading">
              <div>
                <h2>Review your text</h2>
                <p>Model output stays editable until you confirm it.</p>
              </div>
              {prediction && <span className="review-status">Needs review</span>}
            </div>

            <div className="editor-block">
              <section className="draft-panel" aria-labelledby="draft-heading">
                <div className="panel-heading">
                  <h3 id="draft-heading">Draft</h3>
                  {prediction && <span className="model-chip">{prediction.model_version}</span>}
                </div>
                {prediction ? (
                  <>
                    <p className="original-prediction">Raw model output: <q>{prediction.prediction || "(no characters decoded)"}</q></p>
                    <label className="sr-only" htmlFor="draft-text">Edit predicted text</label>
                    <textarea id="draft-text" value={draftText} onChange={(event) => editDraft(event.currentTarget.value)} placeholder="Edit the prediction before confirming" />
                    <div className="prediction-meta">
                      <span>{prediction.input_frames} frames · {prediction.input_duration_seconds.toFixed(1)} s · {prediction.landmark_frames} hand frames · {(prediction.hand_detection_rate * 100).toFixed(0)}% detection</span>
                      <span>{prediction.processing_time_ms.toFixed(0)} ms backend · {browserElapsedMs?.toFixed(0) ?? "—"} ms browser · {prediction.device}</span>
                    </div>
                    <details className="timing-details">
                      <summary>Processing timings</summary>
                      <dl>{Object.entries(prediction.stage_timings_ms).map(([stage, milliseconds]) => <div key={stage}><dt>{stage.replaceAll("_", " ")}</dt><dd>{milliseconds.toFixed(1)} ms</dd></div>)}</dl>
                    </details>
                  </>
                ) : (
                  <div className="draft-empty">
                    <span className="empty-mark" aria-hidden="true" />
                    <div><strong>No draft yet</strong><p>Add a clip and run recognition. You can edit the result before keeping it.</p></div>
                  </div>
                )}
                {error && <p className="inline-error" role="alert">{error}</p>}
                <div className="draft-actions">
                  <button className="button button-primary" type="button" onClick={() => void runRecognition()} disabled={!videoFile || status === "processing" || status === "recording"}>
                    {status === "processing" ? <><span className="button-spinner" /> Recognizing…</> : prediction || status === "error" ? "Retry recognition" : "Recognize video"}
                  </button>
                  <button className="button button-secondary" type="button" onClick={confirmDraft} disabled={!draftText.trim() || !prediction}>Confirm text</button>
                </div>
                {prediction && <p className="coordinate-warning">Feature compatibility with FSboard training data is not verified.</p>}
              </section>

              <section className="confirmed-panel" aria-labelledby="final-heading">
                <div className="panel-heading">
                  <div><h3 id="final-heading">Final text</h3><p>Text you have confirmed</p></div>
                  <span className="confirmed-count">{confirmedText.length} characters</span>
                </div>
                {confirmedPhrases.length > 0 && (
                  <div className="phrase-chips" aria-label="Confirmed phrases">
                    {confirmedPhrases.map((phrase, index) => (
                      <button className="phrase-chip" key={`${index}-${phrase}`} type="button" onClick={() => removeConfirmedPhrase(index)} aria-label={`Remove phrase: ${phrase}`} title="Remove this phrase">
                        <span>{phrase}</span><i aria-hidden="true">×</i>
                      </button>
                    ))}
                  </div>
                )}
                <label className="sr-only" htmlFor="confirmed-text">Edit final text</label>
                <textarea id="confirmed-text" value={confirmedText} onChange={(event) => editConfirmed(event.currentTarget.value)} placeholder="Confirmed phrases appear here. You can still edit this text." />
                <div className="confirmed-actions">
                  <button className="text-button" type="button" onClick={() => void copyText()} disabled={!confirmedText}>Copy text</button>
                  <span aria-live="polite">{copyMessage}</span>
                  <button className="text-button" type="button" onClick={undo} disabled={!history.length} title="Undo the last confirmed phrase or clear action">Undo</button>
                  <button className="text-button text-danger" type="button" onClick={clearAll} disabled={!confirmedText && !draftText && !videoFile}>Clear</button>
                </div>
              </section>
            </div>
          </section>
        </section>

        <footer className="privacy-footer">
          <div className="privacy-copy">
            <strong>Your clip stays temporary.</strong>
            <span>The backend removes the uploaded video after processing. This page does not save video by default.</span>
          </div>
          <div className="event-consent">
            <label><input type="checkbox" checked={consent} onChange={(event) => setConsent(event.currentTarget.checked)} /> Allow a local interaction log</label>
            <span>Only event names, times and character counts; never the video or text content.</span>
            {consent && <button className="text-button" type="button" onClick={exportEvents} disabled={!events.length}>Download {events.length} events</button>}
          </div>
        </footer>
      </div>
    </main>
  );
}

export default App;
