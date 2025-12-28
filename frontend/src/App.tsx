import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import "./global.css";

type StreamState = "idle" | "starting" | "active" | "error";
type ModelState = "idle" | "loading" | "ready" | "error";
type EventKind = "video" | "audio" | "system";
type Severity = "info" | "warn" | "error";

type EventRecord = {
  id: string;
  ts: string;
  message: string;
  severity: Severity;
  kind: EventKind;
};

type FaceMetrics = {
  distanceCm: number;
  yawDeg: number;
};

type FaceLandmarks = FaceLandmarkerResult["faceLandmarks"][number];

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const MEDIAPIPE_WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm";
const FACE_LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

const DEFAULT_IPD_CM = 6.3;
const BASELINE_VIDEO_WIDTH = 1280;
const BASELINE_FOCAL_PX = 750;
const HEAD_YAW_MAX_DEG = 90;
const RAD2DEG = 180 / Math.PI;
const YAW_OFFSET_SCALE = 0.55;
const DISTANCE_RANGE_CM = { min: 10, max: 150 } as const;
const LANDMARK_INDEX = {
  rightEyeOuter: 33,
  leftEyeOuter: 263,
  noseTip: 1
} as const;

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const distance2D = (a: { x: number; y: number }, b: { x: number; y: number }) => {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
};

const getLandmark = (landmarks: FaceLandmarks, index: number) => landmarks[index] ?? null;

const computeFaceMetrics = (landmarks: FaceLandmarks, videoWidth: number): FaceMetrics | null => {
  if (!videoWidth) {
    return null;
  }
  const leftEye = getLandmark(landmarks, LANDMARK_INDEX.leftEyeOuter);
  const rightEye = getLandmark(landmarks, LANDMARK_INDEX.rightEyeOuter);
  const noseTip = getLandmark(landmarks, LANDMARK_INDEX.noseTip);

  if (!leftEye || !rightEye || !noseTip) {
    return null;
  }

  const interpupilNorm = distance2D(leftEye, rightEye);
  if (!interpupilNorm) {
    return null;
  }
  const interpupilPx = interpupilNorm * videoWidth;
  const widthScale = videoWidth / BASELINE_VIDEO_WIDTH;
  const effectiveFocalPx = BASELINE_FOCAL_PX * widthScale;
  const distanceCmRaw = (DEFAULT_IPD_CM * effectiveFocalPx) / interpupilPx;
  const distanceCm = clamp(distanceCmRaw, DISTANCE_RANGE_CM.min, DISTANCE_RANGE_CM.max);

  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const noseOffset = noseTip.x - eyeMidX;
  const yawRatio = clamp(noseOffset / (interpupilNorm * YAW_OFFSET_SCALE), -1, 1);
  const yawDegRaw = clamp(-Math.asin(yawRatio) * RAD2DEG, -HEAD_YAW_MAX_DEG, HEAD_YAW_MAX_DEG);

  return {
    distanceCm,
    yawDeg: yawDegRaw,
  };
};

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const lastEventRef = useRef<Record<string, number>>({});
  const sessionId = useMemo(
    () => (typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now()}`),
    []
  );

  const [cameraState, setCameraState] = useState<StreamState>("idle");
  const [micState, setMicState] = useState<StreamState>("idle");
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [faceCount, setFaceCount] = useState(0);
  const [primaryYaw, setPrimaryYaw] = useState<number | null>(null);
  const [primaryDistance, setPrimaryDistance] = useState<number | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [speechActive, setSpeechActive] = useState(false);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [overlayMessage, setOverlayMessage] = useState<string | null>("Requesting camera + mic access");

  const addLocalEvent = useCallback((entry: EventRecord) => {
    setEvents((current) => [...current.slice(-9), entry]);
  }, []);

  const postEvent = useCallback(
    async (key: string, kind: EventKind, severity: Severity, message: string) => {
      const now = Date.now();
      const last = lastEventRef.current[key];
      if (last && now - last < 4000) {
        return;
      }
      lastEventRef.current[key] = now;

      const entry: EventRecord = {
        id: `${now}-${Math.random().toString(16).slice(2)}`,
        ts: new Date(now).toISOString(),
        message,
        severity,
        kind,
      };
      addLocalEvent(entry);

      try {
        await fetch(`${API_BASE}/events`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            kind,
            severity,
            message,
            ts: entry.ts,
          }),
        });
      } catch (error) {
        console.warn("Failed to send event to backend", error);
      }
    },
    [addLocalEvent, sessionId]
  );

  useEffect(() => {
    let cancelled = false;

    async function loadFaceLandmarker() {
      setModelState("loading");
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_BASE);
        const instance = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL },
          runningMode: "VIDEO",
          numFaces: 2,
        });

        if (cancelled) {
          instance.close();
          return;
        }

        faceLandmarkerRef.current = instance;
        setModelState("ready");
      } catch (error) {
        console.error("Unable to load MediaPipe face landmarker", error);
        setModelState("error");
        setOverlayMessage("Face model failed to load");
      }
    }

    loadFaceLandmarker();

    return () => {
      cancelled = true;
      animationFrameRef.current && cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
      faceLandmarkerRef.current?.close();
      faceLandmarkerRef.current = null;
    };
  }, []);

  const syncCanvasSize = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      return;
    }
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
  }, []);

  const drawLandmarks = useCallback((result?: FaceLandmarkerResult) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const primaryFace = result?.faceLandmarks?.[0];
    if (!primaryFace) {
      return;
    }

    ctx.fillStyle = "rgba(245, 217, 126, 0.95)";
    ctx.strokeStyle = "rgba(12, 18, 33, 0.8)";
    ctx.lineWidth = 1;

    primaryFace.forEach((landmark) => {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }, []);

  const stopStreams = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState("idle");
    setMicState("idle");
    setOverlayMessage("Camera and mic stopped");
    audioContextRef.current?.close();
    audioContextRef.current = null;
    analyserRef.current = null;
  }, []);

  const startStreams = useCallback(async () => {
    if (cameraState === "starting") {
      return;
    }
    setCameraState("starting");
    setMicState("starting");
    setOverlayMessage("Requesting camera + mic access");

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 24, max: 30 },
        },
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = mediaStream;
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.muted = true;
        await videoRef.current.play();
      }

      const audioCtx = new AudioContext();
      const analyser = audioCtx.createAnalyser();
      analyser.smoothingTimeConstant = 0.5;
      analyser.fftSize = 1024;
      const source = audioCtx.createMediaStreamSource(mediaStream);
      source.connect(analyser);
      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;

      setCameraState("active");
      setMicState("active");
      setOverlayMessage(null);
      postEvent("media-on", "system", "info", "Camera and mic granted");
    } catch (error) {
      console.error("Unable to start streams", error);
      setCameraState("error");
      setMicState("error");
      setOverlayMessage("Permissions denied or unavailable");
      postEvent("media-error", "system", "error", "Failed to start camera or mic");
    }
  }, [cameraState, postEvent]);

  useEffect(() => {
    startStreams();
    return () => {
      stopStreams();
    };
  }, [startStreams, stopStreams]);

  useEffect(() => {
    let cancelled = false;
    const TIME_DOMAIN_SIZE = 1024;
    const dataArray = new Uint8Array(TIME_DOMAIN_SIZE);
    const speechThresholdOn = 0.05;
    const speechThresholdOff = 0.025;

    const analyze = () => {
      if (cancelled) {
        return;
      }

      const video = videoRef.current;
      const landmarker = faceLandmarkerRef.current;

      if (video && landmarker && modelState === "ready" && video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA) {
        syncCanvasSize();
        const result = landmarker.detectForVideo(video, performance.now());
        drawLandmarks(result);

        const count = result.faceLandmarks?.length ?? 0;
        setFaceCount(count);
        if (count > 1) {
          postEvent("multi-face", "video", "warn", "Multiple faces detected");
        }
        const primaryFace = result.faceLandmarks?.[0];
        if (primaryFace) {
          const metrics = computeFaceMetrics(primaryFace, video.videoWidth);
          if (metrics) {
            setPrimaryYaw(metrics.yawDeg);
            setPrimaryDistance(metrics.distanceCm);
            if (Math.abs(metrics.yawDeg) > 35) {
              postEvent("yaw", "video", "warn", "Viewer looking away from screen");
            }
          }
        } else {
          setPrimaryYaw(null);
          setPrimaryDistance(null);
        }
      } else {
        setFaceCount(0);
        drawLandmarks();
      }

      const analyser = analyserRef.current;
      if (analyser) {
        analyser.getByteTimeDomainData(dataArray);
        let sumSquares = 0;
        for (let i = 0; i < dataArray.length; i += 1) {
          const centered = (dataArray[i] - 128) / 128;
          sumSquares += centered * centered;
        }
        const rms = Math.sqrt(sumSquares / dataArray.length);
        const level = clamp(rms * 4, 0, 1);
        setAudioLevel(level);

        if (!speechActive && rms > speechThresholdOn) {
          setSpeechActive(true);
          postEvent("speech-on", "audio", "info", "Speech detected");
        } else if (speechActive && rms < speechThresholdOff) {
          setSpeechActive(false);
          postEvent("speech-off", "audio", "info", "Silence detected");
        }
      }

      animationFrameRef.current = requestAnimationFrame(analyze);
    };

    animationFrameRef.current = requestAnimationFrame(analyze);

    return () => {
      cancelled = true;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [drawLandmarks, modelState, postEvent, speechActive, syncCanvasSize]);

  const isCameraActive = cameraState === "active";
  const isMicActive = micState === "active";

  return (
    <main className="app-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">Proctoring prototype</p>
          <h1>Video + Audio Watcher</h1>
          <p className="lede">
            Local-only capture with light heuristics: face presence, gaze drift, and speech activity. Events are pushed to the FastAPI backend for debugging.
          </p>
        </div>
        <div className="session-tag">Session {sessionId.slice(-6)}</div>
      </header>

      <section className="status-grid">
        <div className={`status-card ${isCameraActive ? "ok" : "warn"}`}>
          <p className="label">Camera</p>
          <p className="value">{isCameraActive ? "Active" : cameraState === "starting" ? "Requesting" : "Off"}</p>
          <small>{overlayMessage ?? "Streaming stays on device"}</small>
        </div>
        <div className={`status-card ${isMicActive ? "ok" : "warn"}`}>
          <p className="label">Microphone</p>
          <p className="value">{isMicActive ? "Active" : micState === "starting" ? "Requesting" : "Off"}</p>
          <div className="meter">
            <div className="meter-fill" style={{ width: `${audioLevel * 100}%` }} />
          </div>
          <small>{speechActive ? "Speech detected" : "Silence"}</small>
        </div>
        <div className="status-card">
          <p className="label">Faces in frame</p>
          <p className="value">{faceCount}</p>
          <small>{faceCount > 1 ? "Multiple people present" : faceCount === 1 ? "Single face" : "No face"}</small>
        </div>
        <div className="status-card">
          <p className="label">Gaze + distance</p>
          <p className="value">
            {primaryYaw !== null ? `${primaryYaw.toFixed(1)}°` : "--"}
            <span className="muted"> | </span>
            {primaryDistance !== null ? `${primaryDistance.toFixed(0)} cm` : "--"}
          </p>
          <small>{primaryYaw !== null ? "Yaw from nose vs eyes" : "Waiting for a face"}</small>
        </div>
      </section>

      <section className="video-panel">
        <video ref={videoRef} autoPlay playsInline muted className={isCameraActive ? "ready" : "dimmed"} />
        <canvas ref={canvasRef} className="overlay" />
        {!isCameraActive && overlayMessage && (
          <div className="video-overlay">
            <p>{overlayMessage}</p>
          </div>
        )}
        <div className="pill-row">
          <span className={`pill ${isCameraActive ? "pill-ok" : "pill-warn"}`}>Camera {isCameraActive ? "on" : "off"}</span>
          <span className={`pill ${speechActive ? "pill-ok" : "pill-muted"}`}>{speechActive ? "Speech" : "Silence"}</span>
          <span className="pill pill-neutral">Faces: {faceCount}</span>
        </div>
      </section>

      <section className="event-stream">
        <div className="stream-header">
          <div>
            <p className="label">Events</p>
            <h2>Recent telemetry</h2>
          </div>
          <div className="actions">
            <button type="button" className="ghost" onClick={startStreams} disabled={cameraState === "starting"}>
              Restart capture
            </button>
            <button type="button" className="ghost" onClick={stopStreams}>
              Stop capture
            </button>
          </div>
        </div>
        {events.length === 0 ? (
          <p className="muted">No events yet. Move, speak, or add another person to trigger telemetry.</p>
        ) : (
          <ul className="event-list">
            {events
              .slice()
              .reverse()
              .map((event) => (
                <li key={event.id} className={`event-row severity-${event.severity}`}>
                  <div>
                    <p className="event-kind">{event.kind}</p>
                    <p className="event-message">{event.message}</p>
                  </div>
                  <p className="event-ts">{new Date(event.ts).toLocaleTimeString()}</p>
                </li>
              ))}
          </ul>
        )}
      </section>
    </main>
  );
}

export default App;
