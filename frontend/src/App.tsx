import { useCallback, useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult
} from "@mediapipe/tasks-vision";
import "./global.css";

type MediaState = "idle" | "starting" | "active" | "error";
type ModelState = "idle" | "loading" | "ready" | "error";
type QualityPreset = "low" | "medium" | "high";
type FaceMetrics = {
  distanceCm: number;
  yawDeg: number;
  yawDirection: "left" | "right" | "center";
  closeness: number;
};
const MEDIAPIPE_WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm";
const FACE_LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const QUALITY_CONFIG: Record<QualityPreset, { label: string; description: string; constraints: MediaTrackConstraints }> = {
  low: {
    label: "360p / 10fps",
    description: "Best for low bandwidth",
    constraints: {
      width: { ideal: 640, max: 640 },
      height: { ideal: 360, max: 360 },
      frameRate: { ideal: 10, max: 10 }
    }
  },
  medium: {
    label: "720p / 24fps",
    description: "Balanced quality",
    constraints: {
      width: { ideal: 1280, max: 1280 },
      height: { ideal: 720, max: 720 },
      frameRate: { ideal: 24, max: 30 }
    }
  },
  high: {
    label: "1080p / 30fps",
    description: "Maximum clarity",
    constraints: {
      width: { ideal: 1920, max: 1920 },
      height: { ideal: 1080, max: 1080 },
      frameRate: { ideal: 30, max: 30 }
    }
  }
};
const DEFAULT_IPD_CM = 6.3;
const BASELINE_VIDEO_WIDTH = 1280;
const BASELINE_FOCAL_PX = 750;
const HEAD_YAW_MAX_DEG = 90;
const RAD2DEG = 180 / Math.PI;
const YAW_OFFSET_SCALE = 0.55;
const METRICS_UPDATE_INTERVAL_MS = 80;
const DISTANCE_RANGE_CM = { min: 10, max: 150 } as const;
const LANDMARK_INDEX = {
  rightEyeOuter: 33,
  leftEyeOuter: 263,
  noseTip: 1
} as const;

type FaceLandmarks = FaceLandmarkerResult["faceLandmarks"][number];

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
  const closeness = clamp(
    1 - (distanceCm - DISTANCE_RANGE_CM.min) / (DISTANCE_RANGE_CM.max - DISTANCE_RANGE_CM.min),
    0,
    1
  );

  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const noseOffset = noseTip.x - eyeMidX;
  const yawRatio = clamp(noseOffset / (interpupilNorm * YAW_OFFSET_SCALE), -1, 1);
  const yawDegRaw = clamp(-Math.asin(yawRatio) * RAD2DEG, -HEAD_YAW_MAX_DEG, HEAD_YAW_MAX_DEG);
  const yawDirection = yawDegRaw > 3 ? "right" : yawDegRaw < -3 ? "left" : "center";

  return {
    distanceCm,
    yawDeg: yawDegRaw,
    yawDirection,
    closeness
  };
};

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const metricsUpdateRef = useRef(0);
  const mediaStateRef = useRef<MediaState>("idle");
  const [mediaState, setMediaState] = useState<MediaState>("idle");
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [quality, setQuality] = useState<QualityPreset>("low");
  const [faceMetrics, setFaceMetrics] = useState<FaceMetrics | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  useEffect(() => {
    mediaStateRef.current = mediaState;
  }, [mediaState]);

  useEffect(() => {
    let isCancelled = false;

    async function loadFaceLandmarker() {
      setModelState("loading");
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_BASE);
        const faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: FACE_LANDMARKER_MODEL
          },
          runningMode: "VIDEO",
          outputFacialTransformationMatrixes: false,
          numFaces: 1
        });

        if (isCancelled) {
          faceLandmarker.close();
          return;
        }

        faceLandmarkerRef.current = faceLandmarker;
        setModelState("ready");
      } catch (error) {
        console.error("Unable to load MediaPipe face landmarker", error);
        if (!isCancelled) {
          setModelState("error");
        }
      }
    }

    loadFaceLandmarker();

    return () => {
      isCancelled = true;
      animationFrameRef.current && cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
      faceLandmarkerRef.current?.close();
      faceLandmarkerRef.current = null;
    };
  }, []);

  const syncCanvasSize = useCallback(() => {
    const video = videoRef.current;
    const canvas = overlayCanvasRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
      return;
    }
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
  }, []);

  const drawLandmarks = useCallback((result?: FaceLandmarkerResult) => {
    const canvas = overlayCanvasRef.current;
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

    ctx.fillStyle = "rgba(249, 115, 22, 0.9)";
    ctx.strokeStyle = "rgba(15, 23, 42, 0.8)";
    ctx.lineWidth = 1;

    primaryFace.forEach((landmark) => {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }, []);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    mediaStateRef.current = "idle";
    setMediaState("idle");
    setErrorMessage("");
    setFaceMetrics(null);
    metricsUpdateRef.current = 0;
  }, []);

  const startStream = useCallback(async () => {
    if (mediaStateRef.current === "starting" || mediaStateRef.current === "active") {
      return;
    }
    mediaStateRef.current = "starting";
    setMediaState("starting");
    setErrorMessage("");

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: QUALITY_CONFIG[quality].constraints,
        audio: false
      });

      streamRef.current = mediaStream;
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        try {
          await videoRef.current.play();
          syncCanvasSize();
        } catch (playError) {
          console.warn("Video element was unable to autoplay", playError);
        }
      }
      mediaStateRef.current = "active";
      setMediaState("active");
    } catch (error) {
      console.error("Unable to start webcam", error);
      mediaStateRef.current = "error";
      setMediaState("error");
      setErrorMessage(
        error instanceof DOMException
          ? error.message
          : "Unable to access webcam. Please check permissions and try again."
      );
    }
  }, [quality, syncCanvasSize]);

  useEffect(() => {
    startStream();
    return () => {
      stopStream();
    };
  }, [startStream, stopStream]);

  useEffect(() => {
    if (
      mediaState !== "active" ||
      modelState !== "ready" ||
      !videoRef.current ||
      !faceLandmarkerRef.current
    ) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      drawLandmarks();
      setFaceMetrics(null);
      metricsUpdateRef.current = 0;
      return;
    }

    let isCancelled = false;

    const renderResult = () => {
      if (isCancelled) {
        return;
      }
      const video = videoRef.current;
      const landmarker = faceLandmarkerRef.current;
      if (!video || !landmarker || video.readyState < HTMLMediaElement.HAVE_ENOUGH_DATA) {
        animationFrameRef.current = requestAnimationFrame(renderResult);
        return;
      }

      syncCanvasSize();

      const result = landmarker.detectForVideo(video, performance.now());
      drawLandmarks(result);
      const primaryFace = result.faceLandmarks?.[0];
      if (primaryFace) {
        const metrics = computeFaceMetrics(primaryFace, video.videoWidth);
        const now = performance.now();
        if (metrics && now - metricsUpdateRef.current > METRICS_UPDATE_INTERVAL_MS) {
          setFaceMetrics(metrics);
          metricsUpdateRef.current = now;
        }
      } else {
        setFaceMetrics(null);
      }

      animationFrameRef.current = requestAnimationFrame(renderResult);
    };

    animationFrameRef.current = requestAnimationFrame(renderResult);

    return () => {
      isCancelled = true;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [mediaState, modelState, drawLandmarks, syncCanvasSize]);

  const isActive = mediaState === "active";
  const isIdle = mediaState === "idle";
  const overlayMessage =
    mediaState === "error"
      ? errorMessage
      : modelState !== "ready"
        ? modelState === "error"
          ? "Face landmark model failed to load"
          : "Loading face landmark model..."
        : "Waiting for webcam permission";

  return (
    <main className="app-shell">
      <header>
        <p className="eyebrow">FastAPI + React demo</p>
        <h1>Local Webcam Preview</h1>
        <p className="lede">
          Start your webcam and preview the feed locally. No video stream ever
          leaves your browser.
        </p>
      </header>

      <section className="video-panel">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onLoadedMetadata={syncCanvasSize}
          className={isActive ? "ready" : "dimmed"}
        />
        <canvas ref={overlayCanvasRef} className="landmarks-layer" />
        {!isActive && (
          <div className="video-overlay">
            <p>{overlayMessage}</p>
          </div>
        )}
      </section>

      <section className="controls">
        <button type="button" onClick={isActive ? stopStream : startStream} disabled={mediaState === "starting"}>
          {isActive ? "Stop" : "Start"} camera
        </button>
        <div className="quality-toggle" role="group" aria-label="Quality presets">
          {Object.entries(QUALITY_CONFIG).map(([key, preset]) => {
            const typedKey = key as QualityPreset;
            const isSelected = quality === typedKey;
            return (
              <button
                key={key}
                type="button"
                className={isSelected ? "selected" : "ghost"}
                onClick={() => setQuality(typedKey)}
                disabled={mediaState === "starting" && !isSelected}
              >
                <span>{preset.label}</span>
                <small>{preset.description}</small>
              </button>
            );
          })}
        </div>
        {!isIdle && (
          <p className="status">
            {mediaState === "starting" && "Requesting permission..."}
            {isActive && "Streaming from your device."}
            {mediaState === "error" && errorMessage}
          </p>
        )}
      </section>

      <section className="telemetry">
        <div className="metric-card">
          <p className="label">Estimated distance</p>
          <p className="value">{faceMetrics ? `~${faceMetrics.distanceCm.toFixed(0)} cm` : "--"}</p>
          <div className="gauge" aria-hidden="true">
            <div className="fill" style={{ width: `${(faceMetrics?.closeness ?? 0) * 100}%` }} />
          </div>
          <small>Assumes average 6.3 cm interpupillary distance.</small>
        </div>
        <div className="metric-card">
          <p className="label">Head rotation</p>
          <p className="value">{faceMetrics ? `${faceMetrics.yawDeg.toFixed(1)}°` : "--"}</p>
          <div className="radar-track" aria-hidden="true">
            <div
              className="radar-indicator"
              style={{
                left: `${clamp(((faceMetrics?.yawDeg ?? 0) / HEAD_YAW_MAX_DEG + 1) * 50, 0, 100)}%`
              }}
            />
          </div>
          <small>
            {faceMetrics
              ? faceMetrics.yawDirection === "center"
                ? "Looking straight"
                : `Turning ${faceMetrics.yawDirection}`
              : "No face detected"}
          </small>
        </div>
      </section>
    </main>
  );
}

export default App;
