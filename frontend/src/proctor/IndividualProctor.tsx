import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import {
  FACE_LANDMARKER_MODEL,
  MEDIAPIPE_WASM_BASE,
  MIN_FACE_AREA,
  clamp,
  computeFaceArea,
  computeFaceMetrics,
  type FaceLandmarks,
} from "./utils";
import type { EventKind, IndividualFlags, ModelState, PostEventFn, StreamState } from "./types";

const GAZE_YAW_THRESHOLD = 35;
const YOLO_SAMPLE_MS = 1800;

const gadgetLabels = ["cell phone", "laptop", "tv", "remote", "keyboard", "mouse", "tablet", "monitor"];

const overlayForState = (state: StreamState, fallback: string | null) => {
  if (state === "starting") return "Requesting camera + mic access";
  if (state === "error") return "Permissions denied or unavailable";
  return fallback;
};

const toKind = (category: "audio" | "gaze" | "faces" | "gadgets" | "system"): EventKind => {
  if (category === "audio") return "audio";
  if (category === "system") return "system";
  return "video";
};

const YoloBadge = ({ state }: { state: ModelState }) => {
  const text = state === "ready" ? "YOLO active" : state === "loading" ? "Loading YOLO" : "YOLO idle";
  return <span className={`pill ${state === "ready" ? "pill-ok" : "pill-muted"}`}>{text}</span>;
};

const MediaPills = ({
  isCameraActive,
  speechActive,
  faceCount,
  yoloState,
}: {
  isCameraActive: boolean;
  speechActive: boolean;
  faceCount: number;
  yoloState: ModelState;
}) => (
  <div className="pill-row">
    <span className={`pill ${isCameraActive ? "pill-ok" : "pill-warn"}`}>Camera {isCameraActive ? "on" : "off"}</span>
    <span className={`pill ${speechActive ? "pill-ok" : "pill-muted"}`}>{speechActive ? "Speech" : "Silence"}</span>
    <span className="pill pill-neutral">Faces: {faceCount}</span>
    <YoloBadge state={yoloState} />
  </div>
);

const drawLandmarks = (canvas: HTMLCanvasElement | null, result?: FaceLandmarkerResult) => {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const primaryFace = result?.faceLandmarks?.[0];
  if (!primaryFace) return;

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
};

const useOffscreenCanvas = () => {
  const ref = useRef<HTMLCanvasElement | null>(null);
  if (!ref.current && typeof document !== "undefined") {
    ref.current = document.createElement("canvas");
  }
  return ref;
};

export function IndividualProctor({ flags, postEvent }: { flags: IndividualFlags; postEvent: PostEventFn }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const yoloRef = useRef<((input: unknown) => Promise<any[]>) | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const [cameraState, setCameraState] = useState<StreamState>("idle");
  const [micState, setMicState] = useState<StreamState>("idle");
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [yoloState, setYoloState] = useState<ModelState>("idle");
  const [faceCount, setFaceCount] = useState(0);
  const [primaryYaw, setPrimaryYaw] = useState<number | null>(null);
  const [primaryDistance, setPrimaryDistance] = useState<number | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [speechActive, setSpeechActive] = useState(false);
  const [overlayMessage, setOverlayMessage] = useState<string | null>("Requesting camera + mic access");
  const offscreenCanvasRef = useOffscreenCanvas();
  const [gadgetHit, setGadgetHit] = useState<string | null>(null);

  const isCameraActive = cameraState === "active";
  const isMicActive = micState === "active";

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

  const startStreams = useCallback(async () => {
    if (cameraState === "starting" || cameraState === "active") return;
    setCameraState("starting");
    setMicState("starting");
    setOverlayMessage("Requesting camera + mic access");

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920, max: 1920 },
          height: { ideal: 1080, max: 1080 },
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
      postEvent("media-on", "system", toKind("system"), "info", "Camera and mic granted");
    } catch (error) {
      console.error("Unable to start streams", error);
      setCameraState("error");
      setMicState("error");
      setOverlayMessage("Permissions denied or unavailable");
      postEvent("media-error", "system", toKind("system"), "error", "Failed to start camera or mic");
    }
  }, [cameraState, postEvent]);

  const stopStreams = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    audioContextRef.current?.close();
    audioContextRef.current = null;
    analyserRef.current = null;
    setCameraState("idle");
    setMicState("idle");
    setOverlayMessage("Camera and mic stopped");
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadFaceLandmarker() {
      setModelState("loading");
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_BASE);
        const instance = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL },
          runningMode: "VIDEO",
          numFaces: 3,
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
      stopStreams();
    };
  }, [stopStreams]);

  useEffect(() => {
    startStreams();
  }, [startStreams]);

  useEffect(() => {
    let cancelled = false;
    const TIME_DOMAIN_SIZE = 1024;
    const dataArray = new Uint8Array(TIME_DOMAIN_SIZE);
    const speechThresholdOn = 0.05;
    const speechThresholdOff = 0.025;

    const analyze = () => {
      if (cancelled) return;

      const video = videoRef.current;
      const landmarker = faceLandmarkerRef.current;

      if (video && landmarker && modelState === "ready" && video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA) {
        syncCanvasSize();
        const result = landmarker.detectForVideo(video, performance.now());
        drawLandmarks(canvasRef.current, result);

        const count = result.faceLandmarks?.length ?? 0;
        setFaceCount(count);
        if (flags.faces && count > 1) {
          postEvent("multi-face", "faces", toKind("faces"), "warn", "Multiple faces detected");
        }
        const primaryFace = result.faceLandmarks?.[0];
        if (primaryFace) {
          const area = computeFaceArea(primaryFace as FaceLandmarks);
          if (area < MIN_FACE_AREA) {
            setOverlayMessage("Face is too small — move closer or increase brightness");
            if (flags.faces) {
              postEvent("face-small", "faces", toKind("faces"), "warn", "Face too small for reliable landmarks");
            }
          } else {
            setOverlayMessage(null);
          }
          const metrics = computeFaceMetrics(primaryFace as FaceLandmarks, video.videoWidth);
          if (metrics) {
            setPrimaryYaw(metrics.yawDeg);
            setPrimaryDistance(metrics.distanceCm);
            if (flags.gaze && Math.abs(metrics.yawDeg) > GAZE_YAW_THRESHOLD) {
              postEvent("yaw", "gaze", toKind("gaze"), "warn", "Viewer looking away from screen");
            }
          }
        } else {
          setPrimaryYaw(null);
          setPrimaryDistance(null);
          setOverlayMessage("No face detected");
          if (flags.faces) {
            postEvent("no-face", "faces", toKind("faces"), "warn", "No face detected");
          }
        }
      } else {
        setFaceCount(0);
        drawLandmarks(canvasRef.current);
        setOverlayMessage("No face detected");
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

        if (flags.audio) {
          if (!speechActive && rms > speechThresholdOn) {
            setSpeechActive(true);
            postEvent("speech-on", "audio", toKind("audio"), "info", "Speech detected");
          } else if (speechActive && rms < speechThresholdOff) {
            setSpeechActive(false);
            postEvent("speech-off", "audio", toKind("audio"), "info", "Silence detected");
          }
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
  }, [flags.audio, flags.faces, flags.gaze, modelState, postEvent, speechActive, syncCanvasSize]);

  useEffect(() => {
    let cancelled = false;

    async function loadYolo() {
      setYoloState("loading");
      try {
        const { pipeline } = await import("@xenova/transformers");
        const detector = await pipeline("object-detection", "Xenova/yolov8n");
        if (cancelled) return;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        yoloRef.current = detector as any;
        setYoloState("ready");
      } catch (error) {
        console.error("Failed to load YOLO", error);
        if (!cancelled) setYoloState("error");
      }
    }
    loadYolo();

    const runDetector = async () => {
      const detector = yoloRef.current;
      const video = videoRef.current;
      const offscreen = offscreenCanvasRef.current;
      if (!detector || !video || !offscreen || video.readyState < HTMLMediaElement.HAVE_ENOUGH_DATA) {
        return;
      }
      offscreen.width = 640;
      offscreen.height = Math.floor((video.videoHeight / video.videoWidth) * 640) || 360;
      const ctx = offscreen.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
      try {
        const detections = await detector(offscreen);
        const persons = detections.filter((d) => d.label === "person" && d.score >= 0.45).length;
        const gadget = detections.find((d) => gadgetLabels.includes(d.label) && d.score >= 0.35);

        if (flags.faces && persons > 1) {
          postEvent("yolo-multi-person", "faces", toKind("faces"), "warn", `${persons} people detected in frame (YOLO)`);
        }
        if (flags.gadgets && gadget) {
          const label = gadget.label as string;
          setGadgetHit(`${label} (${Math.round(gadget.score * 100)}%)`);
          postEvent(
            "yolo-gadget",
            "gadgets",
            toKind("gadgets"),
            "warn",
            `Gadget detected: ${label}`
          );
        }
      } catch (error) {
        console.warn("YOLO detection failed", error);
      }
    };

    const interval = window.setInterval(runDetector, YOLO_SAMPLE_MS);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [flags.faces, flags.gadgets, offscreenCanvasRef, postEvent]);

  const overlay = overlayForState(cameraState, overlayMessage);

  const statusCards = useMemo(
    () => [
      {
        label: "Camera",
        value: isCameraActive ? "Active" : cameraState === "starting" ? "Requesting" : "Off",
        detail: overlay ?? "Streaming stays on device",
        tone: isCameraActive ? "ok" : "warn",
      },
      {
        label: "Microphone",
        value: isMicActive ? "Active" : micState === "starting" ? "Requesting" : "Off",
        detail: speechActive ? "Speech detected" : "Silence",
        tone: isMicActive ? "ok" : "warn",
      },
      {
        label: "Faces in frame",
        value: faceCount.toString(),
        detail: faceCount > 1 ? "Multiple people present" : faceCount === 1 ? "Single face" : "No face",
        tone: faceCount === 1 ? "ok" : "warn",
      },
      {
        label: "Gaze + distance",
        value: `${primaryYaw !== null ? `${primaryYaw.toFixed(1)}°` : "--"} | ${primaryDistance !== null ? `${primaryDistance.toFixed(0)} cm` : "--"}`,
        detail: primaryYaw !== null ? "Yaw from nose vs eyes" : "Waiting for a face",
        tone: primaryYaw !== null ? "ok" : "warn",
      },
    ],
    [cameraState, faceCount, isCameraActive, isMicActive, micState, overlay, primaryDistance, primaryYaw, speechActive]
  );

  return (
    <div className="mode-panel">
      <section className="status-grid">
        {statusCards.map((card) => (
          <div key={card.label} className={`status-card ${card.tone}`}>
            <p className="label">{card.label}</p>
            <p className="value">{card.value}</p>
            {card.label === "Microphone" && (
              <div className="meter">
                <div className="meter-fill" style={{ width: `${audioLevel * 100}%` }} />
              </div>
            )}
            <small>{card.detail}</small>
          </div>
        ))}
      </section>

      <section className="video-panel">
        <video ref={videoRef} autoPlay playsInline muted className={isCameraActive ? "ready" : "dimmed"} />
        <canvas ref={canvasRef} className="overlay" />
        {overlay && (
          <div className="video-overlay">
            <p>{overlay}</p>
          </div>
        )}
        <MediaPills yoloState={yoloState} isCameraActive={isCameraActive} speechActive={speechActive} faceCount={faceCount} />
      </section>

      <section className="event-stream">
        <div className="stream-header">
          <div>
            <p className="label">Individual mode</p>
            <h2>Solo proctoring</h2>
            {gadgetHit && <p className="muted">Last gadget hit: {gadgetHit}</p>}
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
      </section>
    </div>
  );
}

export default IndividualProctor;
