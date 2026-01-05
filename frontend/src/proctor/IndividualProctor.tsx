import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { EventKind, IndividualFlags, ModelState, PostEventFn, StreamState } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const DETECT_SAMPLE_MS = 1600;
const FACE_STICKY_MS = 3000;

const gadgetLabels = ["cell phone", "laptop", "tv", "remote", "keyboard", "mouse", "tablet", "monitor"];

type Box = { x1: number; y1: number; x2: number; y2: number; label?: string; score?: number };

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

const YoloBadge = ({ state, label }: { state: ModelState; label: string }) => {
  const text = state === "ready" ? `${label} ready` : state === "loading" ? `Loading ${label}` : `${label} idle`;
  return <span className={`pill ${state === "ready" ? "pill-ok" : "pill-muted"}`}>{text}</span>;
};

const MediaPills = ({
  isCameraActive,
  speechActive,
  faceCount,
  faceApiState,
  gadgetYoloState,
}: {
  isCameraActive: boolean;
  speechActive: boolean;
  faceCount: number;
  faceApiState: ModelState;
  gadgetYoloState: ModelState;
}) => (
  <div className="pill-row">
    <span className={`pill ${isCameraActive ? "pill-ok" : "pill-warn"}`}>Camera {isCameraActive ? "on" : "off"}</span>
    <span className={`pill ${speechActive ? "pill-ok" : "pill-muted"}`}>{speechActive ? "Speech" : "Silence"}</span>
    <span className="pill pill-neutral">Faces: {faceCount}</span>
    <YoloBadge state={faceApiState} label="Face API" />
    <YoloBadge state={gadgetYoloState} label="Gadget YOLO" />
  </div>
);

const useOffscreenCanvas = () => {
  const ref = useRef<HTMLCanvasElement | null>(null);
  if (!ref.current && typeof document !== "undefined") {
    ref.current = document.createElement("canvas");
  }
  return ref;
};

export function IndividualProctor({ flags, postEvent }: { flags: IndividualFlags; postEvent: PostEventFn }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const faceRequestRef = useRef(false);
  const lastFacePositiveRef = useRef<number | null>(null);
  const lastFaceCountRef = useRef(0);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectionDataRef = useRef<{ sourceW: number; sourceH: number; faces: Box[]; gadgets: Box[] } | null>(null);
  const faceLandmarkerRef = useRef<any>(null);
  const gazeBusyRef = useRef(false);
  const gazeBaselineRef = useRef<{ yaw: number; pitch: number }>({ yaw: 0, pitch: 0 });
  const cropCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const [cameraState, setCameraState] = useState<StreamState>("idle");
  const [micState, setMicState] = useState<StreamState>("idle");
  const [faceApiState, setFaceApiState] = useState<ModelState>("idle");
  const [gadgetYoloState, setGadgetYoloState] = useState<ModelState>("idle");
  const [faceCount, setFaceCount] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [speechActive, setSpeechActive] = useState(false);
  const [overlayMessage, setOverlayMessage] = useState<string | null>("Requesting camera + mic access");
  const offscreenCanvasRef = useOffscreenCanvas();
  const [gadgetHit, setGadgetHit] = useState<string | null>(null);
  const [gazeAngles, setGazeAngles] = useState<{ yaw: number | null; pitch: number | null }>({ yaw: null, pitch: null });
  const [showAngles, setShowAngles] = useState(false);

  const isCameraActive = cameraState === "active";
  const isMicActive = micState === "active";

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
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
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
      if (!video || video.readyState < HTMLMediaElement.HAVE_ENOUGH_DATA) {
        setFaceCount((prev) => prev);
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
  }, [flags.audio, postEvent, speechActive]);
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const vision = await import("@mediapipe/tasks-vision");
        const fileset = await vision.FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
        );
        const landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          },
          runningMode: "IMAGE",
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });
        if (!cancelled) {
          faceLandmarkerRef.current = landmarker;
        }
      } catch (error) {
        console.error("Failed to load MediaPipe FaceLandmarker", error);
      }
    })();

    return () => {
      cancelled = true;
      if (faceLandmarkerRef.current?.close) {
        try {
          faceLandmarkerRef.current.close();
        } catch (error) {
          console.warn("Failed to close face landmarker", error);
        }
      }
      faceLandmarkerRef.current = null;
    };
  }, []);

  const renderFrame = useCallback((): HTMLCanvasElement | null => {
    const video = videoRef.current;
    const offscreen = offscreenCanvasRef.current;
    if (!video || !offscreen || video.readyState < HTMLMediaElement.HAVE_ENOUGH_DATA) {
      return null;
    }
    offscreen.width = 640;
    offscreen.height = Math.floor((video.videoHeight / video.videoWidth) * 640) || 360;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
    return offscreen;
  }, [offscreenCanvasRef]);

  const captureFrameBlob = useCallback(
    (frame: HTMLCanvasElement) =>
      new Promise<Blob>((resolve, reject) => {
        frame.toBlob((blob) => {
          if (blob) return resolve(blob);
          reject(new Error("Could not capture frame"));
        }, "image/jpeg", 0.8);
      }),
    []
  );

  const runFaceDetection = useCallback(
    async (frame: HTMLCanvasElement | null) => {
      if (!flags.faces || !frame || faceRequestRef.current) return;

      faceRequestRef.current = true;
      if (faceApiState === "idle") setFaceApiState("loading");

      try {
        const blob = await captureFrameBlob(frame);
        const form = new FormData();
        form.append("file", blob, "frame.jpg");

        const resp = await fetch(`${API_BASE}/detect/face?conf=0.12`, {
          method: "POST",
          body: form,
        });

        if (!resp.ok) {
          throw new Error(`Face API error ${resp.status}`);
        }

        const data: { count?: number; width?: number; height?: number; boxes?: Box[] } = await resp.json();
        setFaceApiState("ready");

        const detected = typeof data.count === "number" ? Math.max(0, data.count) : 0;
        const now = Date.now();

        detectionDataRef.current = {
          sourceW: frame.width,
          sourceH: frame.height,
          faces: Array.isArray(data.boxes)
            ? data.boxes.map((b) => ({ x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2, score: b.score, label: "face" }))
            : [],
          gadgets: detectionDataRef.current?.gadgets ?? [],
        };

        if (detected > 0) {
          lastFacePositiveRef.current = now;
          lastFaceCountRef.current = detected;
          setFaceCount(detected);
        } else {
          const lastTs = lastFacePositiveRef.current;
          const stale = !lastTs || now - lastTs > FACE_STICKY_MS;
          const fallback = stale ? 0 : lastFaceCountRef.current;
          setFaceCount(fallback);
        }

        const effectiveCount = detected > 0 ? detected : lastFaceCountRef.current;
        if (flags.faces && effectiveCount > 1) {
          postEvent("yolo-multi-face", "faces", toKind("faces"), "warn", `${effectiveCount} faces detected (backend)`);
        }
      } catch (error) {
        console.warn("Face API detection failed", error);
        setFaceApiState((prev) => (prev === "ready" ? "ready" : "error"));
      } finally {
        faceRequestRef.current = false;
      }
    },
    [captureFrameBlob, faceApiState, flags.faces, postEvent]
  );

  const runGadgetDetection = useCallback(
    async (frame: HTMLCanvasElement | null) => {
      if (!flags.gadgets || !frame) return;

      if (gadgetYoloState === "idle") setGadgetYoloState("loading");

      try {
        const blob = await captureFrameBlob(frame);
        const form = new FormData();
        form.append("file", blob, "frame.jpg");

        const resp = await fetch(`${API_BASE}/detect/gadget?conf=0.2`, {
          method: "POST",
          body: form,
        });

        if (!resp.ok) throw new Error(`Gadget API error ${resp.status}`);

        const data: { count?: number; width?: number; height?: number; boxes?: Box[] } = await resp.json();
        setGadgetYoloState("ready");

        const boxes: Box[] = Array.isArray(data.boxes)
          ? data.boxes.map((b) => ({ x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2, score: b.score, label: b.label }))
          : [];

        detectionDataRef.current = {
          sourceW: frame.width,
          sourceH: frame.height,
          faces: detectionDataRef.current?.faces ?? [],
          gadgets: boxes,
        };

        const gadget = boxes.find((b) => gadgetLabels.includes((b.label ?? "").toLowerCase()) && (b.score ?? 0) >= 0.35);
        if (gadget && gadget.label) {
          const label = gadget.label;
          setGadgetHit(`${label} (${Math.round((gadget.score ?? 0) * 100)}%)`);
          postEvent("yolo-gadget", "gadgets", toKind("gadgets"), "warn", `Gadget detected: ${label}`);
        }
      } catch (error) {
        console.warn("Gadget API detection failed", error);
        setGadgetYoloState((prev) => (prev === "ready" ? "ready" : "error"));
      }
    },
    [captureFrameBlob, flags.gadgets, gadgetYoloState, postEvent]
  );

  const pickPrimaryFace = useCallback(() => {
    const data = detectionDataRef.current;
    if (!data || !data.faces.length) return null;
    const cx = data.sourceW / 2;
    const cy = data.sourceH / 2;
    const best = data.faces.reduce<null | { face: Box; score: number; dist: number }>((acc, face) => {
      const score = face.score ?? 0;
      const fx = (face.x1 + face.x2) / 2;
      const fy = (face.y1 + face.y2) / 2;
      const dist = Math.hypot(fx - cx, fy - cy);
      if (!acc) return { face, score, dist };
      const betterScore = score > acc.score + 0.01;
      const closeTie = Math.abs(score - acc.score) <= 0.01 && dist < acc.dist;
      return betterScore || closeTie ? { face, score, dist } : acc;
    }, null);
    return best?.face ?? null;
  }, []);

  const runGazeEstimation = useCallback(async () => {
    if (!flags.gaze) return;
    if (gazeBusyRef.current) return;
    const landmarker = faceLandmarkerRef.current;
    const video = videoRef.current;
    const data = detectionDataRef.current;
    if (!landmarker || !video || !data) return;

    const primary = pickPrimaryFace();
    if (!primary) return;

    const base = offscreenCanvasRef.current;
    if (!base || !video.videoWidth || !video.videoHeight) return;
    base.width = data.sourceW;
    base.height = data.sourceH;
    const bctx = base.getContext("2d");
    if (!bctx) return;
    bctx.drawImage(video, 0, 0, base.width, base.height);

    const crop = cropCanvasRef.current ?? document.createElement("canvas");
    cropCanvasRef.current = crop;
    const w = Math.max(2, Math.round(primary.x2 - primary.x1));
    const h = Math.max(2, Math.round(primary.y2 - primary.y1));
    crop.width = w;
    crop.height = h;
    const cctx = crop.getContext("2d");
    if (!cctx) return;
    cctx.clearRect(0, 0, w, h);
    cctx.drawImage(base, primary.x1, primary.y1, w, h, 0, 0, w, h);
    const imageData = cctx.getImageData(0, 0, w, h);

    gazeBusyRef.current = true;
    try {
      const result = landmarker.detect(imageData);
      const landmarks = result.faceLandmarks?.[0];
      if (!landmarks || landmarks.length < 300) return;

      const leftEye = landmarks[33];
      const rightEye = landmarks[263];
      const nose = landmarks[1];
      const mouth = landmarks[13];

      const eyeCenter = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 };
      const rawYaw = nose.x - eyeCenter.x; // + means looking right from camera POV
      const rawPitch = nose.y - eyeCenter.y; // + means head lowered; eyes vs nose is more stable than mouth

      // Slowly learn a neutral baseline when the head is approximately centered
      const baseline = gazeBaselineRef.current;
      const inNeutralWindow = Math.abs(rawYaw) < 0.05 && Math.abs(rawPitch) < 0.05;
      if (inNeutralWindow) {
        baseline.yaw = lerp(baseline.yaw, rawYaw, 0.15);
        baseline.pitch = lerp(baseline.pitch, rawPitch, 0.15);
      }

      const yaw = rawYaw - baseline.yaw;
      const pitch = rawPitch - baseline.pitch;

      setGazeAngles({ yaw, pitch });

      const yawThreshold = 0.08;
      // Temporarily disable pitch-based flags

      if (yaw > yawThreshold) {
        postEvent("gaze-right", "gaze", toKind("gaze"), "warn", "Head turned right");
      } else if (yaw < -yawThreshold) {
        postEvent("gaze-left", "gaze", toKind("gaze"), "warn", "Head turned left");
      }

      // Pitch warnings disabled for now
    } catch (error) {
      console.warn("Gaze estimation failed", error);
    } finally {
      gazeBusyRef.current = false;
    }
  }, [flags.gaze, offscreenCanvasRef, pickPrimaryFace, postEvent]);

  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    const data = detectionDataRef.current;
    if (!canvas || !video) return;

    const width = video.clientWidth || video.videoWidth || 640;
    const height = video.clientHeight || video.videoHeight || 360;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (!flags.overlays || !data) return;

    const scaleX = width / data.sourceW;
    const scaleY = height / data.sourceH;

    const drawBoxes = (boxes: Box[], color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.font = "13px Space Grotesk, sans-serif";
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.textBaseline = "top";

      boxes.forEach((b) => {
        const x = b.x1 * scaleX;
        const y = b.y1 * scaleY;
        const w = (b.x2 - b.x1) * scaleX;
        const h = (b.y2 - b.y1) * scaleY;
        ctx.strokeRect(x, y, w, h);
        if (b.label) {
          const label = `${b.label}${b.score ? ` ${(b.score * 100).toFixed(0)}%` : ""}`;
          const metrics = ctx.measureText(label);
          const padX = 6;
          const padY = 3;
          const textW = metrics.width + padX * 2;
          const textH = 16 + padY;
          ctx.fillRect(x, y - textH < 0 ? y : y - textH, textW, textH);
          ctx.fillStyle = "#fff";
          ctx.fillText(label, x + padX, y - textH < 0 ? y + padY : y - textH + padY);
          ctx.fillStyle = "rgba(0,0,0,0.6)";
        }
      });
    };

    drawBoxes(data.faces, "#46d6a2");
    drawBoxes(data.gadgets, "#f5d97e");
  }, [flags.overlays]);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      const frame = renderFrame();
      if (!frame) return;

      // Face first (feeds primary face selection), then parallelize gaze + gadgets
      await runFaceDetection(frame);
      await Promise.all([runGazeEstimation(), runGadgetDetection(frame)]);
      drawOverlay();
    };

    const interval = window.setInterval(tick, DETECT_SAMPLE_MS);
    tick();

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [drawOverlay, renderFrame, runFaceDetection, runGadgetDetection, runGazeEstimation]);

  useEffect(() => {
    // Clear overlay when toggling off
    drawOverlay();
  }, [drawOverlay, flags.overlays]);

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
    ],
    [cameraState, faceCount, isCameraActive, isMicActive, micState, overlay, speechActive]
  );

  const formatAngle = useCallback((val: number | null, axis: "yaw" | "pitch") => {
    if (val == null) return "--";
    const deg = val * 90;
    const direction = axis === "yaw" ? (deg > 1 ? "right" : deg < -1 ? "left" : "centered") : deg > 1 ? "down" : deg < -1 ? "up" : "level";
    return `${deg.toFixed(1)}° ${direction}`;
  }, []);

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

      <section className="video-panel" style={{ position: "relative" }}>
        <video ref={videoRef} autoPlay playsInline muted className={isCameraActive ? "ready" : "dimmed"} />
        <canvas ref={overlayCanvasRef} className="overlay" />
        {overlay && (
          <div className="video-overlay">
            <p>{overlay}</p>
          </div>
        )}
        {showAngles && (
          <div
            className="angle-overlay"
            style={{
              position: "absolute",
              left: "12px",
              bottom: "12px",
              background: "rgba(0,0,0,0.6)",
              color: "#fff",
              padding: "10px 12px",
              borderRadius: "10px",
              fontSize: "13px",
              lineHeight: 1.35,
              backdropFilter: "blur(6px)",
              pointerEvents: "none",
              zIndex: 3,
            }}
          >
            <p>Yaw: {formatAngle(gazeAngles.yaw, "yaw")}</p>
            <p>Pitch: {formatAngle(gazeAngles.pitch, "pitch")}</p>
          </div>
        )}
        <MediaPills
          faceApiState={faceApiState}
          gadgetYoloState={gadgetYoloState}
          isCameraActive={isCameraActive}
          speechActive={speechActive}
          faceCount={faceCount}
        />
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
            <button type="button" className="ghost" onClick={() => setShowAngles((prev) => !prev)}>
              {showAngles ? "Hide angles" : "Show angles"}
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
