import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import {
  FACE_LANDMARKER_MODEL,
  MEDIAPIPE_WASM_BASE,
  MIN_FACE_AREA,
  computeFaceArea,
  computeFaceMetrics,
  type FaceLandmarks,
} from "./utils";
import type { ModelState, PostEventFn, StreamState, TeamFlags } from "./types";

const GAZE_YAW_THRESHOLD = 40;
const NO_FACE_GRACE_MS = 5000;

const drawLandmarks = (canvas: HTMLCanvasElement | null, faces?: FaceLandmarks[]) => {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!faces || faces.length === 0) return;
  const palette = ["rgba(245, 217, 126, 0.95)", "rgba(70, 214, 162, 0.9)", "rgba(255, 179, 71, 0.9)"];
  faces.forEach((face, idx) => {
    ctx.fillStyle = palette[idx % palette.length];
    ctx.strokeStyle = "rgba(12, 18, 33, 0.8)";
    ctx.lineWidth = 1;
    face.forEach((landmark) => {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  });
};

export function TeamProctor({
  teamLimit,
  flags,
  postEvent,
}: {
  teamLimit: number;
  flags: TeamFlags;
  postEvent: PostEventFn;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [cameraState, setCameraState] = useState<StreamState>("idle");
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [faceCount, setFaceCount] = useState(0);
  const [primaryYaw, setPrimaryYaw] = useState<number | null>(null);
  const [overlayMessage, setOverlayMessage] = useState<string | null>("Requesting camera access");
  const lastFaceTsRef = useRef<number>(Date.now());

  const isCameraActive = cameraState === "active";

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

  const startStream = useCallback(async () => {
    if (cameraState === "starting" || cameraState === "active") return;
    setCameraState("starting");
    setOverlayMessage("Requesting camera access");
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920, max: 1920 },
          height: { ideal: 1080, max: 1080 },
          frameRate: { ideal: 24, max: 30 },
        },
        audio: false,
      });
      streamRef.current = mediaStream;
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.muted = true;
        await videoRef.current.play();
      }
      setCameraState("active");
      setOverlayMessage(null);
      postEvent("team-media-on", "system", "system", "info", "Camera granted for team mode");
    } catch (error) {
      console.error("Unable to start camera", error);
      setCameraState("error");
      setOverlayMessage("Permissions denied or unavailable");
      postEvent("team-media-error", "system", "system", "error", "Failed to start camera in team mode");
    }
  }, [cameraState, postEvent]);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState("idle");
    setOverlayMessage("Camera stopped");
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
          numFaces: 8,
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
    startStream();

    return () => {
      cancelled = true;
      animationFrameRef.current && cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
      faceLandmarkerRef.current?.close();
      faceLandmarkerRef.current = null;
      stopStream();
    };
  }, [startStream, stopStream]);

  useEffect(() => {
    let cancelled = false;

    const analyze = () => {
      if (cancelled) return;
      const video = videoRef.current;
      const landmarker = faceLandmarkerRef.current;
      if (video && landmarker && modelState === "ready" && video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA) {
        syncCanvasSize();
        const result = landmarker.detectForVideo(video, performance.now());
        const faces = (result.faceLandmarks ?? []) as FaceLandmarks[];
        drawLandmarks(canvasRef.current, faces);

        const count = faces.length;
        setFaceCount(count);
        if (count > 0) {
          lastFaceTsRef.current = Date.now();
        }

        if (flags.capacity && count > teamLimit) {
          postEvent(
            "team-overflow",
            "capacity",
            "video",
            "warn",
            `${count} people detected (limit ${teamLimit})`
          );
        }
        if (flags.presence && count === 0 && Date.now() - lastFaceTsRef.current > NO_FACE_GRACE_MS) {
          postEvent("team-empty", "presence", "video", "warn", "No teammates detected");
        }

        const primaryFace = faces[0];
        if (primaryFace) {
          const area = computeFaceArea(primaryFace);
          if (area < MIN_FACE_AREA) {
            setOverlayMessage("Faces too small — bring camera closer");
          } else {
            setOverlayMessage(null);
          }
          const metrics = computeFaceMetrics(primaryFace, video.videoWidth);
          if (flags.gaze && metrics && Math.abs(metrics.yawDeg) > GAZE_YAW_THRESHOLD) {
            postEvent("team-yaw", "gaze", "video", "warn", "Primary viewer looking away");
          }
          setPrimaryYaw(metrics?.yawDeg ?? null);
        } else {
          setPrimaryYaw(null);
        }
      } else {
        setFaceCount(0);
        drawLandmarks(canvasRef.current);
        if (flags.presence && Date.now() - lastFaceTsRef.current > NO_FACE_GRACE_MS) {
          setOverlayMessage("No face detected");
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
  }, [flags.capacity, flags.gaze, flags.presence, modelState, postEvent, syncCanvasSize, teamLimit]);

  const statusCards = useMemo(
    () => [
      {
        label: "Camera",
        value: isCameraActive ? "Active" : cameraState === "starting" ? "Requesting" : "Off",
        detail: overlayMessage ?? "Streaming stays on device",
        tone: isCameraActive ? "ok" : "warn",
      },
      {
        label: "Headcount",
        value: `${faceCount} / ${teamLimit}`,
        detail: faceCount > teamLimit ? "Over limit" : faceCount === 0 ? "No faces" : "Within limit",
        tone: faceCount > 0 && faceCount <= teamLimit ? "ok" : "warn",
      },
      {
        label: "Primary focus",
        value: primaryYaw !== null ? `${primaryYaw.toFixed(1)}° yaw` : "--",
        detail: primaryYaw !== null ? "Yaw from nose vs eyes" : "Waiting for a face",
        tone: primaryYaw !== null && Math.abs(primaryYaw) < GAZE_YAW_THRESHOLD ? "ok" : "warn",
      },
    ],
    [cameraState, faceCount, isCameraActive, overlayMessage, primaryYaw, teamLimit]
  );

  return (
    <div className="mode-panel">
      <section className="status-grid">
        {statusCards.map((card) => (
          <div key={card.label} className={`status-card ${card.tone}`}>
            <p className="label">{card.label}</p>
            <p className="value">{card.value}</p>
            <small>{card.detail}</small>
          </div>
        ))}
      </section>

      <section className="video-panel">
        <video ref={videoRef} autoPlay playsInline muted className={isCameraActive ? "ready" : "dimmed"} />
        <canvas ref={canvasRef} className="overlay" />
        {overlayMessage && (
          <div className="video-overlay">
            <p>{overlayMessage}</p>
          </div>
        )}
        <div className="pill-row">
          <span className={`pill ${isCameraActive ? "pill-ok" : "pill-warn"}`}>
            Camera {isCameraActive ? "on" : "off"}
          </span>
          <span className="pill pill-neutral">Faces: {faceCount}</span>
          <span className="pill pill-neutral">Limit: {teamLimit}</span>
        </div>
      </section>

      <section className="event-stream">
        <div className="stream-header">
          <div>
            <p className="label">Team mode</p>
            <h2>Team proctoring</h2>
          </div>
          <div className="actions">
            <button type="button" className="ghost" onClick={startStream} disabled={cameraState === "starting"}>
              Restart camera
            </button>
            <button type="button" className="ghost" onClick={stopStream}>
              Stop camera
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

export default TeamProctor;
