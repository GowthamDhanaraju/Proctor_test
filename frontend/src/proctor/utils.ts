import type { FaceLandmarkerResult } from "@mediapipe/tasks-vision";

export const MEDIAPIPE_WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm";
export const FACE_LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

export type FaceLandmarks = FaceLandmarkerResult["faceLandmarks"][number];
export type FaceMetrics = {
  distanceCm: number;
  yawDeg: number;
};

const DEFAULT_IPD_CM = 6.3;
const BASELINE_VIDEO_WIDTH = 1280;
const BASELINE_FOCAL_PX = 750;
const HEAD_YAW_MAX_DEG = 90;
const RAD2DEG = 180 / Math.PI;
const YAW_OFFSET_SCALE = 0.55;
export const DISTANCE_RANGE_CM = { min: 10, max: 150 } as const;
export const MIN_FACE_AREA = 0.015;

const LANDMARK_INDEX = {
  rightEyeOuter: 33,
  leftEyeOuter: 263,
  noseTip: 1,
} as const;

export const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

export const distance2D = (a: { x: number; y: number }, b: { x: number; y: number }) => {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
};

export const getLandmark = (landmarks: FaceLandmarks, index: number) => landmarks[index] ?? null;

export const computeFaceMetrics = (landmarks: FaceLandmarks, videoWidth: number): FaceMetrics | null => {
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

export const computeFaceArea = (landmarks: FaceLandmarks): number => {
  let minX = 1;
  let maxX = 0;
  let minY = 1;
  let maxY = 0;
  landmarks.forEach((lm) => {
    minX = Math.min(minX, lm.x);
    maxX = Math.max(maxX, lm.x);
    minY = Math.min(minY, lm.y);
    maxY = Math.max(maxY, lm.y);
  });
  const width = Math.max(0, maxX - minX);
  const height = Math.max(0, maxY - minY);
  return width * height;
};
