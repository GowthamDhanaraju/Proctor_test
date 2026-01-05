from __future__ import annotations

import base64
import io
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel, Field
from supervision import Detections
from ultralytics import YOLO

app = FastAPI(title="Proctoring Prototype API")

# Allow the dev server to reach the API during local work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EventPayload(BaseModel):
    session_id: str = Field(..., description="Logical session id from the client")
    kind: Literal["video", "audio", "system"]
    severity: Literal["info", "warn", "error"] = "info"
    message: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EventResponse(EventPayload):
    id: int


_events: deque[EventResponse] = deque(maxlen=500)
_lock = Lock()
_id_counter = 0
_face_model = None
_face_model_lock = Lock()
_coco_model = None
_coco_model_lock = Lock()


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int | None = None
    label: str | None = None


class DetectResponse(BaseModel):
    count: int
    width: int
    height: int
    boxes: list[DetectionBox]


class FaceFramePayload(BaseModel):
    image_base64: str = Field(
        ..., description="Base64-encoded image data, optionally prefixed with a data URL header"
    )


class GadgetFramePayload(BaseModel):
    image_base64: str = Field(
        ..., description="Base64-encoded image data, optionally prefixed with a data URL header"
    )


@app.get("/")
def read_root() -> dict[str, str]:
    """Landing message for browsers hitting the API root."""
    return {
        "message": "Proctoring prototype backend is running.",
        "docs": "/docs",
    }


@app.get("/health")
def read_health() -> dict[str, str]:
    """Simple endpoint so we can confirm the API is running."""
    return {"status": "ok"}


@app.post("/events", response_model=EventResponse)
def record_event(payload: EventPayload) -> EventResponse:
    """Accepts lightweight telemetry from the frontend (audio/video/system)."""
    global _id_counter

    with _lock:
        _id_counter += 1
        record = EventResponse(id=_id_counter, **payload.model_dump())
        _events.append(record)
        return record


@app.get("/events", response_model=list[EventResponse])
def list_events(limit: int = 50) -> list[EventResponse]:
    """Returns the newest events for quick debugging in the prototype."""
    # Events are already capped by deque length; we slice without copying too much.
    limit = max(1, min(limit, len(_events)))
    return list(_events)[-limit:]


@app.get("/status")
def status_snapshot() -> dict[str, object]:
    """Aggregated view so the frontend can poll a basic status."""
    total = len(_events)
    if not total:
        return {"total": 0, "by_severity": {}, "by_kind": {}}

    by_severity: dict[str, int] = {"info": 0, "warn": 0, "error": 0}
    by_kind: dict[str, int] = {"video": 0, "audio": 0, "system": 0}
    for event in _events:
        by_severity[event.severity] = by_severity.get(event.severity, 0) + 1
        by_kind[event.kind] = by_kind.get(event.kind, 0) + 1

    latest_ts = _events[-1].ts

    return {
        "total": total,
        "by_severity": by_severity,
        "by_kind": by_kind,
        "latest": latest_ts,
    }


def _get_face_model() -> YOLO:
    global _face_model
    if _face_model is not None:
        return _face_model

    with _face_model_lock:
        if _face_model is not None:
            return _face_model
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        _face_model = YOLO(model_path)
        return _face_model


def _predict_faces(image: Image.Image, conf: float) -> DetectResponse:
    model = _get_face_model()
    try:
        results = model.predict(image, conf=conf, verbose=False)
    except Exception as exc:  # pragma: no cover - model inference issues
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    dets = Detections.from_ultralytics(results[0])
    boxes: list[DetectionBox] = []
    for (x1, y1, x2, y2), score, class_id in zip(dets.xyxy, dets.confidence, dets.class_id):
        boxes.append(
            DetectionBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(score),
                class_id=int(class_id) if class_id is not None else None,
                label="face",
            )
        )

    width, height = image.size
    return DetectResponse(count=len(boxes), width=width, height=height, boxes=boxes)


def _detect_from_bytes(content: bytes, conf: float) -> DetectResponse:
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:  # pragma: no cover - pillow parsing edge cases
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    return _predict_faces(image, conf)


def _get_coco_model() -> YOLO:
    global _coco_model
    if _coco_model is not None:
        return _coco_model

    with _coco_model_lock:
        if _coco_model is not None:
            return _coco_model
        _coco_model = YOLO("yolov8n.pt")
        return _coco_model


_ALLOWED_GADGET_LABELS = {
    "cell phone",
    "laptop",
    "tv",
    "remote",
    "keyboard",
    "mouse",
    "tablet",
    "monitor",
}


def _predict_gadgets(image: Image.Image, conf: float) -> DetectResponse:
    model = _get_coco_model()
    try:
        results = model.predict(image, conf=conf, verbose=False)
    except Exception as exc:  # pragma: no cover - model inference issues
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    dets = Detections.from_ultralytics(results[0])
    names = results[0].names if hasattr(results[0], "names") else {}
    boxes: list[DetectionBox] = []
    for (x1, y1, x2, y2), score, class_id in zip(dets.xyxy, dets.confidence, dets.class_id):
        label = names.get(int(class_id), None) if class_id is not None else None
        if label and label.lower() not in _ALLOWED_GADGET_LABELS:
            continue
        boxes.append(
            DetectionBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                score=float(score),
                class_id=int(class_id) if class_id is not None else None,
                label=label,
            )
        )

    width, height = image.size
    return DetectResponse(count=len(boxes), width=width, height=height, boxes=boxes)


def _detect_gadgets_from_bytes(content: bytes, conf: float) -> DetectResponse:
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:  # pragma: no cover - pillow parsing edge cases
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    return _predict_gadgets(image, conf)


@app.post("/detect/face", response_model=DetectResponse)
async def detect_face(file: UploadFile = File(...), conf: float = 0.1) -> DetectResponse:
    """Run YOLOv8 face detection on an uploaded image (multipart/form-data)."""

    if not file.content_type or not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    return _detect_from_bytes(content, conf)


@app.post("/detect/face/frame", response_model=DetectResponse)
async def detect_face_frame(payload: FaceFramePayload, conf: float = 0.1) -> DetectResponse:
    """Run YOLOv8 face detection on a base64 frame payload (stream-friendly)."""

    try:
        b64_data = payload.image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        content = base64.b64decode(b64_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}") from exc

    return _detect_from_bytes(content, conf)


@app.post("/detect/gadget", response_model=DetectResponse)
async def detect_gadget(file: UploadFile = File(...), conf: float = 0.25) -> DetectResponse:
    """Detect handheld and screen devices (multipart/form-data)."""

    if not file.content_type or not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    return _detect_gadgets_from_bytes(content, conf)


@app.post("/detect/gadget/frame", response_model=DetectResponse)
async def detect_gadget_frame(payload: GadgetFramePayload, conf: float = 0.25) -> DetectResponse:
    """Detect handheld and screen devices from a base64 frame."""

    try:
        b64_data = payload.image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        content = base64.b64decode(b64_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}") from exc

    return _detect_gadgets_from_bytes(content, conf)
