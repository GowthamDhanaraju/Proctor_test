from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
