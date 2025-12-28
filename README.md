# Proctoring Prototype (Video + Audio)

Lightweight prototype to exercise the proctoring surface area with only camera/mic signals. The browser runs on-device heuristics (face presence, gaze drift, speech activity) and ships telemetry to a minimal FastAPI backend. No video/audio leaves the browser beyond the short JSON events.

## What’s Implemented
- Single camera capture with MediaPipe Face Landmarker (2 faces max) to flag presence and rough head yaw/distance.
- Audio RMS-based VAD badge (speech/no speech) using the Web Audio API.
- Event stream sent to the backend for: permission changes, multiple faces, gaze drift, and speech start/stop.
- Dashboard UI showing camera/mic state, face count, yaw + distance estimate, audio meter, and recent events.

## Quick Start

### Backend (FastAPI)
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
The API lives at `http://127.0.0.1:8000`.

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
Open the printed Vite URL (usually `http://localhost:5173`) and grant camera + mic permissions when prompted.

## API Surface
- `GET /health` — liveness check.
- `POST /events` — ingest telemetry `{ session_id, kind: "video"|"audio"|"system", severity: "info"|"warn"|"error", message, ts }`.
- `GET /events?limit=50` — newest events (in-memory ring buffer).
- `GET /status` — counts by severity/kind + latest timestamp.

## Notes and Next Steps
- Everything is on-device; swap in heavier models (YOLO/InsightFace/pyannote/OpenVINO gaze) server-side later.
- If you host the frontend elsewhere, update CORS in `backend/app/main.py` and set `VITE_API_BASE` in the frontend to point at the backend.
- The heuristics are intentionally simple (RMS-based VAD, yaw from landmarks). Replace with production detectors as they’re integrated.
