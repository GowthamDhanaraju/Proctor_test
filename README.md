# Webcam Preview (FastAPI + React)

This project contains a FastAPI backend scaffold plus a React (Vite) frontend that captures your local webcam feed and displays it in the browser. The video stream never leaves the client.

## Requirements

- Python 3.11+
- Node.js 18+
- npm (bundled with Node.js)

## Backend (FastAPI)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. A `/health` endpoint is included for quick checks.

## Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open the printed Vite URL (defaults to `http://localhost:5173`). Grant camera permission when prompted to see your live feed. Use the Start/Stop button to control the stream.

## Notes

- The frontend talks directly to the browser media APIs, so the backend is only needed if you plan to extend the app with server features.
- The React client loads the MediaPipe Face Landmarker model locally (no server round-trips) and draws the detected landmark points on an overlaid canvas.
- Distance and head-rotation telemetry are estimated from landmark geometry (assuming an average 6.3cm interpupillary distance) and shown in realtime.
- Use the quality toggle to switch between low/medium/high webcam constraints (resolution + FPS) without leaving the browser.
- Update the allowed origin in `backend/app/main.py` if you change the frontend host/port.
