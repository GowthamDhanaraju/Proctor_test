from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Webcam Viewer API")

# Allow frontend dev server access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Simple landing message for browsers hitting the API root."""
    return {"message": "FastAPI backend is running. Frontend lives at http://localhost:5173."}


@app.get("/health")
def read_health() -> dict[str, str]:
    """Simple endpoint so we can confirm the API is running."""
    return {"status": "ok"}
