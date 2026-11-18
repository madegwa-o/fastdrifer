"""
main.py - simple crash analytics API

Every endpoint takes a video upload and returns something different:

    POST /info       video details (duration, fps, resolution)
    POST /severity   minor / moderate / major (calls your Hugging Face Space)
    POST /frames     a ZIP of evenly spaced frames (?count=8)
    POST /motion     motion over time + the moment of peak motion (the impact)
    POST /report     all of the above summarised in one response

Run:
    export HF_TOKEN="hf_..."
    uvicorn main:app --reload

Then open http://localhost:8000/docs to upload videos from your browser.
"""

import io
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from gradio_client import Client, handle_file

SPACE = os.getenv("HF_SPACE", "Madegwa/Accident_Severity_API")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="Drifer Crash Analytics")


# ---------- helpers ----------

@contextmanager
def temp_video(upload: UploadFile):
    """Save the upload to a temp file (OpenCV needs a path), delete it after."""
    suffix = os.path.splitext(upload.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        shutil.copyfileobj(upload.file, f)
    try:
        yield f.name
    finally:
        os.remove(f.name)


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frames <= 0:
        raise HTTPException(422, "Could not read this video")
    return {
        "duration_seconds": round(frames / fps, 2),
        "fps": round(fps, 2),
        "frame_count": frames,
        "resolution": f"{width}x{height}",
    }


def motion_timeline(path):
    """Average frame-to-frame change for each second of video."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    per_second = {}
    prev, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        if prev is not None:
            per_second.setdefault(int(i / fps), []).append(float(cv2.absdiff(gray, prev).mean()))
        prev, i = gray, i + 1
    cap.release()

    timeline = [{"second": s, "motion": round(sum(v) / len(v), 2)} for s, v in sorted(per_second.items())]
    if not timeline:
        raise HTTPException(422, "Could not read this video")
    return timeline


@lru_cache
def get_client():
    return Client(SPACE, token=HF_TOKEN)


def get_severity(path):
    try:
        result = get_client().predict(handle_file(path), api_name="/predict")
    except Exception as e:
        raise HTTPException(502, f"Severity model unavailable: {e}")
    probs = {c["label"]: round(c["confidence"], 4) for c in result["confidences"]}
    return {
        "severity": result["label"],
        "confidence": probs[result["label"]],
        "probabilities": probs,
    }


# ---------- endpoints ----------

@app.post("/info")
def info(video: UploadFile = File(...)):
    with temp_video(video) as path:
        return video_info(path)


@app.post("/severity")
def severity(video: UploadFile = File(...)):
    with temp_video(video) as path:
        return get_severity(path)


@app.post("/frames")
def frames(video: UploadFile = File(...), count: int = 8):
    count = max(1, min(count, 60))
    with temp_video(video) as path:
        total = video_info(path)["frame_count"]
        wanted = {int(total * (k + 0.5) / count) for k in range(count)}

        zip_buffer = io.BytesIO()
        cap = cv2.VideoCapture(path)
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i in wanted:
                    _, jpg = cv2.imencode(".jpg", frame)
                    zf.writestr(f"frame_{i:05d}.jpg", jpg.tobytes())
                i += 1
        cap.release()

    return Response(
        zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=frames.zip"},
    )


@app.post("/motion")
def motion(video: UploadFile = File(...)):
    with temp_video(video) as path:
        timeline = motion_timeline(path)
    return {"peak": max(timeline, key=lambda t: t["motion"]), "timeline": timeline}


@app.post("/report")
def report(video: UploadFile = File(...)):
    with temp_video(video) as path:
        timeline = motion_timeline(path)
        return {
            "reported_at": datetime.now(timezone.utc).isoformat(),
            "video": video_info(path),
            "peak_motion": max(timeline, key=lambda t: t["motion"]),
            "severity": get_severity(path),
        }