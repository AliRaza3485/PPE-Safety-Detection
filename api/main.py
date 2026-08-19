"""FastAPI inference service for PPE safety detection.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health   -> service + model status
    POST /predict  -> upload an image, get detections + a compliance summary
"""
from __future__ import annotations

import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.utils import (
    CLASS_NAMES,
    DEFAULT_WEIGHTS,
    load_model,
    parse_results,
    summarize,
)

app = FastAPI(title="PPE Safety Detection API", version="1.0.0")

# Allow the Next.js frontend (configurable via env) to call the API.
_origins = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup.
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
model = load_model(os.getenv("WEIGHTS", str(DEFAULT_WEIGHTS)))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "conf_threshold": CONF_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # PIL(RGB) -> OpenCV(BGR) ndarray for the model.
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
    detections = parse_results(results[0])

    return {
        "filename": file.filename,
        "detections": detections,
        "summary": summarize(detections),
    }
