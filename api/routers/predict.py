# api/routers/predict.py
"""Prediction and health-check endpoints for the PPE detection API."""

from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.dependencies import get_detector, get_weights_path
from api.schemas.prediction import (
    DetectionOut,
    HealthResponse,
    PredictResponse,
    SummaryOut,
)
from ppe_detector.inference.predictor import PPEDetector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


@router.get("/health", response_model=HealthResponse)
def health(detector: PPEDetector = Depends(get_detector)) -> HealthResponse:
    """Report service and model status."""
    classes = list(detector.model.names.values())
    return HealthResponse(
        status="ok",
        model_loaded=True,
        weights_path=get_weights_path(),
        classes=classes,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    detector: PPEDetector = Depends(get_detector),
) -> PredictResponse:
    """Run PPE detection on an uploaded image and return detections + summary."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. Please upload an image file.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400, detail="Could not read the uploaded image."
        ) from exc

    detections, summary = detector.predict(image)

    return PredictResponse(
        filename=file.filename or "unknown",
        detections=[DetectionOut(**d.to_dict()) for d in detections],
        summary=SummaryOut(**summary.to_dict()),
    )
