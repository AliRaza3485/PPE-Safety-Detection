# api/schemas/prediction.py
"""Pydantic v2 response models for the PPE detection API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DetectionOut(BaseModel):
    label: str
    confidence: float
    bbox: list[float]
    violation: bool

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "head",
                "confidence": 0.94,
                "bbox": [120.5, 45.0, 260.0, 210.0],
                "violation": True,
            }
        }
    )


class SummaryOut(BaseModel):
    total: int
    counts: dict[str, int]
    violation_count: int
    compliant: bool

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 3,
                "counts": {"head": 1, "helmet": 2},
                "violation_count": 1,
                "compliant": False,
            }
        }
    )


class PredictResponse(BaseModel):
    filename: str
    detections: list[DetectionOut]
    summary: SummaryOut

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "img.jpg",
                "detections": [
                    {
                        "label": "head",
                        "confidence": 0.94,
                        "bbox": [120.5, 45.0, 260.0, 210.0],
                        "violation": True,
                    }
                ],
                "summary": {
                    "total": 3,
                    "counts": {"head": 1, "helmet": 2},
                    "violation_count": 1,
                    "compliant": False,
                },
            }
        }
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    weights_path: str
    classes: list[str]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
                "weights_path": "models/best.pt",
                "classes": ["head", "helmet"],
            }
        }
    )
