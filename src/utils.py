"""Shared helpers for the PPE safety detection project.

Central place for project paths, the class list, and small utilities used by
both the training/CLI scripts and the API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── Project paths ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"

# ── Classes (Construction Site Safety dataset convention) ──
CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

# Classes that represent a safety violation (person missing required PPE).
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}


def is_violation(label: str) -> bool:
    """True if the given class label is a safety violation."""
    return label in VIOLATION_CLASSES


def color_for(label: str) -> tuple[int, int, int]:
    """BGR color for a label — red for violations, green otherwise."""
    return (0, 0, 255) if is_violation(label) else (0, 200, 0)


def load_model(weights: str | Path = DEFAULT_WEIGHTS):
    """Load a YOLO model.

    Falls back to the pretrained ``yolov8n.pt`` checkpoint when no trained
    weights are present, so the API/CLI still runs before the first training.
    """
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        print(f"[utils] {weights} not found — falling back to yolov8n.pt")
        return YOLO("yolov8n.pt")
    return YOLO(str(weights))


def parse_results(result) -> list[dict[str, Any]]:
    """Convert one Ultralytics ``Results`` object into plain dicts."""
    detections: list[dict[str, Any]] = []
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append(
            {
                "label": label,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "violation": is_violation(label),
            }
        )
    return detections


def summarize(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll detections up into per-class counts and a compliance flag."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det["label"]] = counts.get(det["label"], 0) + 1
    violations = [d for d in detections if d["violation"]]
    return {
        "total": len(detections),
        "counts": counts,
        "violation_count": len(violations),
        "compliant": len(violations) == 0,
    }


def draw_detections(image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw labeled bounding boxes onto a copy of ``image`` (BGR)."""
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        color = color_for(det["label"])
        caption = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, caption, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out
