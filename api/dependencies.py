# api/dependencies.py
"""Startup wiring for the PPEDetector singleton, injected via FastAPI Depends."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ppe_detector.inference.predictor import PPEDetector

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = "models/best.pt"
_DEFAULT_CONF = 0.25
_DEFAULT_IOU = 0.45
_DEFAULT_DEVICE = "cpu"

_detector: PPEDetector | None = None
_weights_path: Path | None = None


def _config_from_env() -> dict[str, str | float]:
    return {
        "weights_path": os.getenv("PPE_WEIGHTS", _DEFAULT_WEIGHTS),
        "conf": float(os.getenv("PPE_CONF", _DEFAULT_CONF)),
        "iou": float(os.getenv("PPE_IOU", _DEFAULT_IOU)),
        "device": os.getenv("PPE_DEVICE", _DEFAULT_DEVICE),
    }


def load_detector() -> PPEDetector:
    """Instantiate the PPEDetector singleton. Call once, at API startup."""
    global _detector, _weights_path

    if _detector is not None:
        return _detector

    config = _config_from_env()
    weights_path = Path(config["weights_path"])

    if not weights_path.exists():
        raise FileNotFoundError(
            f"PPE model weights not found at '{weights_path}'. "
            "Set the PPE_WEIGHTS environment variable to a valid path, "
            f"or place trained weights at the default location ('{_DEFAULT_WEIGHTS}')."
        )

    logger.info(
        "Loading PPEDetector weights=%s conf=%s iou=%s device=%s",
        weights_path,
        config["conf"],
        config["iou"],
        config["device"],
    )

    _detector = PPEDetector(
        weights_path=weights_path,
        conf=config["conf"],
        iou=config["iou"],
        device=config["device"],
    )
    _weights_path = weights_path
    return _detector


def get_detector() -> PPEDetector:
    """FastAPI dependency: returns the already-loaded singleton detector."""
    if _detector is None:
        raise RuntimeError(
            "PPEDetector has not been loaded yet. "
            "Ensure the application's startup/lifespan event calls load_detector()."
        )
    return _detector


def get_weights_path() -> str:
    """Returns the resolved weights path (loaded value if available, else env/default)."""
    if _weights_path is not None:
        return str(_weights_path)
    return os.getenv("PPE_WEIGHTS", _DEFAULT_WEIGHTS)


def reset_detector() -> None:
    """Test helper: clears the cached singleton so tests can override it cleanly."""
    global _detector, _weights_path
    _detector = None
    _weights_path = None
