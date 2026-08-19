"""
Framework-agnostic YOLOv8 inference wrapper.

Deliberately has no FastAPI (or any web-framework) imports — this module
should be usable from the CLI, a notebook, the API layer, or a batch script
without dragging in HTTP concerns. The API layer is a thin adapter on top
of this, not the other way around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO

from ppe_detector.inference.postprocess import (
    Detection,
    Summary,
    build_summary,
    results_to_detections,
)

ImageInput = Union[str, Path, np.ndarray, Image.Image]


class PPEDetector:
    """
    Thin, reusable wrapper around a YOLOv8 model for PPE detection.

    Loads weights once at construction time; `.predict()` is cheap to call
    repeatedly (no reload per call — see api/dependencies.py for the
    request-scoped caching layer that relies on this).
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found at '{weights_path}'. "
                f"Point at a checkpoint under runs/train/.../weights/, "
                f"models/best.pt, or the yolov8n.pt placeholder."
            )

        self.weights_path = weights_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(str(weights_path))

    def predict(self, image: ImageInput) -> tuple[list[Detection], Summary]:
        """
        Run inference on a single image (path, numpy array, or PIL Image).
        Returns (detections, summary) — ready to serialize straight into
        the API response / frontend contract.
        """
        results = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        detections = results_to_detections(result, class_names=self.model.names)
        summary = build_summary(detections)
        return detections, summary

    def predict_batch(
        self, images: list[ImageInput]
    ) -> list[tuple[list[Detection], Summary]]:
        """
        Run inference on a batch of images. Uses Ultralytics' native batching
        (single forward pass per batch) rather than looping .predict() —
        meaningfully faster on GPU for anything beyond a couple of images.
        """
        results = self.model.predict(
            images,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        output: list[tuple[list[Detection], Summary]] = []
        for result in results:
            detections = results_to_detections(result, class_names=self.model.names)
            summary = build_summary(detections)
            output.append((detections, summary))
        return output

    def __repr__(self) -> str:
        return (
            f"PPEDetector(weights='{self.weights_path.name}', "
            f"conf={self.conf}, iou={self.iou}, device='{self.device}')"
        )
