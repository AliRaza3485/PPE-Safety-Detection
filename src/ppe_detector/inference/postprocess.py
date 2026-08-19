"""
Pure postprocessing for PPE detection outputs.

No model loading, no I/O, no framework imports (FastAPI/Ultralytics-agnostic
where possible) — just typed transforms from raw Ultralytics `Results` into
the detection/summary shapes the API and frontend both consume. Keeping this
pure makes it trivially unit-testable and reusable from the CLI, the API,
and the evaluation script alike.

Frontend contract (frontend/app/page.tsx):
    Detection: {label, confidence, bbox: [x1, y1, x2, y2], violation}
    Summary:   {total, counts: {label: n}, violation_count, compliant}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Single source of truth for which class(es) count as a safety violation.
# class 0 = 'head' = no helmet worn = VIOLATION. class 1 = 'helmet' = compliant.
# Extending to more violation classes later (e.g. 'no-vest') means adding
# to this set only — nothing else in the pipeline needs to change.
VIOLATION_LABELS: set[str] = {"head"}


@dataclass
class Detection:
    """A single detected object, in the shape the frontend expects."""

    label: str
    confidence: float
    bbox: list[float] = field(default_factory=list)  # [x1, y1, x2, y2] in pixels
    violation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "violation": self.violation,
        }


@dataclass
class Summary:
    """Aggregate summary over a list of detections."""

    total: int
    counts: dict[str, int]
    violation_count: int
    compliant: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "counts": self.counts,
            "violation_count": self.violation_count,
            "compliant": self.compliant,
        }


def is_violation(label: str) -> bool:
    """Single choke point for the violation rule. Extend VIOLATION_LABELS, not this."""
    return label in VIOLATION_LABELS


def results_to_detections(result: Any, class_names: dict[int, str] | None = None) -> list[Detection]:
    """
    Convert one Ultralytics `Results` object (i.e. `model.predict(img)[0]`)
    into a list of typed `Detection`s.

    `class_names` overrides result.names if provided (rarely needed — Ultralytics
    results already carry the model's class map, but this keeps the function
    usable with plain box data too).
    """
    names = class_names if class_names is not None else result.names
    detections: list[Detection] = []

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0])
        label = names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                bbox=[x1, y1, x2, y2],
                violation=is_violation(label),
            )
        )

    return detections


def build_summary(detections: list[Detection]) -> Summary:
    """
    Aggregate a list of detections into the frontend's Summary shape.

    compliant == True only when there are zero violations. An image with
    zero detections at all is treated as compliant (nothing flagged) —
    this mirrors "no violation found", not "no data"; callers who need to
    distinguish "no people present" from "fully compliant" should inspect
    `total` themselves.
    """
    counts: dict[str, int] = {}
    violation_count = 0

    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
        if det.violation:
            violation_count += 1

    return Summary(
        total=len(detections),
        counts=counts,
        violation_count=violation_count,
        compliant=(violation_count == 0),
    )
