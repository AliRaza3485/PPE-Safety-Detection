# tests/unit/test_api.py
"""Unit tests for the PPE detection API using a fake detector (no real model load)."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.dependencies import get_detector
from api.main import app
from ppe_detector.inference.postprocess import Detection, Summary


class _FakeModel:
    """Minimal stand-in for the ultralytics model — only what /health reads."""

    names = {0: "head", 1: "helmet"}


class FakeDetector:
    """Stand-in for PPEDetector: returns canned detections, no real model load."""

    def __init__(self, detections=None, summary=None):
        self.model = _FakeModel()
        self._detections = detections if detections is not None else []
        self._summary = (
            summary
            if summary is not None
            else Summary(total=0, counts={}, violation_count=0, compliant=True)
        )

    def predict(self, image):
        return self._detections, self._summary


def _make_image_bytes(fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def client():
    fake = FakeDetector(
        detections=[
            Detection(
                label="head",
                confidence=0.94,
                bbox=[10.0, 10.0, 50.0, 50.0],
                violation=True,
            ),
            Detection(
                label="helmet",
                confidence=0.88,
                bbox=[60.0, 10.0, 100.0, 50.0],
                violation=False,
            ),
        ],
        summary=Summary(
            total=2, counts={"head": 1, "helmet": 1}, violation_count=1, compliant=False
        ),
    )
    # NOTE: no `with TestClient(app)` context manager here on purpose —
    # that would trigger main.py's lifespan and try to load real weights.
    # Overriding get_detector directly keeps these tests hermetic.
    app.dependency_overrides[get_detector] = lambda: fake
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "classes" in body


def test_predict_valid_image_returns_expected_shape(client):
    image_bytes = _make_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("img.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["filename"] == "img.jpg"
    assert isinstance(body["detections"], list)
    assert len(body["detections"]) == 2
    assert body["detections"][0] == {
        "label": "head",
        "confidence": 0.94,
        "bbox": [10.0, 10.0, 50.0, 50.0],
        "violation": True,
    }
    assert body["summary"] == {
        "total": 2,
        "counts": {"head": 1, "helmet": 1},
        "violation_count": 1,
        "compliant": False,
    }


def test_predict_non_image_returns_400(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_violation_case_marks_non_compliant(client):
    image_bytes = _make_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("img.jpg", image_bytes, "image/jpeg")},
    )
    body = response.json()
    assert body["summary"]["compliant"] is False
    assert body["summary"]["violation_count"] == 1
