"""
Unit tests for ppe_detector.inference.postprocess.

Covers the violation rule (is_violation / VIOLATION_LABELS) and
build_summary across the cases that matter most for a safety-critical
label: fully compliant, at least one violation, no detections at all,
and correct per-label counting.
"""

from ppe_detector.inference.postprocess import (
    Detection,
    build_summary,
    is_violation,
)


def _det(label: str, confidence: float = 0.9) -> Detection:
    # Mirrors what results_to_detections() does in production: violation is
    # derived from is_violation(label), never left to the dataclass default.
    return Detection(
        label=label,
        confidence=confidence,
        bbox=[0.0, 0.0, 10.0, 10.0],
        violation=is_violation(label),
    )


class TestIsViolation:
    def test_head_is_a_violation(self):
        assert is_violation("head") is True

    def test_helmet_is_not_a_violation(self):
        assert is_violation("helmet") is False

    def test_unknown_label_is_not_a_violation(self):
        # Anything not explicitly in VIOLATION_LABELS must default to False —
        # a new/unexpected class should never silently become a violation.
        assert is_violation("safety_vest") is False


class TestBuildSummaryCompliantCase:
    def test_all_helmets_is_compliant(self):
        detections = [_det("helmet"), _det("helmet"), _det("helmet")]
        summary = build_summary(detections)

        assert summary.total == 3
        assert summary.counts == {"helmet": 3}
        assert summary.violation_count == 0
        assert summary.compliant is True


class TestBuildSummaryViolationCase:
    def test_single_head_triggers_violation(self):
        detections = [_det("helmet"), _det("helmet"), _det("head")]
        summary = build_summary(detections)

        assert summary.total == 3
        assert summary.counts == {"helmet": 2, "head": 1}
        assert summary.violation_count == 1
        assert summary.compliant is False

    def test_multiple_heads_counted_correctly(self):
        detections = [_det("head"), _det("head"), _det("helmet")]
        summary = build_summary(detections)

        assert summary.violation_count == 2
        assert summary.compliant is False
        assert summary.counts == {"head": 2, "helmet": 1}


class TestBuildSummaryEmptyDetections:
    def test_no_detections_is_compliant_with_zero_total(self):
        summary = build_summary([])

        assert summary.total == 0
        assert summary.counts == {}
        assert summary.violation_count == 0
        assert summary.compliant is True


class TestBuildSummaryCounts:
    def test_counts_match_detection_labels_exactly(self):
        detections = [
            _det("helmet"),
            _det("helmet"),
            _det("helmet"),
            _det("head"),
        ]
        summary = build_summary(detections)

        assert summary.counts["helmet"] == 3
        assert summary.counts["head"] == 1
        assert sum(summary.counts.values()) == summary.total

    def test_detection_violation_flag_is_set_on_construction(self):
        # Detection.violation should reflect is_violation(label) at the point
        # results_to_detections builds it — here we just confirm the flag
        # is consistent with what build_summary derives from it.
        head = Detection(label="head", confidence=0.8, bbox=[0, 0, 1, 1], violation=True)
        helmet = Detection(label="helmet", confidence=0.8, bbox=[0, 0, 1, 1], violation=False)

        summary = build_summary([head, helmet])
        assert summary.violation_count == 1
