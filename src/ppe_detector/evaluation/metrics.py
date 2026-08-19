"""
Per-class evaluation for the PPE detector.

Aggregate mAP hides exactly the thing we care about most here: with a 3:1
helmet:head imbalance, a model can post a great overall mAP50 while quietly
missing a large fraction of `head` (violation) instances. This module always
surfaces per-class numbers first, and calls out `head` recall explicitly,
because a missed violation is the failure mode that actually matters for
this project's stated goal (safety compliance), not aggregate score.

CLI:
    python -m ppe_detector.evaluation.metrics \\
        --weights models/best.pt \\
        --data data/raw/data.yaml \\
        --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

# Class we treat as the safety violation — kept in sync with
# ppe_detector.inference.postprocess.VIOLATION_LABELS conceptually, but this
# module only needs the name for the printed callout, not the routing logic.
VIOLATION_CLASS_NAME = "head"

DEFAULT_METRICS_PATH = Path("models/metrics.json")


def evaluate(
    weights_path: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    save_json_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """
    Run Ultralytics `model.val()` on the given split and extract per-class
    precision, recall, mAP50, and mAP50-95 — not just the aggregate scores.

    Returns the metrics dict that also gets written to `save_json_path`.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at '{weights_path}'")

    model = YOLO(str(weights_path))
    results = model.val(data=str(data_yaml), split=split, verbose=False)

    box = results.box
    class_indices = list(box.ap_class_index)
    names = model.names  # {class_id: class_name}

    per_class: dict[str, dict[str, float]] = {}
    for i, class_id in enumerate(class_indices):
        class_name = names[int(class_id)]
        per_class[class_name] = {
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "mAP50": round(float(box.ap50[i]), 4),
            "mAP50-95": round(float(box.ap[i]), 4),
        }

    metrics: dict[str, Any] = {
        "weights": str(weights_path),
        "data_yaml": str(data_yaml),
        "split": split,
        "aggregate": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
        },
        "per_class": per_class,
        "artifacts": {
            "confusion_matrix": str(Path(results.save_dir) / "confusion_matrix.png"),
            "pr_curve": str(Path(results.save_dir) / "PR_curve.png"),
            "save_dir": str(results.save_dir),
        },
    }

    _print_report(metrics)
    _save_json(metrics, save_json_path)

    return metrics


def _print_report(metrics: dict[str, Any]) -> None:
    per_class = metrics["per_class"]
    agg = metrics["aggregate"]

    print("\n" + "=" * 66)
    print(f"  Evaluation — split: {metrics['split']}  |  weights: {Path(metrics['weights']).name}")
    print("=" * 66)
    header = f"{'class':<10} {'precision':>10} {'recall':>10} {'mAP50':>10} {'mAP50-95':>10}"
    print(header)
    print("-" * 66)
    for class_name, vals in per_class.items():
        print(
            f"{class_name:<10} {vals['precision']:>10.4f} {vals['recall']:>10.4f} "
            f"{vals['mAP50']:>10.4f} {vals['mAP50-95']:>10.4f}"
        )
    print("-" * 66)
    print(
        f"{'all':<10} {agg['precision']:>10.4f} {agg['recall']:>10.4f} "
        f"{agg['mAP50']:>10.4f} {agg['mAP50-95']:>10.4f}"
    )
    print("=" * 66)

    if VIOLATION_CLASS_NAME in per_class:
        head_recall = per_class[VIOLATION_CLASS_NAME]["recall"]
        head_precision = per_class[VIOLATION_CLASS_NAME]["precision"]
        print(
            f"\n  >>> VIOLATION CLASS ('{VIOLATION_CLASS_NAME}') "
            f"— recall: {head_recall:.4f}  precision: {head_precision:.4f}  <<<"
        )
        print(
            "      (recall = fraction of real violations the model actually catches;\n"
            "       missed recall here means missed safety violations in production)\n"
        )
    else:
        print(f"\n  WARNING: '{VIOLATION_CLASS_NAME}' class not found in eval results.\n")

    print(f"  Confusion matrix: {metrics['artifacts']['confusion_matrix']}")
    print(f"  PR curve:         {metrics['artifacts']['pr_curve']}\n")


def _save_json(metrics: dict[str, Any], save_json_path: str | Path) -> None:
    save_json_path = Path(save_json_path)
    save_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics -> {save_json_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-class evaluation for the PPE detector.")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--out", default=str(DEFAULT_METRICS_PATH), help="Where to save metrics.json"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        save_json_path=args.out,
    )


if __name__ == "__main__":
    main()
