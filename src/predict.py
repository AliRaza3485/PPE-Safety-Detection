"""Run PPE detection inference from the command line.

Examples
--------
    python -m src.predict --source image.jpg
    python -m src.predict --source folder/ --weights models/best.pt --conf 0.35
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.utils import (
    DEFAULT_WEIGHTS,
    draw_detections,
    load_model,
    parse_results,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPE detection inference")
    parser.add_argument("--source", required=True, help="image / folder / video path")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--out", default="runs/predict", help="output directory")
    parser.add_argument("--no-save", action="store_true", help="skip saving annotated images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)
    results = model.predict(source=args.source, conf=args.conf, verbose=False)

    for i, result in enumerate(results):
        detections = parse_results(result)
        summary = summarize(detections)
        name = Path(getattr(result, "path", f"frame_{i}")).stem

        status = "✅ COMPLIANT" if summary["compliant"] else "⚠️  VIOLATION"
        print(f"\n{name}: {status} — {summary['total']} detections")
        print(json.dumps(summary["counts"], indent=2))

        if not args.no_save and result.orig_img is not None:
            annotated = draw_detections(result.orig_img, detections)
            dest = out_dir / f"{name}_pred.jpg"
            cv2.imwrite(str(dest), annotated)
            print(f"  saved -> {dest}")


if __name__ == "__main__":
    main()
