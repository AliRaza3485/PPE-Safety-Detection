"""Train / fine-tune a YOLO model for PPE safety detection.

Examples
--------
    python -m src.train                                   # uses params.yaml
    python -m src.train --epochs 100 --imgsz 640 --batch 32
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.utils import MODELS_DIR, ROOT_DIR

PARAMS_FILE = ROOT_DIR / "params.yaml"


def load_params() -> dict:
    if PARAMS_FILE.exists():
        return yaml.safe_load(PARAMS_FILE.read_text()).get("train", {})
    return {}


def parse_args() -> argparse.Namespace:
    p = load_params()
    parser = argparse.ArgumentParser(description="Train the PPE detection model")
    parser.add_argument("--data", default=str(ROOT_DIR / "data" / "data.yaml"))
    parser.add_argument("--model", default=p.get("model", "yolov8n.pt"))
    parser.add_argument("--epochs", type=int, default=p.get("epochs", 50))
    parser.add_argument("--imgsz", type=int, default=p.get("imgsz", 640))
    parser.add_argument("--batch", type=int, default=p.get("batch", 16))
    parser.add_argument("--patience", type=int, default=p.get("patience", 15))
    parser.add_argument("--lr0", type=float, default=p.get("lr0", 0.01))
    parser.add_argument("--seed", type=int, default=p.get("seed", 42))
    parser.add_argument("--name", default="ppe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[train] base model: {args.model} | data: {args.data}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        lr0=args.lr0,
        seed=args.seed,
        name=args.name,
    )

    # Copy the best checkpoint to models/best.pt (the DVC-tracked output).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, MODELS_DIR / "best.pt")
        print(f"[train] saved -> {MODELS_DIR / 'best.pt'}")

    # Persist headline metrics for DVC.
    metrics = getattr(results, "results_dict", {}) or {}
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps({k: round(float(v), 5) for k, v in metrics.items()}, indent=2)
    )
    print(f"[train] metrics -> {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
