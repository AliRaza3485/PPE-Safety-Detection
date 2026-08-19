"""
Training script for PPE detection (YOLOv8).
Reads hyperparameters from configs/train_config.yaml and logs
the run to MLflow.
"""

import yaml
from pathlib import Path
import mlflow
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str = "configs/train_config.yaml"):
    config = load_config(config_path)

    model_cfg = config["model"]
    train_cfg = config["training"]
    aug_cfg = config["augmentation"]
    paths_cfg = config["paths"]
    tracking_cfg = config["experiment_tracking"]

    mlflow.set_experiment(tracking_cfg["mlflow_experiment_name"])
    run_name = f"{tracking_cfg['run_name_prefix']}_{train_cfg['epochs']}ep"

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(
            {
                "model_architecture": model_cfg["architecture"],
                "epochs": train_cfg["epochs"],
                "batch_size": train_cfg["batch_size"],
                "img_size": train_cfg["img_size"],
                "learning_rate": train_cfg["learning_rate"],
                "patience": train_cfg["patience"],
                "seed": train_cfg["seed"],
                "mosaic": aug_cfg["mosaic"],
                "mixup": aug_cfg["mixup"],
            }
        )

        # Load pretrained model
        model = YOLO(model_cfg["architecture"])

        # Train
        results = model.train(
            data=paths_cfg["data_yaml"],
            epochs=train_cfg["epochs"],
            batch=train_cfg["batch_size"],
            imgsz=train_cfg["img_size"],
            patience=train_cfg["patience"],
            lr0=train_cfg["learning_rate"],
            device=train_cfg["device"],
            seed=train_cfg["seed"],
            hsv_h=aug_cfg["hsv_h"],
            hsv_s=aug_cfg["hsv_s"],
            hsv_v=aug_cfg["hsv_v"],
            degrees=aug_cfg["degrees"],
            translate=aug_cfg["translate"],
            scale=aug_cfg["scale"],
            fliplr=aug_cfg["fliplr"],
            mosaic=aug_cfg["mosaic"],
            mixup=aug_cfg["mixup"],
            project=paths_cfg["output_dir"],
            name=run_name,
        )

        # Log key metrics from training results
        metrics = results.results_dict
        mlflow.log_metrics(
            {
                "mAP50": metrics.get("metrics/mAP50(B)", 0),
                "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
                "precision": metrics.get("metrics/precision(B)", 0),
                "recall": metrics.get("metrics/recall(B)", 0),
            }
        )

        # Save and log the best model as an MLflow artifact
        best_model_path = (
            Path(paths_cfg["output_dir"]) / run_name / "weights" / "best.pt"
        )
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path))
            print(f"\nBest model saved at: {best_model_path}")

        print(
            f"\nTraining complete. Run logged to MLflow experiment: {tracking_cfg['mlflow_experiment_name']}"
        )


if __name__ == "__main__":
    train()
