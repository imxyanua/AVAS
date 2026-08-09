"""Evaluate a trained checkpoint on one split and write a report.

Reports both the action metrics and the normal against abnormal decision, plus
per-clip predictions so mistakes can be traced back to individual recordings.

Example:
    python -m src.training.evaluate --checkpoint models/checkpoints/baseline.pt --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.features.cache_index import (
    CachedClip,
    FeatureCacheError,
    build_dataset,
    clips_for_split,
    read_feature_index,
    validate_against_config,
)
from src.models.classifier import BehaviorClassifier, BehaviorDecision
from src.models.cnn_lstm import build_model
from src.utils.config import DatasetConfig, ModelConfig, TrainingConfig
from src.utils.device import resolve_device
from src.utils.metrics import (
    ClassificationReport,
    anomaly_detection_metrics,
    compute_metrics,
)

logger = logging.getLogger("avas.evaluate")

PREDICTION_COLUMNS = (
    "video",
    "clip",
    "label",
    "prediction",
    "confidence",
    "anomaly_score",
    "risk_level",
    "correct",
)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, object]:
    """Load a checkpoint written by the training entry point."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing = [key for key in ("model_state", "classes", "feature_dim") if key not in payload]
    if missing:
        raise ValueError(f"checkpoint {checkpoint_path} is missing keys: {missing}")
    return payload


@torch.no_grad()
def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], torch.Tensor]:
    """Return the true labels and the predicted class probabilities."""
    model.eval()
    targets: list[int] = []
    probabilities: list[torch.Tensor] = []

    for features, labels in loader:
        logits = model(features.to(device))
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu())
        targets.extend(labels.tolist())

    return targets, torch.cat(probabilities)


def write_predictions(
    clips: list[CachedClip], decisions: list[BehaviorDecision], path: Path
) -> Path:
    """Store one row per clip so individual errors can be inspected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        for clip, decision in zip(clips, decisions, strict=True):
            writer.writerow(
                [
                    clip.video.as_posix(),
                    clip.clip,
                    clip.label,
                    decision.action,
                    f"{decision.confidence:.6f}",
                    f"{decision.anomaly_score:.6f}",
                    decision.risk_level,
                    int(decision.action == clip.label),
                ]
            )
    return path


def write_confusion_matrix(report: ClassificationReport, path: Path) -> Path:
    """Store the confusion matrix as CSV with class names on both axes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual", *report.classes])
        for name, row in zip(report.classes, report.confusion, strict=True):
            writer.writerow([name, *(int(value) for value in row)])
    return path


def save_confusion_figure(report: ClassificationReport, path: Path) -> Path:
    """Render the confusion matrix as an image for inclusion in a written report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(1.4 * len(report.classes) + 2,) * 2)
    axes.imshow(report.confusion, cmap="Blues")
    axes.set_xticks(np.arange(len(report.classes)), report.classes, rotation=45, ha="right")
    axes.set_yticks(np.arange(len(report.classes)), report.classes)
    axes.set_xlabel("predicted")
    axes.set_ylabel("actual")

    threshold = report.confusion.max() / 2 if report.confusion.max() else 0
    for row in range(len(report.classes)):
        for column in range(len(report.classes)):
            value = int(report.confusion[row, column])
            axes.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="override the architecture stored in the checkpoint",
    )
    parser.add_argument("--feature-index", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="report directory (default: outputs/reports/<checkpoint name>)",
    )
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--figure", action="store_true", help="also save the confusion matrix as an image"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    dataset_config = DatasetConfig.from_yaml(args.dataset_config)
    training_config = TrainingConfig.from_yaml(args.training_config)
    device = resolve_device(args.device or training_config.device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    classes = tuple(str(name) for name in checkpoint["classes"])
    if classes != dataset_config.classes:
        raise SystemExit(
            f"checkpoint was trained on classes {list(classes)}, but the dataset config "
            f"declares {list(dataset_config.classes)}"
        )

    if args.model_config is not None:
        model_config = ModelConfig.from_yaml(args.model_config)
    elif "model_config" in checkpoint:
        model_config = ModelConfig.from_mapping(checkpoint["model_config"])
    else:
        raise SystemExit(
            "checkpoint does not record its architecture; pass --model-config explicitly"
        )

    feature_dim = int(checkpoint["feature_dim"])
    model = build_model(model_config, len(classes), feature_dim=feature_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    index_path = args.feature_index or training_config.feature_cache_root / "index.csv"
    clips = read_feature_index(index_path)
    validate_against_config(clips, dataset_config)
    split_clips = clips_for_split(clips, args.split)

    dataset = build_dataset(split_clips, expected_dim=feature_dim)
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    targets, probabilities = predict(model, loader, device)
    if len(targets) != len(split_clips):
        raise FeatureCacheError(
            f"evaluated {len(targets)} clips but the split lists {len(split_clips)}"
        )

    behaviour = BehaviorClassifier.from_config(dataset_config, model_config)
    decisions = behaviour.decide_batch(probabilities)
    predictions = [dataset_config.class_to_index[decision.action] for decision in decisions]

    report = compute_metrics(targets, predictions, classes)
    anomaly = anomaly_detection_metrics(targets, predictions, behaviour.abnormal_indices)

    output_dir = args.output_dir or Path("outputs/reports") / args.checkpoint.stem
    predictions_path = write_predictions(split_clips, decisions, output_dir / "predictions.csv")
    confusion_path = write_confusion_matrix(report, output_dir / "confusion_matrix.csv")
    figure_path = (
        save_confusion_figure(report, output_dir / "confusion_matrix.png") if args.figure else None
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset": dataset_config.name,
        "split": args.split,
        "clips": len(split_clips),
        "device": str(device),
        "feature_dim": feature_dim,
        "action_metrics": report.to_dict(),
        "anomaly_metrics": anomaly,
        "anomaly_thresholds": {
            "suspicious": model_config.anomaly.suspicious_threshold,
            "high_risk": model_config.anomaly.high_risk_threshold,
        },
        "risk_levels": _count_risk_levels(decisions),
        "predictions": str(predictions_path),
        "confusion_matrix": str(confusion_path),
    }
    if figure_path is not None:
        payload["confusion_figure"] = str(figure_path)

    report_path = output_dir / f"{args.split}_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("evaluated %d clips from split %r", len(split_clips), args.split)
    logger.info("\n%s", report.format_table())
    logger.info(
        "abnormal recall=%.4f precision=%.4f false alarm rate=%.4f",
        anomaly["abnormal_recall"],
        anomaly["abnormal_precision"],
        anomaly["false_alarm_rate"],
    )
    logger.info("report written to %s", report_path)
    return 0


def _count_risk_levels(decisions: list[BehaviorDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.risk_level] = counts.get(decision.risk_level, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
