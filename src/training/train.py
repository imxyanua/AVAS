"""Train the temporal head on cached backbone features.

Example:
    python -m src.training.train --run-name baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.features.cache_index import (
    assert_no_group_leakage,
    build_dataset,
    clips_for_split,
    feature_dimension,
    read_feature_index,
    validate_against_config,
)
from src.models.cnn_lstm import build_model
from src.utils.config import DatasetConfig, ModelConfig, TrainingConfig
from src.utils.device import resolve_device
from src.utils.metrics import balanced_class_weights, compute_metrics
from src.utils.seeding import set_seed

logger = logging.getLogger("avas.train")

MONITOR_MODES = {"val_loss": "min", "val_accuracy": "max", "val_macro_f1": "max"}
HISTORY_COLUMNS = ("epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1")


@dataclass(frozen=True)
class EpochResult:
    """Losses and validation metrics for one epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float

    def monitored(self, monitor: str) -> float:
        values = {
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "val_macro_f1": self.val_macro_f1,
        }
        if monitor not in values:
            raise ValueError(f"unknown monitor {monitor!r}; expected one of {sorted(values)}")
        return values[monitor]


@dataclass(frozen=True)
class TrainingRun:
    """Outcome of a training run."""

    history: list[EpochResult]
    best_epoch: int
    best_value: float
    monitor: str
    checkpoint_path: Path
    stopped_early: bool


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> tuple[float, list[int], list[int]]:
    """Run one pass over ``loader``, training when an optimizer is supplied."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0
    targets: list[int] = []
    predictions: list[int] = []

    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_items += batch_size
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())

    return total_loss / total_items, targets, predictions


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: tuple[str, ...],
    training: TrainingConfig,
    device: torch.device,
    *,
    class_weights: torch.Tensor | None = None,
    checkpoint_path: Path,
    extra_checkpoint_fields: dict[str, object] | None = None,
) -> TrainingRun:
    """Fit the model, keeping the checkpoint of the best monitored epoch."""
    monitor = training.early_stopping.monitor
    if monitor not in MONITOR_MODES:
        raise ValueError(
            f"unsupported monitor {monitor!r}; expected one of {sorted(MONITOR_MODES)}"
        )
    mode = MONITOR_MODES[monitor]

    model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    history: list[EpochResult] = []
    best_value = float("inf") if mode == "min" else float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, training.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)

        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            gradient_clip_norm=training.gradient_clip_norm,
        )
        val_loss, val_targets, val_predictions = run_epoch(model, val_loader, criterion, device)
        report = compute_metrics(val_targets, val_predictions, classes)

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=report.accuracy,
            val_macro_f1=report.macro_f1,
        )
        history.append(result)
        logger.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_accuracy=%.4f val_macro_f1=%.4f",
            epoch,
            training.epochs,
            train_loss,
            val_loss,
            report.accuracy,
            report.macro_f1,
        )

        value = result.monitored(monitor)
        improved = value < best_value if mode == "min" else value > best_value
        if improved:
            best_value = value
            best_epoch = epoch
            epochs_without_improvement = 0
            payload: dict[str, object] = {
                "model_state": model.state_dict(),
                "classes": list(classes),
                "epoch": epoch,
                "monitor": monitor,
                "monitor_value": value,
            }
            payload.update(extra_checkpoint_fields or {})
            torch.save(payload, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training.early_stopping.patience:
                logger.info(
                    "stopping early: %s did not improve for %d epochs",
                    monitor,
                    epochs_without_improvement,
                )
                stopped_early = True
                break

    return TrainingRun(
        history=history,
        best_epoch=best_epoch,
        best_value=best_value,
        monitor=monitor,
        checkpoint_path=checkpoint_path,
        stopped_early=stopped_early,
    )


def write_history(run: TrainingRun, path: Path) -> Path:
    """Store per-epoch metrics as CSV so training curves can be plotted later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HISTORY_COLUMNS))
        writer.writeheader()
        for result in run.history:
            writer.writerow(asdict(result))
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument(
        "--feature-index",
        type=Path,
        default=None,
        help="feature cache index (default: <feature_cache_root>/index.csv)",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--run-name", default="baseline", help="subdirectory for logs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    dataset_config = DatasetConfig.from_yaml(args.dataset_config)
    model_config = ModelConfig.from_yaml(args.model_config)
    training_config = TrainingConfig.from_yaml(args.training_config)
    overrides = {
        key: value
        for key, value in (
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("learning_rate", args.learning_rate),
            ("device", args.device),
        )
        if value is not None
    }
    if overrides:
        training_config = TrainingConfig.from_mapping({**_as_dict(training_config), **overrides})

    set_seed(training_config.seed)
    device = resolve_device(training_config.device)

    index_path = args.feature_index or training_config.feature_cache_root / "index.csv"
    clips = read_feature_index(index_path)
    assert_no_group_leakage(clips)
    validate_against_config(clips, dataset_config)

    train_clips = clips_for_split(clips, args.train_split)
    val_clips = clips_for_split(clips, args.val_split)
    dimension = feature_dimension(train_clips)

    train_dataset = build_dataset(train_clips, expected_dim=dimension)
    val_dataset = build_dataset(val_clips, expected_dim=dimension)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    class_weights = None
    if training_config.class_weighting == "balanced":
        weights = balanced_class_weights(
            [clip.label_index for clip in train_clips], len(dataset_config.classes)
        )
        class_weights = torch.from_numpy(weights)
        logger.info(
            "class weights: %s",
            {
                name: round(float(weight), 3)
                for name, weight in zip(dataset_config.classes, weights, strict=True)
            },
        )

    model = build_model(model_config, len(dataset_config.classes), feature_dim=dimension)
    logger.info(
        "training %s on %d clips, validating on %d clips, feature_dim=%d, device=%s",
        model_config.architecture,
        len(train_clips),
        len(val_clips),
        dimension,
        device,
    )

    run_dir = training_config.log_dir / args.run_name
    checkpoint_path = training_config.checkpoint_dir / f"{args.run_name}.pt"
    run = train_model(
        model,
        train_loader,
        val_loader,
        dataset_config.classes,
        training_config,
        device,
        class_weights=class_weights,
        checkpoint_path=checkpoint_path,
        extra_checkpoint_fields={
            "feature_dim": dimension,
            "model_config": model_config.to_mapping(),
            "dataset": dataset_config.name,
        },
    )

    history_path = write_history(run, run_dir / "history.csv")
    summary = {
        "run_name": args.run_name,
        "dataset": dataset_config.name,
        "device": str(device),
        "feature_dim": dimension,
        "train_clips": len(train_clips),
        "validation_clips": len(val_clips),
        "epochs_run": len(run.history),
        "stopped_early": run.stopped_early,
        "monitor": run.monitor,
        "best_epoch": run.best_epoch,
        "best_value": run.best_value,
        "checkpoint": str(run.checkpoint_path),
        "history": str(history_path),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "best %s=%.4f at epoch %d; checkpoint %s",
        run.monitor,
        run.best_value,
        run.best_epoch,
        run.checkpoint_path,
    )
    return 0


def _as_dict(config: TrainingConfig) -> dict[str, object]:
    return {
        "seed": config.seed,
        "device": config.device,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip_norm": config.gradient_clip_norm,
        "class_weighting": config.class_weighting,
        "early_stopping": {
            "monitor": config.early_stopping.monitor,
            "patience": config.early_stopping.patience,
        },
        "feature_cache_root": config.feature_cache_root,
        "checkpoint_dir": config.checkpoint_dir,
        "log_dir": config.log_dir,
    }


if __name__ == "__main__":
    raise SystemExit(main())
