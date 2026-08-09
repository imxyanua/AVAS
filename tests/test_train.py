import csv
import json

import pytest
import torch
from torch.utils.data import DataLoader

from src.features.cache_index import build_dataset, clips_for_split, read_feature_index
from src.models.cnn_lstm import TemporalClassifier
from src.training import train as train_module
from src.training.train import EpochResult, main, run_epoch, train_model, write_history
from src.utils.config import EarlyStoppingConfig, ModelConfig, TrainingConfig


def cli_args(feature_cache, *extra):
    return [
        "--dataset-config",
        str(feature_cache["dataset_config"]),
        "--model-config",
        str(feature_cache["model_config"]),
        "--training-config",
        str(feature_cache["training_config"]),
        "--feature-index",
        str(feature_cache["index"]),
        *extra,
    ]


def read_history(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_loaders(feature_cache, split="train", batch_size=4):
    clips = clips_for_split(read_feature_index(feature_cache["index"]), split)
    dataset = build_dataset(clips, expected_dim=feature_cache["feature_dim"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_training_writes_checkpoint_history_and_summary(feature_cache):
    exit_code = main(cli_args(feature_cache, "--run-name", "unit"))

    checkpoint = feature_cache["checkpoint_dir"] / "unit.pt"
    run_dir = feature_cache["log_dir"] / "unit"
    history = read_history(run_dir / "history.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert checkpoint.is_file()
    assert len(history) == 4
    assert summary["best_epoch"] >= 1
    assert summary["monitor"] == "val_macro_f1"
    assert summary["train_clips"] == feature_cache["split_sizes"]["train"]
    assert summary["validation_clips"] == feature_cache["split_sizes"]["validation"]
    assert summary["feature_dim"] == feature_cache["feature_dim"]


def test_checkpoint_records_what_is_needed_to_rebuild_the_model(feature_cache):
    main(cli_args(feature_cache, "--run-name", "unit"))

    payload = torch.load(
        feature_cache["checkpoint_dir"] / "unit.pt", map_location="cpu", weights_only=True
    )

    assert payload["classes"] == list(feature_cache["classes"])
    assert payload["feature_dim"] == feature_cache["feature_dim"]
    assert ModelConfig.from_mapping(payload["model_config"]).temporal.hidden_size == 8
    assert "model_state" in payload


def test_training_reduces_the_loss_on_separable_features(feature_cache):
    main(cli_args(feature_cache, "--run-name", "unit", "--epochs", "8"))

    history = read_history(feature_cache["log_dir"] / "unit" / "history.csv")

    assert float(history[-1]["train_loss"]) < float(history[0]["train_loss"])
    assert float(history[-1]["val_macro_f1"]) >= float(history[0]["val_macro_f1"])
    assert float(history[-1]["val_macro_f1"]) > 0.8


def test_command_line_overrides_replace_config_values(feature_cache):
    main(cli_args(feature_cache, "--run-name", "override", "--epochs", "2", "--batch-size", "2"))

    history = read_history(feature_cache["log_dir"] / "override" / "history.csv")

    assert len(history) == 2


def test_missing_split_is_reported(feature_cache):
    with pytest.raises(Exception, match="no cached clips for split"):
        main(cli_args(feature_cache, "--val-split", "holdout"))


def test_run_epoch_updates_weights_only_when_training(feature_cache):
    loader = build_loaders(feature_cache)
    model = TemporalClassifier(feature_cache["feature_dim"], 2, ModelConfig.from_mapping({}))
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    before = model.classifier.weight.detach().clone()

    loss, targets, predictions = run_epoch(model, loader, criterion, device)

    assert torch.equal(model.classifier.weight, before)
    assert len(targets) == len(predictions) == feature_cache["split_sizes"]["train"]
    assert loss > 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    run_epoch(model, loader, criterion, device, optimizer=optimizer)

    assert not torch.equal(model.classifier.weight, before)


def test_early_stopping_keeps_the_best_epoch(feature_cache, monkeypatch, tmp_path):
    val_losses = iter([1.0, 0.9, 0.95, 0.96, 0.97])

    def fake_run_epoch(
        model, loader, criterion, device, *, optimizer=None, gradient_clip_norm=None
    ):
        if optimizer is not None:
            return 0.1, [0, 1], [0, 1]
        return next(val_losses), [0, 1], [0, 1]

    monkeypatch.setattr(train_module, "run_epoch", fake_run_epoch)

    training = TrainingConfig(
        epochs=10,
        device="cpu",
        early_stopping=EarlyStoppingConfig(monitor="val_loss", patience=2),
    )
    model = TemporalClassifier(feature_cache["feature_dim"], 2, ModelConfig.from_mapping({}))
    loader = build_loaders(feature_cache)
    checkpoint = tmp_path / "early.pt"

    run = train_model(
        model,
        loader,
        loader,
        feature_cache["classes"],
        training,
        torch.device("cpu"),
        checkpoint_path=checkpoint,
    )

    assert run.stopped_early is True
    assert len(run.history) == 4
    assert run.best_epoch == 2
    assert run.best_value == pytest.approx(0.9)
    assert torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"] == 2


def test_unsupported_monitor_is_rejected(feature_cache, tmp_path):
    training = TrainingConfig(
        device="cpu", early_stopping=EarlyStoppingConfig(monitor="val_precision")
    )
    model = TemporalClassifier(feature_cache["feature_dim"], 2, ModelConfig.from_mapping({}))
    loader = build_loaders(feature_cache)

    with pytest.raises(ValueError, match="unsupported monitor"):
        train_model(
            model,
            loader,
            loader,
            feature_cache["classes"],
            training,
            torch.device("cpu"),
            checkpoint_path=tmp_path / "unused.pt",
        )


def test_epoch_result_exposes_the_monitored_value():
    result = EpochResult(epoch=1, train_loss=0.4, val_loss=0.5, val_accuracy=0.8, val_macro_f1=0.7)

    assert result.monitored("val_loss") == 0.5
    assert result.monitored("val_accuracy") == 0.8
    assert result.monitored("val_macro_f1") == 0.7
    with pytest.raises(ValueError, match="unknown monitor"):
        result.monitored("val_recall")


def test_write_history_stores_one_row_per_epoch(tmp_path):
    run = train_module.TrainingRun(
        history=[
            EpochResult(epoch=1, train_loss=0.5, val_loss=0.6, val_accuracy=0.5, val_macro_f1=0.4),
            EpochResult(epoch=2, train_loss=0.3, val_loss=0.4, val_accuracy=0.7, val_macro_f1=0.6),
        ],
        best_epoch=2,
        best_value=0.6,
        monitor="val_macro_f1",
        checkpoint_path=tmp_path / "run.pt",
        stopped_early=False,
    )

    path = write_history(run, tmp_path / "history.csv")
    rows = read_history(path)

    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[1]["val_macro_f1"] == "0.6"
